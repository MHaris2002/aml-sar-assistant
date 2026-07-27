"""
Gap-filling ingestion pipeline: reviews weak-match SAR results, searches for
better authoritative source documents (restricted to FinCEN/FATF/regulatory
domains), fetches and chunks them, and adds them to the knowledge base.

This is a SEPARATE, offline/periodic job - never called during live
orchestration. Run manually or on a schedule after reviewing a batch of
results.

Run from project root:
    python scripts/gap_filling_search.py
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from tavily import TavilyClient
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

SAR_RESULTS_PATH = Path("data/sar_outputs/sample_sar_results.json")
DOCS_DIR = Path("data/knowledge_base/aml_docs")
CHROMA_DIR = Path("data/knowledge_base/chroma_store")
LOG_PATH = Path("data/knowledge_base/gap_filling_log.json")

# Only search these trusted domains - never ingest random web content
TRUSTED_DOMAINS = ["fincen.gov", "fatf-gafi.org", "occ.gov", "federalreserve.gov"]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 250


def find_weak_matches() -> list[dict]:
    """Identify SAR results where the typology match was weak."""
    data = json.loads(SAR_RESULTS_PATH.read_text())
    return [r for r in data if "weak" in r["typology_analysis"].lower()]


def build_search_query(weak_case: dict) -> str:
    """
    Turn a weak-match case into a plain search query. Domain restriction
    is handled entirely by Tavily's include_domains parameter, not by
    site: operators in the query text.
    """
    return weak_case["retrieval_query"]


def search_for_documents(query: str, max_results: int = 3) -> list[dict]:
    print(f"  Searching: {query}")
    print(f"  Restricted to domains: {TRUSTED_DOMAINS}")
    results = tavily.search(
        query=query,
        max_results=max_results,
        include_domains=TRUSTED_DOMAINS,
    )
    result_list = results.get("results", [])
    for r in result_list:
        print(f"    -> {r.get('url', 'NO URL')}")
    return result_list


def download_pdf(url: str, save_path: Path) -> bool:
    """Attempt to download a PDF from a URL. Returns True if successful."""
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200 and response.headers.get("content-type", "").lower().startswith("application/pdf"):
            save_path.write_bytes(response.content)
            return True
    except Exception as e:
        print(f"    Failed to download {url}: {e}")
    return False


def extract_text_chunks(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    full_text = ""
    for page in reader.pages:
        full_text += (page.extract_text() or "") + "\n"

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + CHUNK_SIZE
        chunk = full_text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def already_in_knowledge_base(filename: str) -> bool:
    return (DOCS_DIR / filename).exists()


def ingest_document(pdf_path: Path, collection):
    chunks = extract_text_chunks(pdf_path)
    ids = [f"{pdf_path.stem}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": pdf_path.name, "chunk_index": i, "auto_ingested": True}
        for i in range(len(chunks))
    ]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"    Ingested {len(chunks)} chunks from {pdf_path.name}")
    return len(chunks)


def main():
    weak_cases = find_weak_matches()
    print(f"Found {len(weak_cases)} weak-match case(s) to investigate\n")

    if not weak_cases:
        print("No weak matches found - nothing to search for.")
        return

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="aml_typologies", embedding_function=embedding_fn
    )

    log = []

    for case in weak_cases:
        print(f"\n{'='*60}")
        print(f"Transaction {case['transaction_index']}")
        print(f"{'='*60}")

        query = build_search_query(case)
        search_results = search_for_documents(query)

        if not search_results:
            print("  No results found in trusted domains.")
            log.append({"transaction_index": case["transaction_index"], "query": query, "found": False})
            continue

        for result in search_results:
            url = result.get("url", "")
            print(f"  Found: {url}")

            if not url.lower().endswith(".pdf"):
                print("    Skipping - not a direct PDF link")
                continue

            filename = url.split("/")[-1]
            if already_in_knowledge_base(filename):
                print(f"    Already have {filename} - skipping")
                continue

            save_path = DOCS_DIR / filename
            if download_pdf(url, save_path):
                num_chunks = ingest_document(save_path, collection)
                log.append({
                    "transaction_index": case["transaction_index"],
                    "query": query,
                    "found": True,
                    "ingested_document": filename,
                    "chunks_added": num_chunks,
                })
            time.sleep(1)

    LOG_PATH.write_text(json.dumps(log, indent=2))
    print(f"\n\nGap-filling run complete. Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()