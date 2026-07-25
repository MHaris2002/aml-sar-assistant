"""
Extracts text from AML typology PDFs and builds a local Chroma vector store
for RAG retrieval. Uses a local sentence-transformers embedding model —
no API key needed for this step.

Run from project root:
    python scripts/build_knowledge_base.py
"""

from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = Path("data/knowledge_base/aml_docs")
CHROMA_DIR = Path("data/knowledge_base/chroma_store")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 800  # characters per chunk — small enough for precise retrieval
CHUNK_OVERLAP = 250


def extract_text_chunks(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    full_text = ""
    for page in reader.pages:
        text = page.extract_text() or ""
        full_text += text + "\n"

    # Simple sliding-window chunking
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + CHUNK_SIZE
        chunk = full_text[start:end].strip()
        if len(chunk) > 50:  # skip near-empty chunks
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def main():
    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DOCS_DIR}. Add some AML typology PDFs first.")
        return

    print(f"Found {len(pdf_files)} PDF(s): {[p.name for p in pdf_files]}")

    # Local embedding model — no API cost, runs on CPU fine for this scale
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name="aml_typologies",
        embedding_function=embedding_fn,
    )

    doc_id_counter = 0
    for pdf_path in pdf_files:
        print(f"\nProcessing {pdf_path.name}...")
        chunks = extract_text_chunks(pdf_path)
        print(f"  Extracted {len(chunks)} chunks")

        ids = [f"{pdf_path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_path.name, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )
        doc_id_counter += len(chunks)

    print(f"\nTotal chunks stored: {doc_id_counter}")
    print(f"Vector store saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()