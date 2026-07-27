"""
3-role LLM orchestration for AML SAR drafting, using Groq (fast, free-tier friendly).

1. Summarizer      - turns raw transaction data into plain-language description
2. Typology matcher - retrieves relevant AML typology text (RAG, using red-flag
                       phrased queries rather than raw technical language) and
                       classifies into Money Laundering vs Account Takeover
3. SAR drafter     - produces a structured Suspicious Activity Report

Key design decision: retrieval queries use real-world red-flag language
(e.g. "account takeover", "credential theft") rather than PaySim's raw
technical balance-error terminology, since regulatory documents describe
fraud behaviorally, not in ledger-arithmetic terms. This was empirically
confirmed in test_retrieval.py - technical phrasing never surfaced the
FinCEN account-takeover documents, while red-flag phrasing did consistently.

Run from project root:
    python scripts/llm_orchestration.py
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

CHROMA_DIR = Path("data/knowledge_base/chroma_store")
FLAGGED_CSV = Path("data/model_outputs/flagged_transactions_supervised.csv")
OUT_DIR = Path("data/sar_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# How many flagged transactions to process this run
NUM_TRANSACTIONS = 25

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(name="aml_typologies", embedding_function=embedding_fn)


def call_llm(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature reduces invented/hallucinated detail
    )
    time.sleep(2)  # small courtesy pause
    return response.choices[0].message.content


def summarize_transaction(row: pd.Series) -> str:
    prompt = f"""You are a financial crime analyst. Describe the following transaction
in plain, professional English, highlighting anything unusual. Be factual and concise
(3-4 sentences max). Only describe what is in the data below - do not speculate about
causes or intent.

Transaction details:
- Amount: {row['amount']:.2f}
- Origin balance before: {row['oldbalanceOrg']:.2f}
- Origin balance after: {row['newbalanceOrig']:.2f}
- Destination balance before: {row['oldbalanceDest']:.2f}
- Destination balance after: {row['newbalanceDest']:.2f}
- Balance error (origin): {row['orig_balance_error']:.2f}
- Balance error (destination): {row['dest_balance_error']:.2f}
- Destination balance stayed zero despite transaction: {bool(row['dest_balance_stayed_zero'])}
- Origin account fully drained: {bool(row['orig_balance_drained'])}
- Transaction type: {'TRANSFER' if row['is_transfer'] == 1 else 'CASH_OUT'}
"""
    return call_llm(prompt)


def build_retrieval_query(row: pd.Series) -> str:
    if row['orig_balance_drained'] and row['dest_balance_stayed_zero']:
        return ("malware credential theft sudden wire transfer unusual account "
                "activity account takeover unauthorized access")
    elif row['orig_balance_drained']:
        return ("rapid full account drain immediate large withdrawal after "
                "account access account takeover")
    else:
        return ("suspicious activity report large transaction financial "
                "institution monitoring red flags")


def retrieve_typology_context(query: str, n_results: int = 6) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=n_results)
    return [
        {"source": meta["source"], "text": doc, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

def retrieve_with_neighbors(query: str, n_results: int = 6, expand_top_n: int = 2) -> list[dict]:
    """
    Retrieves top-k matches. Only expands neighbors (adjacent chunk_index,
    same source) for the top `expand_top_n` strongest matches, to avoid
    pulling in low-value boilerplate around weaker matches.
    """
    results = collection.query(query_texts=[query], n_results=n_results)

    seen_keys = set()
    expanded = []

    for rank, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        source = meta["source"]
        idx = meta["chunk_index"]
        key = (source, idx)
        if key not in seen_keys:
            seen_keys.add(key)
            expanded.append({"source": source, "text": doc, "distance": dist, "chunk_index": idx})

        if rank < expand_top_n:
            for neighbor_idx in [idx - 1, idx + 1]:
                neighbor_key = (source, neighbor_idx)
                if neighbor_key in seen_keys:
                    continue
                neighbor_results = collection.get(
                    where={"$and": [{"source": source}, {"chunk_index": neighbor_idx}]}
                )
                if neighbor_results["ids"]:
                    seen_keys.add(neighbor_key)
                    expanded.append({
                        "source": source, "text": neighbor_results["documents"][0],
                        "distance": None, "chunk_index": neighbor_idx,
                    })

    expanded.sort(key=lambda x: (x["source"], x["chunk_index"]))
    return expanded

def match_typology(transaction_summary: str, retrieved_chunks: list[dict]) -> str:
    def format_chunk(c):
        if c["distance"] is not None:
            label = f"relevance_distance: {c['distance']:.4f}"
        else:
            label = "neighboring context chunk (not independently scored)"
        return f"[Source: {c['source']} | {label}]\n{c['text']}"

    context_text = "\n\n".join(format_chunk(c) for c in retrieved_chunks)
    prompt = f"""You are an AML/fraud compliance analyst. Below is a transaction summary
and excerpts from official regulatory documents (FinCEN and FATF).

Classify this transaction into ONE of these two categories, based on which the
retrieved evidence better supports:

1. MONEY LAUNDERING TYPOLOGY - network-style patterns such as layering, structuring,
   alternative remittance systems, or multi-party fund pooling.
2. ACCOUNT TAKEOVER / UNAUTHORIZED ACCESS FRAUD - a single compromised account being
   rapidly drained by someone who gained unauthorized access, often via credential theft,
   malware, or social engineering.

CRITICAL RULES:
- Only reference facts, cases, entities, countries, or details that appear VERBATIM in
  the excerpts below. Do NOT invent company names, countries, case numbers, or any
  specific detail not present in the text provided.
- If the excerpts do not contain enough specific detail to support either category
  confidently, say so explicitly rather than adding plausible-sounding detail.
- Quote or closely paraphrase only what is actually written in the excerpts as support.

TRANSACTION SUMMARY:
{transaction_summary}

RETRIEVED DOCUMENT EXCERPTS:
{context_text}

State clearly which category the evidence supports, cite the specific source
document(s) that justify your classification, and explicitly note if the match is
weak even after considering both categories. 3-4 sentences.
"""
    return call_llm(prompt)


def draft_sar(transaction_summary: str, typology_analysis: str) -> str:
    prompt = f"""You are drafting a Suspicious Activity Report (SAR) narrative section.
Use a formal, structured tone appropriate for regulatory submission. Base it ONLY on
the information provided below - do not invent details, entities, or cases not
mentioned in the inputs.

TRANSACTION SUMMARY:
{transaction_summary}

TYPOLOGY ANALYSIS:
{typology_analysis}

Structure your response with these headers:
- Transaction Overview
- Red Flags Identified
- Typology Match
- Recommended Action (e.g., file SAR, escalate for manual review, no action needed)
"""
    return call_llm(prompt)


def main():
    flagged = pd.read_csv(FLAGGED_CSV)
    print(f"Loaded {len(flagged)} flagged transactions")

    sample = flagged.head(NUM_TRANSACTIONS)

    results = []
    for idx, row in sample.iterrows():
        print(f"\n{'='*70}\nProcessing transaction {idx}\n{'='*70}")

        print("Step 1: Summarizing...")
        summary = summarize_transaction(row)
        print(summary)

        print("\nStep 2: Retrieving typology context + matching...")
        retrieval_query = build_retrieval_query(row)
        print(f"[Retrieval query used]: {retrieval_query}")
        retrieved = retrieve_with_neighbors(retrieval_query, expand_top_n=2)
        typology = match_typology(summary, retrieved)
        print(typology)

        print("\nStep 3: Drafting SAR...")
        sar = draft_sar(summary, typology)
        print(sar)

        results.append({
            "transaction_index": int(idx),
            "summary": summary,
            "retrieval_query": retrieval_query,
            "typology_analysis": typology,
            "sar_draft": sar,
            "retrieved_sources": [c["source"] for c in retrieved],
            "retrieved_distances": [c["distance"] for c in retrieved],
        })

    with open(OUT_DIR / "sample_sar_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved {len(results)} results to {OUT_DIR}/sample_sar_results.json")


if __name__ == "__main__":
    main()