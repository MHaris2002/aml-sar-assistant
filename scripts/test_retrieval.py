"""
Retrieval sanity check for the AML knowledge base.

Tests two kinds of queries side by side:
1. Raw technical/ledger-style phrasing (how PaySim summaries currently read)
2. Real-world red-flag phrasing (how FinCEN/FATF documents actually describe fraud)

Confirms whether the FinCEN account-takeover documents surface with
better-phrased queries.

Run from project root:
    python scripts/test_retrieval.py
"""

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

CHROMA_DIR = Path("data/knowledge_base/chroma_store")
N_RESULTS = 6

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name="aml_typologies", embedding_function=embedding_fn)

test_queries = {
    "Raw/technical phrasing": [
        "cash out completely drained origin account destination balance discrepancy",
        "account fully drained transfer destination balance did not update",
    ],
    "Real-world red-flag phrasing": [
        "sudden complete withdrawal of entire account balance unauthorized access",
        "rapid full account drain immediate large withdrawal after account access account takeover",
        "malware credential theft sudden wire transfer unusual account activity",
    ],
}


def run_query(query: str):
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    results = collection.query(query_texts=[query], n_results=N_RESULTS)

    fincen_hit = False
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        source = meta["source"]
        if "FIN-" in source:
            fincen_hit = True
        print(f"\n[Result {i+1}] source={source} | distance={dist:.4f}")
        print(doc[:300].replace("\n", " "))

    print(f"\n>>> FinCEN document appeared in top {N_RESULTS}: {fincen_hit}")
    print()


def main():
    for category, queries in test_queries.items():
        print(f"\n{'#'*70}")
        print(f"# CATEGORY: {category}")
        print(f"{'#'*70}\n")
        for query in queries:
            run_query(query)


if __name__ == "__main__":
    main()