"""
Retrieval comparison test: standard top-k retrieval vs. neighbor-expanded
retrieval (pulls in adjacent chunks around each match to avoid losing
content split across a chunk boundary).

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


def retrieve_standard(query: str, n_results: int = N_RESULTS) -> list[dict]:
    """Original method: just the top-k matches, no neighbors."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return [
        {"source": meta["source"], "text": doc, "distance": dist, "chunk_index": meta["chunk_index"]}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def retrieve_with_neighbors(query: str, n_results: int = N_RESULTS) -> list[dict]:
    """New method: top-k matches PLUS each match's immediate neighbor chunks."""
    results = collection.query(query_texts=[query], n_results=n_results)

    seen_keys = set()
    expanded = []

    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        source = meta["source"]
        idx = meta["chunk_index"]
        key = (source, idx)
        if key not in seen_keys:
            seen_keys.add(key)
            expanded.append({"source": source, "text": doc, "distance": dist, "chunk_index": idx})

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
                    "source": source,
                    "text": neighbor_results["documents"][0],
                    "distance": None,
                    "chunk_index": neighbor_idx,
                })

    expanded.sort(key=lambda x: (x["source"], x["chunk_index"]))
    return expanded


def print_results(label: str, results: list[dict]):
    print(f"\n--- {label} ({len(results)} chunks) ---")
    for r in results:
        dist_str = f"{r['distance']:.4f}" if r["distance"] is not None else "N/A (neighbor)"
        print(f"\n[source={r['source']} | chunk_index={r['chunk_index']} | distance={dist_str}]")
        print(r["text"][:250].replace("\n", " "))


def compare_query(query: str):
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    standard = retrieve_standard(query)
    expanded = retrieve_with_neighbors(query)

    print_results("STANDARD (top-k only)", standard)
    print_results("WITH NEIGHBORS (expanded)", expanded)

    print(f"\n>>> Standard returned {len(standard)} chunks, expanded returned {len(expanded)} chunks")
    print(f">>> Extra chunks added by neighbor expansion: {len(expanded) - len(standard)}")
    print()


def main():
    test_queries = [
        "malware credential theft sudden wire transfer unusual account activity account takeover unauthorized access",
        "rapid full account drain immediate large withdrawal after account access account takeover",
    ]
    for query in test_queries:
        compare_query(query)


if __name__ == "__main__":
    main()