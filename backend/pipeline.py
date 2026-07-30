"""
Core pipeline logic, extracted so it can be called live by the FastAPI
backend for the /analyze endpoint.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import time
import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

MODEL_PATH = Path("data/model_outputs/random_forest_model.joblib")
CHROMA_DIR = Path("data/knowledge_base/chroma_store")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "llama-3.3-70b-versatile"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(name="aml_typologies", embedding_function=embedding_fn)

rf_model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

FEATURE_COLUMNS = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "orig_balance_error", "dest_balance_error", "dest_balance_stayed_zero",
    "orig_balance_drained", "amount_to_orig_balance_ratio", "is_transfer",
]


def engineer_features(txn: dict) -> dict:
    features = dict(txn)
    features["orig_balance_error"] = (
        (txn["oldbalanceOrg"] - txn["amount"]) - txn["newbalanceOrig"]
    )
    features["dest_balance_error"] = (
        (txn["oldbalanceDest"] + txn["amount"]) - txn["newbalanceDest"]
    )
    features["dest_balance_stayed_zero"] = int(
        txn["oldbalanceDest"] == 0 and txn["newbalanceDest"] == 0 and txn["amount"] > 0
    )
    features["orig_balance_drained"] = int(txn["newbalanceOrig"] == 0)
    features["amount_to_orig_balance_ratio"] = txn["amount"] / (txn["oldbalanceOrg"] + 1)
    features["is_transfer"] = int(txn.get("is_transfer", False))
    return features


def predict_fraud(features: dict) -> tuple[int, float]:
    if rf_model is None:
        return 1, 0.5
    row = pd.DataFrame([{k: features[k] for k in FEATURE_COLUMNS}])
    pred = rf_model.predict(row)[0]
    prob = rf_model.predict_proba(row)[0][1]
    return int(pred), float(prob)


def call_llm(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    time.sleep(1)
    return response.choices[0].message.content


def build_retrieval_query(features: dict) -> str:
    if features["orig_balance_drained"] and features["dest_balance_stayed_zero"]:
        return ("malware credential theft sudden wire transfer unusual account "
                "activity account takeover unauthorized access")
    elif features["orig_balance_drained"]:
        return ("rapid full account drain immediate large withdrawal after "
                "account access account takeover")
    else:
        return ("suspicious activity report large transaction financial "
                "institution monitoring red flags")


def retrieve_with_neighbors(query: str, n_results: int = 6, expand_top_n: int = 2) -> list[dict]:
    results = collection.query(query_texts=[query], n_results=n_results)
    seen_keys = set()
    expanded = []
    for rank, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        source, idx = meta["source"], meta["chunk_index"]
        key = (source, idx)
        if key not in seen_keys:
            seen_keys.add(key)
            expanded.append({"source": source, "text": doc, "distance": dist, "chunk_index": idx})
        if rank < expand_top_n:
            for neighbor_idx in [idx - 1, idx + 1]:
                nk = (source, neighbor_idx)
                if nk in seen_keys:
                    continue
                nr = collection.get(where={"$and": [{"source": source}, {"chunk_index": neighbor_idx}]})
                if nr["ids"]:
                    seen_keys.add(nk)
                    expanded.append({"source": source, "text": nr["documents"][0],
                                      "distance": None, "chunk_index": neighbor_idx})
    expanded.sort(key=lambda x: (x["source"], x["chunk_index"]))
    return expanded


def run_full_pipeline(txn: dict) -> dict:
    features = engineer_features(txn)
    pred, prob = predict_fraud(features)

    summary_prompt = f"""You are a financial crime analyst. Describe this transaction
in plain, professional English (3-4 sentences). Only describe what is in the data.

Amount: {features['amount']:.2f}
Origin balance before/after: {features['oldbalanceOrg']:.2f} / {features['newbalanceOrig']:.2f}
Destination balance before/after: {features['oldbalanceDest']:.2f} / {features['newbalanceDest']:.2f}
Destination balance stayed zero: {bool(features['dest_balance_stayed_zero'])}
Origin fully drained: {bool(features['orig_balance_drained'])}
"""
    summary = call_llm(summary_prompt)

    retrieval_query = build_retrieval_query(features)
    retrieved = retrieve_with_neighbors(retrieval_query)

    def fmt(c):
        label = f"relevance_distance: {c['distance']:.4f}" if c["distance"] is not None else "neighboring context (unscored)"
        return f"[Source: {c['source']} | {label}]\n{c['text']}"
    context_text = "\n\n".join(fmt(c) for c in retrieved)

    typology_prompt = f"""Classify this transaction as MONEY LAUNDERING TYPOLOGY or
ACCOUNT TAKEOVER / UNAUTHORIZED ACCESS FRAUD, based only on the excerpts below.
Do not invent details not present in the excerpts.

TRANSACTION SUMMARY: {summary}
EXCERPTS: {context_text}
"""
    typology = call_llm(typology_prompt)

    sar_prompt = f"""Draft a formal SAR narrative with headers: Transaction Overview,
Red Flags Identified, Typology Match, Recommended Action.

SUMMARY: {summary}
TYPOLOGY ANALYSIS: {typology}
"""
    sar_draft = call_llm(sar_prompt)

    return {
        "predicted_fraud": pred,
        "fraud_probability": prob,
        "summary": summary,
        "retrieval_query": retrieval_query,
        "typology_analysis": typology,
        "sar_draft": sar_draft,
        "retrieved_sources": list({c["source"] for c in retrieved}),
    }