"""
FastAPI backend serving transaction/SAR data to the mobile app,
and providing an endpoint to run new transactions through the full pipeline.

Run from project root:
    uvicorn backend.main:app --reload
"""

import sqlite3
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.pipeline import run_full_pipeline

DB_PATH = Path("data/aml_sar.db")

app = FastAPI(title="AML SAR Assistant API")

# Allow the mobile app (running on a different host/port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/")
def root():
    return {"status": "AML SAR Assistant API is running"}


@app.get("/transactions")
def list_transactions(limit: int = 50, flagged_only: bool = True):
    conn = get_db()
    query = "SELECT * FROM transactions"
    if flagged_only:
        query += " WHERE predicted_fraud = 1"
    query += f" LIMIT {limit}"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.get("/transactions/{transaction_id}")
def get_transaction_detail(transaction_id: int):
    conn = get_db()
    txn = conn.execute(
        "SELECT * FROM transactions WHERE id = ?", (transaction_id,)
    ).fetchone()
    if not txn:
        conn.close()
        raise HTTPException(status_code=404, detail="Transaction not found")

    sar = conn.execute(
        "SELECT * FROM sar_reports WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    conn.close()

    result = dict(txn)
    if sar:
        sar_dict = dict(sar)
        sar_dict["retrieved_sources"] = json.loads(sar_dict["retrieved_sources"])
        result["sar_report"] = sar_dict
    else:
        result["sar_report"] = None

    return result


@app.get("/knowledge-base/log")
def get_gap_filling_log():
    conn = get_db()
    rows = conn.execute("SELECT * FROM knowledge_base_log").fetchall()
    conn.close()
    return [dict(row) for row in rows]


class TransactionInput(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    is_transfer: bool


@app.post("/analyze")
def analyze_transaction(txn: TransactionInput):
    """
    Runs a newly submitted transaction through the full pipeline:
    feature engineering -> model prediction -> RAG retrieval -> LLM SAR draft.
    """
    result = run_full_pipeline(txn.dict())
    return result