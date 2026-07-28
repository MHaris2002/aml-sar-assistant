"""
Consolidates existing CSV/JSON outputs into a single SQLite database,
which becomes the shared data source for the FastAPI backend, Power BI,
and eventually the mobile app.

Run from project root:
    python scripts/build_database.py
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/aml_sar.db")
FLAGGED_CSV = Path("data/model_outputs/flagged_transactions_supervised.csv")
SAR_RESULTS_JSON = Path("data/sar_outputs/sample_sar_results.json")
GAP_LOG_JSON = Path("data/knowledge_base/gap_filling_log.json")


def create_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            amount REAL,
            oldbalanceOrg REAL,
            newbalanceOrig REAL,
            oldbalanceDest REAL,
            newbalanceDest REAL,
            orig_balance_error REAL,
            dest_balance_error REAL,
            dest_balance_stayed_zero INTEGER,
            orig_balance_drained INTEGER,
            is_transfer INTEGER,
            predicted_fraud INTEGER
        );

        CREATE TABLE IF NOT EXISTS sar_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER,
            summary TEXT,
            retrieval_query TEXT,
            typology_analysis TEXT,
            sar_draft TEXT,
            retrieved_sources TEXT,
            FOREIGN KEY (transaction_id) REFERENCES transactions (id)
        );

        CREATE TABLE IF NOT EXISTS knowledge_base_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_index INTEGER,
            query TEXT,
            found INTEGER,
            ingested_document TEXT,
            chunks_added INTEGER
        );
    """)


def load_transactions(conn):
    df = pd.read_csv(FLAGGED_CSV)
    df = df.reset_index().rename(columns={"index": "id"})
    cols = [
        "id", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
        "newbalanceDest", "orig_balance_error", "dest_balance_error",
        "dest_balance_stayed_zero", "orig_balance_drained", "is_transfer",
        "predicted_fraud",
    ]
    df[cols].to_sql("transactions", conn, if_exists="replace", index=False)
    print(f"Loaded {len(df)} transactions")


def load_sar_reports(conn):
    if not SAR_RESULTS_JSON.exists():
        print("No SAR results found - skipping")
        return
    data = json.loads(SAR_RESULTS_JSON.read_text())
    cursor = conn.cursor()
    for r in data:
        cursor.execute(
            """INSERT INTO sar_reports
               (transaction_id, summary, retrieval_query, typology_analysis, sar_draft, retrieved_sources)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                r["transaction_index"],
                r["summary"],
                r.get("retrieval_query", ""),
                r["typology_analysis"],
                r["sar_draft"],
                json.dumps(r.get("retrieved_sources", [])),
            ),
        )
    conn.commit()
    print(f"Loaded {len(data)} SAR reports")


def load_gap_filling_log(conn):
    if not GAP_LOG_JSON.exists():
        print("No gap-filling log found - skipping")
        return
    data = json.loads(GAP_LOG_JSON.read_text())
    cursor = conn.cursor()
    for entry in data:
        cursor.execute(
            """INSERT INTO knowledge_base_log
               (transaction_index, query, found, ingested_document, chunks_added)
               VALUES (?, ?, ?, ?, ?)""",
            (
                entry["transaction_index"],
                entry["query"],
                int(entry["found"]),
                entry.get("ingested_document", ""),
                entry.get("chunks_added", 0),
            ),
        )
    conn.commit()
    print(f"Loaded {len(data)} gap-filling log entries")


def main():
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    load_transactions(conn)
    load_sar_reports(conn)
    load_gap_filling_log(conn)
    conn.close()
    print(f"\nDatabase built at {DB_PATH}")


if __name__ == "__main__":
    main()