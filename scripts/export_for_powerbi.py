"""
Exports clean, Power BI-ready CSVs from the SQLite database.

Run from project root:
    python scripts/export_for_powerbi.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/aml_sar.db")
OUT_DIR = Path("data/powerbi_exports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    conn = sqlite3.connect(DB_PATH)

    # Transactions with a readable fraud pattern label for easier charting
    transactions = pd.read_sql("SELECT * FROM transactions", conn)
    transactions["pattern_type"] = transactions.apply(
        lambda r: "Account Drain + Zero Destination" if r["dest_balance_stayed_zero"] == 1
        else "Account Drain + Balance Mismatch" if r["orig_balance_drained"] == 1
        else "Other",
        axis=1,
    )
    transactions.to_csv(OUT_DIR / "transactions.csv", index=False)

    # SAR reports, with a parsed-out category and match strength column
    sar = pd.read_sql("SELECT * FROM sar_reports", conn)
    sar["category"] = sar["typology_analysis"].apply(
        lambda t: "Account Takeover" if "ACCOUNT TAKEOVER" in t
        else "Money Laundering" if "MONEY LAUNDERING" in t
        else "Unclear"
    )
    sar["match_strength"] = sar["typology_analysis"].apply(
        lambda t: "Weak" if "weak" in t.lower()
        else "Strong" if "strong" in t.lower()
        else "Moderate"
    )
    sar.to_csv(OUT_DIR / "sar_reports.csv", index=False)

    # Knowledge base log
    kb_log = pd.read_sql("SELECT * FROM knowledge_base_log", conn)
    kb_log.to_csv(OUT_DIR / "knowledge_base_log.csv", index=False)

    conn.close()
    print(f"Exported 3 CSVs to {OUT_DIR}/")
    print(f"  transactions.csv: {len(transactions)} rows")
    print(f"  sar_reports.csv: {len(sar)} rows")
    print(f"  knowledge_base_log.csv: {len(kb_log)} rows")


if __name__ == "__main__":
    main()