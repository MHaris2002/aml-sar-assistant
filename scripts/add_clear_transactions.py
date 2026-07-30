"""
Samples legitimate (non-fraud) transactions from the raw PaySim dataset,
runs them through the same feature engineering, and adds them to the
transactions table so the dashboard can show a realistic mix of
flagged and clear transactions.

Run from project root:
    python scripts/add_clear_transactions.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
DB_PATH = Path("data/aml_sar.db")
N_SAMPLES = 40


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["orig_balance_error"] = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
    df["dest_balance_error"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    df["dest_balance_stayed_zero"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)
    ).astype(int)
    df["orig_balance_drained"] = (df["newbalanceOrig"] == 0).astype(int)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["predicted_fraud"] = 0
    return df


def main():
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Sample clean, believable legitimate transactions:
    # - not fraud
    # - reasonable amount range (avoid tiny/weird outliers)
    # - origin account NOT drained (so it looks like a normal transaction)
    legit = df[
        (df["isFraud"] == 0)
        & (df["amount"] > 100)
        & (df["amount"] < 50000)
        & (df["newbalanceOrig"] > 0)
    ].sample(n=N_SAMPLES, random_state=42)

    legit = engineer_features(legit)

    conn = sqlite3.connect(DB_PATH)

    # Find the current max id so we don't collide with existing flagged transactions
    max_id = conn.execute("SELECT MAX(id) FROM transactions").fetchone()[0] or 0

    cols = [
        "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
        "newbalanceDest", "orig_balance_error", "dest_balance_error",
        "dest_balance_stayed_zero", "orig_balance_drained", "is_transfer",
        "predicted_fraud",
    ]

    insert_df = legit[cols].reset_index(drop=True)
    insert_df["id"] = range(max_id + 1, max_id + 1 + len(insert_df))

    insert_df.to_sql("transactions", conn, if_exists="append", index=False)
    conn.close()

    print(f"Added {len(insert_df)} clear transactions to the database")
    print(f"New ID range: {max_id + 1} to {max_id + len(insert_df)}")


if __name__ == "__main__":
    main()