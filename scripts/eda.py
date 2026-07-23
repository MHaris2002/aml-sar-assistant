"""
EDA on the PaySim dataset — class imbalance, transaction type breakdown,
fraud patterns by type and time step.

Run from project root:
    python scripts/eda.py
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
OUT_DIR = Path("data/eda_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading dataset (this may take a moment, ~470MB)...")
    df = pd.read_csv(DATA_PATH)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

    # --- Class imbalance ---
    print("=" * 50)
    print("CLASS IMBALANCE")
    print("=" * 50)
    fraud_counts = df["isFraud"].value_counts()
    fraud_pct = df["isFraud"].value_counts(normalize=True) * 100
    print(fraud_counts)
    print(f"\nFraud rate: {fraud_pct[1]:.4f}%")

    # --- Transaction type breakdown ---
    print("\n" + "=" * 50)
    print("TRANSACTION TYPE BREAKDOWN")
    print("=" * 50)
    print(df["type"].value_counts())

    print("\nFraud rate BY transaction type:")
    fraud_by_type = df.groupby("type")["isFraud"].agg(["sum", "count", "mean"])
    fraud_by_type.columns = ["fraud_count", "total_count", "fraud_rate"]
    fraud_by_type["fraud_rate_pct"] = fraud_by_type["fraud_rate"] * 100
    print(fraud_by_type.sort_values("fraud_rate_pct", ascending=False))

    # --- isFlaggedFraud vs isFraud (system's own flag vs ground truth) ---
    print("\n" + "=" * 50)
    print("SYSTEM FLAG vs GROUND TRUTH")
    print("=" * 50)
    print(pd.crosstab(df["isFraud"], df["isFlaggedFraud"], margins=True))
    print("\n(Note: isFlaggedFraud is PaySim's own naive rule — expect it to catch very few)")

    # --- Amount stats by fraud label ---
    print("\n" + "=" * 50)
    print("TRANSACTION AMOUNT STATS BY FRAUD LABEL")
    print("=" * 50)
    print(df.groupby("isFraud")["amount"].describe())

    # --- Time pattern (step = hour) ---
    print("\n" + "=" * 50)
    print("FRAUD DISTRIBUTION OVER TIME (first/last 5 steps with fraud)")
    print("=" * 50)
    fraud_by_step = df[df["isFraud"] == 1].groupby("step").size()
    print(f"Fraud occurs across {fraud_by_step.shape[0]} distinct time steps")
    print(fraud_by_step.describe())

    # --- Save a summary CSV for later reference ---
    fraud_by_type.to_csv(OUT_DIR / "fraud_by_type.csv")
    df.groupby("isFraud")["amount"].describe().to_csv(OUT_DIR / "amount_stats_by_label.csv")
    print(f"\nSaved summary CSVs to {OUT_DIR}/")


if __name__ == "__main__":
    main()