"""
Feature engineering + Isolation Forest anomaly detection on PaySim data.
Filters to TRANSFER/CASH_OUT only (per EDA finding: 100% of fraud lives here).

Run from project root:
    python scripts/feature_engineering_and_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

DATA_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
OUT_DIR = Path("data/model_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_filter():
    print("Loading full dataset...")
    df = pd.read_csv(DATA_PATH)

    # EDA finding: fraud only exists in TRANSFER and CASH_OUT
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    print(f"Filtered to TRANSFER/CASH_OUT: {len(df):,} rows "
          f"({df['isFraud'].sum():,} fraud cases retained)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features based on known PaySim fraud signatures:
    - balance mismatches (fraud often doesn't update balances 'correctly')
    - whether destination balance is suspiciously zero
    - ratio of transaction amount to origin balance (draining behavior)
    """
    df = df.copy()

    # Expected new balance if the transaction behaved "normally"
    df["expected_new_orig_balance"] = df["oldbalanceOrg"] - df["amount"]
    df["orig_balance_error"] = df["expected_new_orig_balance"] - df["newbalanceOrig"]

    df["expected_new_dest_balance"] = df["oldbalanceDest"] + df["amount"]
    df["dest_balance_error"] = df["expected_new_dest_balance"] - df["newbalanceDest"]

    # Suspicious zero-balance patterns (known fraud fingerprint)
    df["dest_balance_stayed_zero"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)
    ).astype(int)

    df["orig_balance_drained"] = (df["newbalanceOrig"] == 0).astype(int)

    # Ratio features (avoid divide-by-zero)
    df["amount_to_orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # One-hot for transaction type (only 2 categories now)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)

    return df


FEATURE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_error",
    "dest_balance_error",
    "dest_balance_stayed_zero",
    "orig_balance_drained",
    "amount_to_orig_balance_ratio",
    "is_transfer",
]


def train_and_evaluate(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y_true = df["isFraud"]

    print("\nTraining Isolation Forest...")
    # contamination = expected proportion of anomalies; we use the real fraud
    # rate within this filtered subset as a reasonable estimate
    contamination = y_true.mean()
    print(f"Using contamination = {contamination:.4f} (actual fraud rate in filtered data)")

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,  # use all CPU cores, still laptop-friendly
    )
    model.fit(X)

    # IsolationForest outputs -1 for anomaly, 1 for normal — convert to 0/1 matching isFraud
    raw_preds = model.predict(X)
    y_pred = (raw_preds == -1).astype(int)

    print("\n" + "=" * 50)
    print("EVALUATION (Isolation Forest vs ground truth isFraud)")
    print("=" * 50)
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nFull report:")
    print(classification_report(y_true, y_pred, target_names=["legit", "fraud"]))

    print("\nBaseline to beat — PaySim's own isFlaggedFraud recall was ~0.2%")

    # Save flagged transactions for the next step (RAG + LLM layer)
    df["predicted_fraud"] = y_pred
    df["anomaly_score"] = model.decision_function(X)  # lower = more anomalous
    flagged = df[df["predicted_fraud"] == 1]
    flagged.to_csv(OUT_DIR / "flagged_transactions.csv", index=False)
    print(f"\nSaved {len(flagged):,} flagged transactions to {OUT_DIR}/flagged_transactions.csv")

    return model, df


def main():
    df = load_and_filter()
    df = engineer_features(df)
    model, df_scored = train_and_evaluate(df)


if __name__ == "__main__":
    main()