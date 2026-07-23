"""
Supervised fraud classifier on PaySim (TRANSFER/CASH_OUT only), using the
features engineered in feature_engineering_and_model.py, but now learning
directly from the isFraud labels via a Random Forest.

Run from project root:
    python scripts/supervised_model.py
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
OUT_DIR = Path("data/model_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_filter():
    print("Loading full dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    print(f"Filtered to TRANSFER/CASH_OUT: {len(df):,} rows "
          f"({df['isFraud'].sum():,} fraud cases retained)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["orig_balance_error"] = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
    df["dest_balance_error"] = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
    df["dest_balance_stayed_zero"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0)
    ).astype(int)
    df["orig_balance_drained"] = (df["newbalanceOrig"] == 0).astype(int)
    df["amount_to_orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    return df


FEATURE_COLUMNS = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "orig_balance_error", "dest_balance_error",
    "dest_balance_stayed_zero", "orig_balance_drained",
    "amount_to_orig_balance_ratio", "is_transfer",
]


def main():
    df = load_and_filter()
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS]
    y = df["isFraud"]

    # Stratified split so both sets keep the same fraud proportion
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"Fraud in train: {y_train.sum():,} | Fraud in test: {y_test.sum():,}")

    print("\nTraining Random Forest with balanced class weighting...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",  # directly compensates for the 0.3% imbalance
        random_state=42,
        n_jobs=-1,
        max_depth=12,  # keeps it fast on a laptop, avoids overfitting to noise
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print("EVALUATION (Random Forest, held-out test set)")
    print("=" * 50)
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nFull report:")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

    print("\nFeature importances:")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False))

    # Save flagged transactions from the test set for the RAG/LLM layer
    test_df = X_test.copy()
    test_df["isFraud"] = y_test
    test_df["predicted_fraud"] = y_pred
    flagged = test_df[test_df["predicted_fraud"] == 1]
    flagged.to_csv(OUT_DIR / "flagged_transactions_supervised.csv", index=False)
    print(f"\nSaved {len(flagged):,} flagged transactions to "
          f"{OUT_DIR}/flagged_transactions_supervised.csv")


if __name__ == "__main__":
    main()