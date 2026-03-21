"""
Train the Random Forest classifier on the cached dataset.

Loads data/cached_dataset.csv, performs a time-based 80/20 split,
trains an RF model with class balancing, and saves the model artifacts.
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.feature_engineering import FEATURE_ORDER

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cached_dataset.csv",
)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load and sort the cached dataset by date."""
    df = pd.read_csv(path)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def train_test_split_temporal(df: pd.DataFrame, train_ratio: float = 0.8):
    """Time-based split: train on older data, test on recent."""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def train():
    """Train the Random Forest model and save artifacts."""
    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset: {len(df)} samples, {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}\n")

    train_df, test_df = train_test_split_temporal(df)
    print(f"Train: {len(train_df)} samples | Test: {len(test_df)} samples")

    X_train = train_df[FEATURE_ORDER].values
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_ORDER].values
    y_test = test_df["label"].values

    # Impute missing values with training set median
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate
    rf_preds = rf.predict(X_test)
    accuracy = accuracy_score(y_test, rf_preds)

    print(f"\n{'='*50}")
    print(f"  Random Forest Results")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, rf_preds)}")

    # Feature importance
    print("Feature Importance:")
    importances = sorted(
        zip(FEATURE_ORDER, rf.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, imp in importances:
        bar = "#" * int(imp * 100)
        print(f"  {name:<25} {imp:.4f} {bar}")

    # Save artifacts
    rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    imputer_path = os.path.join(MODEL_DIR, "imputer.pkl")

    joblib.dump(rf, rf_path)
    joblib.dump(imputer, imputer_path)

    print(f"\nModel saved to {rf_path}")
    print(f"Imputer saved to {imputer_path}")

    return rf, imputer


if __name__ == "__main__":
    train()
