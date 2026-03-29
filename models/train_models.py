"""
Train per-horizon Random Forest classifiers on the cached dataset.

For each horizon (short/swing/long), loads data/cached_dataset.csv,
filters to rows with a valid label for that horizon, performs a time-based
80/20 split, trains an RF model with class balancing, and saves artifacts.
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.feature_engineering import FEATURE_ORDER
from shared.horizon import HORIZON_LABEL_CONFIG, VALID_HORIZONS

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


def train_horizon(df: pd.DataFrame, horizon: str):
    """Train an RF model for a single horizon and save artifacts."""
    label_col = f"label_{horizon}"
    cfg = HORIZON_LABEL_CONFIG[horizon]

    # Drop rows without a label for this horizon
    hdf = df.dropna(subset=[label_col]).reset_index(drop=True)
    print(f"\n{'='*60}")
    print(f"  {horizon.upper()} horizon  ({cfg['forward_days']}-day forward return)")
    print(f"  Thresholds: {cfg['thresholds']}")
    print(f"{'='*60}")
    print(f"Samples with labels: {len(hdf)}")
    print(f"Label distribution:\n{hdf[label_col].value_counts().to_string()}\n")

    train_df, test_df = train_test_split_temporal(hdf)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    X_train = train_df[FEATURE_ORDER].values
    y_train = train_df[label_col].values
    X_test = test_df[FEATURE_ORDER].values
    y_test = test_df[label_col].values

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    print(f"Training Random Forest ({horizon})...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, preds)}")

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

    rf_path = os.path.join(MODEL_DIR, f"rf_{horizon}.pkl")
    imputer_path = os.path.join(MODEL_DIR, f"imputer_{horizon}.pkl")
    joblib.dump(rf, rf_path)
    joblib.dump(imputer, imputer_path)
    print(f"\nSaved: {rf_path}, {imputer_path}")

    return rf, imputer


def train(horizons: list[str] | None = None):
    """Train RF models for the specified horizons (default: all)."""
    if horizons is None:
        horizons = sorted(VALID_HORIZONS)

    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset: {len(df)} samples, {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    models = {}
    for h in horizons:
        if h not in HORIZON_LABEL_CONFIG:
            print(f"Skipping unknown horizon: {h}")
            continue
        models[h] = train_horizon(df, h)

    return models


if __name__ == "__main__":
    # Allow training a single horizon via CLI: python train_models.py short
    import sys as _sys
    args = _sys.argv[1:]
    selected = [a for a in args if a in VALID_HORIZONS] or None
    train(selected)
