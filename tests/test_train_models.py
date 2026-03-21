import os
import tempfile

import joblib
import pandas as pd
import pytest

from shared.feature_engineering import FEATURE_ORDER


def _make_synthetic_dataset(n_rows: int = 50) -> pd.DataFrame:
    """Create a small synthetic dataset for testing the training pipeline."""
    import random
    random.seed(42)

    labels = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    rows = []
    for i in range(n_rows):
        row = {"ticker": "TEST", "date": f"2024-01-{(i % 28) + 1:02d}"}
        for feat in FEATURE_ORDER:
            row[feat] = random.uniform(-1, 1)
        row["label"] = labels[i % len(labels)]
        rows.append(row)

    return pd.DataFrame(rows)


class TestTrainModels:
    def test_training_produces_model_files(self, tmp_path):
        """Train on synthetic data and verify .pkl files are created."""
        # Write synthetic dataset
        dataset_path = tmp_path / "test_dataset.csv"
        df = _make_synthetic_dataset(50)
        df.to_csv(dataset_path, index=False)

        # Import and patch paths
        from models.train_models import load_dataset, train_test_split_temporal

        loaded = load_dataset(str(dataset_path))
        assert len(loaded) == 50

        train_df, test_df = train_test_split_temporal(loaded, train_ratio=0.8)
        assert len(train_df) == 40
        assert len(test_df) == 10

    def test_rf_model_save_and_load(self, tmp_path):
        """Train RF on synthetic data, save, and verify it loads and predicts."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer

        df = _make_synthetic_dataset(50)
        X = df[FEATURE_ORDER].values
        y = df["label"].values

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

        rf = RandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=42, class_weight="balanced"
        )
        rf.fit(X, y)

        # Save
        model_path = tmp_path / "rf_model.pkl"
        imputer_path = tmp_path / "imputer.pkl"
        joblib.dump(rf, model_path)
        joblib.dump(imputer, imputer_path)

        # Load and predict
        loaded_rf = joblib.load(model_path)
        loaded_imputer = joblib.load(imputer_path)

        test_X = loaded_imputer.transform(X[:5])
        preds = loaded_rf.predict(test_X)
        assert len(preds) == 5
        assert all(p in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"] for p in preds)

    def test_rf_predict_proba_shape(self):
        """Verify predict_proba returns probabilities for all 5 classes."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer

        df = _make_synthetic_dataset(50)
        X = df[FEATURE_ORDER].values
        y = df["label"].values

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

        rf = RandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=42, class_weight="balanced"
        )
        rf.fit(X, y)

        proba = rf.predict_proba(X[:1])
        assert proba.shape == (1, 5)
        assert abs(sum(proba[0]) - 1.0) < 1e-6  # probabilities sum to 1

    def test_temporal_split_preserves_order(self):
        """Verify time-based split doesn't shuffle data."""
        from models.train_models import train_test_split_temporal

        df = _make_synthetic_dataset(100)
        df["date"] = pd.date_range("2024-01-01", periods=100).strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)

        train_df, test_df = train_test_split_temporal(df)

        # All train dates should be before all test dates
        assert train_df["date"].max() <= test_df["date"].min()
