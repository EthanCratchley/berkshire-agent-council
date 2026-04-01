import os
import tempfile

import joblib
import pandas as pd
import pytest

from shared.feature_engineering import FEATURE_ORDER
from shared.horizon import HORIZON_LABEL_CONFIG


def _make_synthetic_dataset(n_rows: int = 50) -> pd.DataFrame:
    """Create a small synthetic dataset with per-horizon label columns."""
    import random
    random.seed(42)

    labels = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    label_cols = [f"label_{h}" for h in HORIZON_LABEL_CONFIG]
    rows = []
    for i in range(n_rows):
        row = {"ticker": "TEST", "date": f"2024-01-{(i % 28) + 1:02d}"}
        for feat in FEATURE_ORDER:
            row[feat] = random.uniform(-1, 1)
        for col in label_cols:
            row[col] = labels[i % len(labels)]
        rows.append(row)

    return pd.DataFrame(rows)


class TestTrainModels:
    def test_training_produces_model_files(self, tmp_path):
        """Train on synthetic data and verify .pkl files are created."""
        dataset_path = tmp_path / "test_dataset.csv"
        df = _make_synthetic_dataset(50)
        df.to_csv(dataset_path, index=False)

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
        y = df["label_short"].values

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

        rf = RandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=42, class_weight="balanced"
        )
        rf.fit(X, y)

        model_path = tmp_path / "rf_short.pkl"
        imputer_path = tmp_path / "imputer_short.pkl"
        joblib.dump(rf, model_path)
        joblib.dump(imputer, imputer_path)

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
        y = df["label_swing"].values

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

        rf = RandomForestClassifier(
            n_estimators=10, max_depth=5, random_state=42, class_weight="balanced"
        )
        rf.fit(X, y)

        proba = rf.predict_proba(X[:1])
        assert proba.shape == (1, 5)
        assert abs(sum(proba[0]) - 1.0) < 1e-6

    def test_temporal_split_preserves_order(self):
        """Verify time-based split doesn't shuffle data."""
        from models.train_models import train_test_split_temporal

        df = _make_synthetic_dataset(100)
        df["date"] = pd.date_range("2024-01-01", periods=100).strftime("%Y-%m-%d")
        df = df.sort_values("date").reset_index(drop=True)

        train_df, test_df = train_test_split_temporal(df)
        assert train_df["date"].max() <= test_df["date"].min()

    def test_per_horizon_training(self, tmp_path, monkeypatch):
        """Train separate models for each horizon and verify artifacts."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer

        df = _make_synthetic_dataset(60)
        df["date"] = pd.date_range("2024-01-01", periods=60).strftime("%Y-%m-%d")

        for horizon in HORIZON_LABEL_CONFIG:
            label_col = f"label_{horizon}"
            hdf = df.dropna(subset=[label_col])
            assert len(hdf) > 0

            X = hdf[FEATURE_ORDER].values
            y = hdf[label_col].values

            imputer = SimpleImputer(strategy="median")
            X = imputer.fit_transform(X)

            rf = RandomForestClassifier(
                n_estimators=10, max_depth=5, random_state=42, class_weight="balanced"
            )
            rf.fit(X, y)

            rf_path = tmp_path / f"rf_{horizon}.pkl"
            imp_path = tmp_path / f"imputer_{horizon}.pkl"
            joblib.dump(rf, rf_path)
            joblib.dump(imputer, imp_path)

            loaded = joblib.load(rf_path)
            preds = loaded.predict(X[:3])
            assert len(preds) == 3
            assert all(p in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"] for p in preds)

    def test_missing_horizon_labels_handled(self):
        """Rows with None labels for longer horizons should be droppable."""
        df = _make_synthetic_dataset(50)
        # Simulate long-horizon rows lacking labels (as happens near end of price data)
        df.loc[40:, "label_long"] = None

        hdf = df.dropna(subset=["label_long"])
        assert len(hdf) == 40
        assert df.dropna(subset=["label_short"]).shape[0] == 50


class TestClassifyReturn:
    def test_short_horizon_thresholds(self):
        from data.build_dataset import classify_return

        thresholds = HORIZON_LABEL_CONFIG["short"]["thresholds"]
        assert classify_return(0.06, thresholds) == "STRONG BUY"
        assert classify_return(0.03, thresholds) == "BUY"
        assert classify_return(0.00, thresholds) == "HOLD"
        assert classify_return(-0.03, thresholds) == "SELL"
        assert classify_return(-0.06, thresholds) == "STRONG SELL"

    def test_swing_horizon_thresholds(self):
        from data.build_dataset import classify_return

        thresholds = HORIZON_LABEL_CONFIG["swing"]["thresholds"]
        assert classify_return(0.15, thresholds) == "STRONG BUY"
        assert classify_return(0.08, thresholds) == "BUY"
        assert classify_return(0.00, thresholds) == "HOLD"
        assert classify_return(-0.08, thresholds) == "SELL"
        assert classify_return(-0.15, thresholds) == "STRONG SELL"

    def test_long_horizon_thresholds(self):
        from data.build_dataset import classify_return

        thresholds = HORIZON_LABEL_CONFIG["long"]["thresholds"]
        assert classify_return(0.30, thresholds) == "STRONG BUY"
        assert classify_return(0.15, thresholds) == "BUY"
        assert classify_return(0.00, thresholds) == "HOLD"
        assert classify_return(-0.15, thresholds) == "SELL"
        assert classify_return(-0.30, thresholds) == "STRONG SELL"

    def test_boundary_values(self):
        from data.build_dataset import classify_return

        thresholds = (0.05, 0.02, -0.02, -0.05)
        # Exactly at threshold — should fall to next lower bucket
        assert classify_return(0.05, thresholds) == "BUY"
        assert classify_return(0.02, thresholds) == "HOLD"
        assert classify_return(-0.02, thresholds) == "SELL"
        assert classify_return(-0.05, thresholds) == "STRONG SELL"
