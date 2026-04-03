import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from shared.feature_engineering import FEATURE_ORDER
from shared.horizon import HORIZON_LABEL_CONFIG


def _make_synthetic_dataset(n_rows: int = 100) -> pd.DataFrame:
    import random
    random.seed(42)

    labels = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    rows = []
    for i in range(n_rows):
        row = {"ticker": "TEST", "date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"}
        for feat in FEATURE_ORDER:
            row[feat] = random.uniform(-1, 1)
        for h in HORIZON_LABEL_CONFIG:
            row[f"label_{h}"] = labels[i % len(labels)]
        rows.append(row)

    return pd.DataFrame(rows)


def _train_and_save(tmp_path, df, horizon):
    """Train RF + KNN on synthetic data and save all artifacts."""
    label_col = f"label_{horizon}"
    X = df[FEATURE_ORDER].values
    y = df[label_col].values

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42, class_weight="balanced")
    rf.fit(X_imp, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X_scaled, y)

    joblib.dump(rf, tmp_path / f"rf_{horizon}.pkl")
    joblib.dump(imputer, tmp_path / f"imputer_{horizon}.pkl")
    joblib.dump(knn, tmp_path / f"knn_{horizon}.pkl")
    joblib.dump(scaler, tmp_path / f"scaler_{horizon}.pkl")


class TestEvaluate:
    def test_load_test_split_returns_correct_shape(self, tmp_path):
        df = _make_synthetic_dataset(100)
        csv_path = tmp_path / "dataset.csv"
        df.to_csv(csv_path, index=False)

        from models.evaluate import load_test_split
        import models.evaluate as ev
        original = ev.DATASET_PATH
        ev.DATASET_PATH = str(csv_path)
        try:
            X_test, y_test, test_df = load_test_split("short")
            assert X_test.shape[1] == len(FEATURE_ORDER)
            assert len(y_test) == len(test_df)
            assert len(y_test) == 20  # 20% of 100
        finally:
            ev.DATASET_PATH = original

    def test_evaluate_horizon_produces_results(self, tmp_path):
        df = _make_synthetic_dataset(100)
        csv_path = tmp_path / "dataset.csv"
        df.to_csv(csv_path, index=False)

        _train_and_save(tmp_path, df, "short")

        from models.evaluate import evaluate_horizon
        import models.evaluate as ev
        orig_ds, orig_md = ev.DATASET_PATH, ev.MODEL_DIR
        ev.DATASET_PATH = str(csv_path)
        ev.MODEL_DIR = str(tmp_path)
        try:
            results = evaluate_horizon("short")
            assert results is not None
            assert "rf" in results
            assert "knn" in results
            assert 0.0 <= results["rf"]["accuracy"] <= 1.0
            assert 0.0 <= results["knn"]["accuracy"] <= 1.0
            assert len(results["rf"]["predictions"]) == len(results["y_test"])
        finally:
            ev.DATASET_PATH = orig_ds
            ev.MODEL_DIR = orig_md

    def test_missing_model_skips_gracefully(self, tmp_path):
        df = _make_synthetic_dataset(100)
        csv_path = tmp_path / "dataset.csv"
        df.to_csv(csv_path, index=False)

        # Save only imputer — no RF or KNN
        X = df[FEATURE_ORDER].values
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X)
        joblib.dump(imputer, tmp_path / "imputer_short.pkl")

        from models.evaluate import evaluate_horizon
        import models.evaluate as ev
        orig_ds, orig_md = ev.DATASET_PATH, ev.MODEL_DIR
        ev.DATASET_PATH = str(csv_path)
        ev.MODEL_DIR = str(tmp_path)
        try:
            results = evaluate_horizon("short")
            assert results is not None
            assert "rf" not in results
            assert "knn" not in results
        finally:
            ev.DATASET_PATH = orig_ds
            ev.MODEL_DIR = orig_md

    def test_rf_and_knn_predictions_valid_labels(self, tmp_path):
        df = _make_synthetic_dataset(100)
        csv_path = tmp_path / "dataset.csv"
        df.to_csv(csv_path, index=False)

        _train_and_save(tmp_path, df, "swing")

        from models.evaluate import evaluate_horizon
        import models.evaluate as ev
        orig_ds, orig_md = ev.DATASET_PATH, ev.MODEL_DIR
        ev.DATASET_PATH = str(csv_path)
        ev.MODEL_DIR = str(tmp_path)
        try:
            results = evaluate_horizon("swing")
            valid = {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}
            for model in ("rf", "knn"):
                preds = results[model]["predictions"]
                assert all(p in valid for p in preds)
        finally:
            ev.DATASET_PATH = orig_ds
            ev.MODEL_DIR = orig_md
