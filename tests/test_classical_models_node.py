from unittest.mock import patch, MagicMock

import numpy as np

from nodes.classical_models_node import classical_models_node, _gather_features
from shared.feature_engineering import FEATURE_ORDER


def _make_state(horizon="swing"):
    return {
        "ticker": "AAPL",
        "horizon": horizon,
        "data": {
            "company_info": {"sector": "Technology"},
        },
        "analyst_signals": {
            "technical": {
                "rating": "buy",
                "confidence": 0.7,
                "features": {
                    "rsi": 45.0,
                    "macd_histogram": 0.3,
                    "sma_20_50_cross": 1,
                    "bollinger_pct": 0.55,
                    "volume_ratio": 1.2,
                    "price_change_5d": 0.02,
                    "price_change_20d": 0.05,
                },
            },
            "fundamental": {
                "rating": "buy",
                "confidence": 0.6,
                "features": {
                    "pe_ratio": 22.0,
                    "debt_to_equity": 45.0,
                    "profit_margin": 0.18,
                    "revenue_growth": 0.08,
                    "eps": 4.5,
                },
            },
            "macro": {
                "rating": "hold",
                "confidence": 0.7,
                "features": {
                    "sector_performance": 0.6,
                    "market_trend": 1,
                },
            },
            "sentiment": {
                "rating": "buy",
                "confidence": 0.8,
                "features": {
                    "sentiment_score": 7.0,
                    "news_volume": 15,
                },
            },
        },
        "debate": {},
        "final_report": {},
    }


class TestGatherFeatures:
    def test_all_features_populated(self):
        state = _make_state()
        features = _gather_features(state)
        for f in FEATURE_ORDER:
            assert f in features or features.get(f) is not None or f.startswith("sector_"), \
                f"Missing feature: {f}"

    def test_sector_one_hot(self):
        state = _make_state()
        features = _gather_features(state)
        assert features["sector_technology"] == 1
        assert features["sector_energy"] == 0
        assert features["sector_healthcare"] == 0

    def test_missing_signals_return_none(self):
        state = {
            "ticker": "AAPL",
            "horizon": "swing",
            "data": {},
            "analyst_signals": {},
            "debate": {},
            "final_report": {},
        }
        features = _gather_features(state)
        assert features.get("rsi") is None
        assert features.get("pe_ratio") is None
        assert features.get("sentiment_score") is None


class TestClassicalModelsNode:
    def test_output_contract(self):
        """Node returns analyst_signals.classical_models with expected keys."""
        state = _make_state()

        mock_rf = MagicMock()
        mock_rf.predict.return_value = np.array(["BUY"])
        mock_rf.predict_proba.return_value = np.array([[0.1, 0.3, 0.2, 0.3, 0.1]])
        mock_rf.classes_ = np.array(["STRONG SELL", "SELL", "HOLD", "BUY", "STRONG BUY"])

        mock_knn = MagicMock()
        mock_knn.predict.return_value = np.array(["HOLD"])
        mock_knn.predict_proba.return_value = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])
        mock_knn.classes_ = np.array(["STRONG SELL", "SELL", "HOLD", "BUY", "STRONG BUY"])

        mock_imputer = MagicMock()
        mock_imputer.transform.return_value = np.zeros((1, len(FEATURE_ORDER)))

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.zeros((1, len(FEATURE_ORDER)))

        def fake_load(filename):
            if "rf_" in filename:
                return mock_rf
            if "knn_" in filename:
                return mock_knn
            if "imputer_" in filename:
                return mock_imputer
            if "scaler_" in filename:
                return mock_scaler
            return None

        with patch("nodes.classical_models_node._load", side_effect=fake_load):
            result = classical_models_node(state)

        cm = result["analyst_signals"]["classical_models"]
        assert cm["rf"]["prediction"] == "BUY"
        assert cm["knn"]["prediction"] == "HOLD"
        assert cm["feature_coverage"] > 0
        assert cm["total_features"] == len(FEATURE_ORDER)

    def test_missing_models_returns_none(self):
        """When no model files exist, rf and knn should be None."""
        state = _make_state()

        with patch("nodes.classical_models_node._load", return_value=None):
            result = classical_models_node(state)

        cm = result["analyst_signals"]["classical_models"]
        assert cm["rf"] is None
        assert cm["knn"] is None

    def test_respects_horizon(self):
        """Node should load models matching the state's horizon."""
        state = _make_state(horizon="short")
        loaded_files = []

        def track_load(filename):
            loaded_files.append(filename)
            return None

        with patch("nodes.classical_models_node._load", side_effect=track_load):
            classical_models_node(state)

        assert "imputer_short.pkl" in loaded_files
        assert "rf_short.pkl" in loaded_files
        assert "knn_short.pkl" in loaded_files
