import pandas as pd
import pytest

from shared.feature_engineering import (
    FEATURE_ORDER,
    SECTORS,
    assemble_feature_vector,
    compute_fundamental_features,
    compute_macro_features,
    compute_sector_features,
    compute_sentiment_features,
    compute_technical_features,
)


class TestFeatureOrder:
    def test_feature_order_length(self):
        assert len(FEATURE_ORDER) == 16 + len(SECTORS)  # 16 base + 9 sector one-hot

    def test_feature_order_has_no_duplicates(self):
        assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER))

    def test_feature_order_all_strings(self):
        assert all(isinstance(f, str) for f in FEATURE_ORDER)


class TestAssembleFeatureVector:
    def test_returns_correct_length(self):
        features = {name: float(i) for i, name in enumerate(FEATURE_ORDER)}
        vector = assemble_feature_vector(features)
        assert len(vector) == len(FEATURE_ORDER)

    def test_preserves_order(self):
        features = {name: float(i) for i, name in enumerate(FEATURE_ORDER)}
        vector = assemble_feature_vector(features)
        for i, name in enumerate(FEATURE_ORDER):
            assert vector[i] == float(i)

    def test_none_replaced_with_zero(self):
        features = {"rsi": None, "macd_histogram": 1.5}
        vector = assemble_feature_vector(features)
        assert vector[0] == 0.0  # rsi -> None -> 0.0
        assert vector[1] == 1.5  # macd_histogram

    def test_missing_keys_default_to_zero(self):
        vector = assemble_feature_vector({})
        assert vector == [0.0] * len(FEATURE_ORDER)

    def test_all_floats(self):
        features = {name: i for i, name in enumerate(FEATURE_ORDER)}  # ints
        vector = assemble_feature_vector(features)
        assert all(isinstance(v, float) for v in vector)


class TestComputeTechnicalFeatures:
    def _make_price_df(self, n_days: int) -> pd.DataFrame:
        """Create a synthetic price DataFrame with n_days rows."""
        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        close = [100 + i * 0.1 + (i % 7) for i in range(n_days)]
        volume = [1000000 + i * 1000 for i in range(n_days)]
        return pd.DataFrame(
            {"Close": close, "Volume": volume},
            index=dates,
        )

    def test_sufficient_data_returns_all_features(self):
        df = self._make_price_df(100)
        features = compute_technical_features(df)
        assert features["rsi"] is not None
        assert features["macd_histogram"] is not None
        assert features["sma_20_50_cross"] is not None
        assert features["bollinger_pct"] is not None
        assert features["volume_ratio"] is not None
        assert features["price_change_5d"] is not None
        assert features["price_change_20d"] is not None

    def test_insufficient_data_returns_nones(self):
        df = self._make_price_df(5)
        features = compute_technical_features(df)
        assert features["rsi"] is None
        assert features["macd_histogram"] is None
        assert features["sma_20_50_cross"] is None

    def test_empty_df_returns_all_nones(self):
        features = compute_technical_features(pd.DataFrame())
        assert all(v is None for v in features.values())

    def test_none_input_returns_all_nones(self):
        features = compute_technical_features(None)
        assert all(v is None for v in features.values())

    def test_rsi_in_valid_range(self):
        df = self._make_price_df(100)
        features = compute_technical_features(df)
        assert 0 <= features["rsi"] <= 100

    def test_sma_cross_is_binary(self):
        df = self._make_price_df(100)
        features = compute_technical_features(df)
        assert features["sma_20_50_cross"] in (0, 1)

    def test_returns_all_expected_keys(self):
        features = compute_technical_features(pd.DataFrame())
        expected = {"rsi", "macd_histogram", "sma_20_50_cross", "bollinger_pct",
                    "volume_ratio", "price_change_5d", "price_change_20d"}
        assert set(features.keys()) == expected


class TestComputeFundamentalFeatures:
    def test_complete_data(self):
        info = {
            "trailingPE": 25.0,
            "debtToEquity": 50.0,
            "profitMargins": 0.15,
            "revenueGrowth": 0.10,
            "trailingEps": 6.0,
        }
        features = compute_fundamental_features(info, {}, {}, {}, [])
        assert features["pe_ratio"] == 25.0
        assert features["debt_to_equity"] == 50.0
        assert features["profit_margin"] == 0.15
        assert features["revenue_growth"] == 0.10
        assert features["eps"] == 6.0

    def test_empty_data_returns_nones(self):
        features = compute_fundamental_features({}, {}, {}, {}, [])
        assert all(v is None for v in features.values())

    def test_fallback_to_basic_financials(self):
        info = {}
        basic = {"metric": {"peRatio": 30.0}}
        features = compute_fundamental_features(info, basic, {}, {}, [])
        assert features["pe_ratio"] == 30.0

    def test_returns_all_expected_keys(self):
        features = compute_fundamental_features({}, {}, {}, {}, [])
        expected = {"pe_ratio", "debt_to_equity", "profit_margin", "revenue_growth", "eps"}
        assert set(features.keys()) == expected


class TestComputeMacroFeatures:
    def test_bullish_conditions(self):
        macro = {
            "vix": 12.0,
            "yield_curve_spread": 1.5,
            "unemployment": 3.5,
        }
        features = compute_macro_features(macro)
        assert features["market_trend"] == 1
        assert features["sector_performance"] is not None
        assert features["sector_performance"] > 0

    def test_bearish_conditions(self):
        macro = {
            "vix": 35.0,
            "yield_curve_spread": -1.0,
            "unemployment": 7.0,
        }
        features = compute_macro_features(macro)
        assert features["market_trend"] == 0
        assert features["sector_performance"] < 0

    def test_empty_indicators(self):
        features = compute_macro_features({})
        assert features["sector_performance"] is None
        assert features["market_trend"] is None

    def test_partial_indicators(self):
        macro = {"vix": 20.0}
        features = compute_macro_features(macro)
        assert features["sector_performance"] is not None
        assert features["market_trend"] is not None


class TestComputeSentimentFeatures:
    def test_passthrough(self):
        features = compute_sentiment_features(7.5, 15)
        assert features["sentiment_score"] == 7.5
        assert features["news_volume"] == 15

    def test_neutral_defaults(self):
        features = compute_sentiment_features(5.0, 0)
        assert features["sentiment_score"] == 5.0
        assert features["news_volume"] == 0


class TestComputeSectorFeatures:
    def test_known_sector_sets_one_hot(self):
        features = compute_sector_features("Technology")
        assert features["sector_technology"] == 1
        assert sum(features.values()) == 1

    def test_different_sector(self):
        features = compute_sector_features("Energy")
        assert features["sector_energy"] == 1
        assert features["sector_technology"] == 0

    def test_unknown_sector_all_zeros(self):
        features = compute_sector_features("Alien Sector")
        assert all(v == 0 for v in features.values())

    def test_empty_sector_all_zeros(self):
        features = compute_sector_features("")
        assert all(v == 0 for v in features.values())

    def test_returns_correct_number_of_keys(self):
        features = compute_sector_features("Technology")
        assert len(features) == len(SECTORS)
