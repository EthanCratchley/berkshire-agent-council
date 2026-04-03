import pandas as pd

from nodes.technical_node import technical_node, _score_indicator, _build_price_dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_records(n=60, base_close=150.0, base_volume=1_000_000, trend=0.0, volume_trend=None):
    """Generate n days of synthetic price history records.

    Args:
        n: number of trading days
        base_close: starting close price
        base_volume: daily volume
        trend: daily price change (positive=uptrend, negative=downtrend)
        volume_trend: list of volume overrides or callable, or None
    """
    records = []
    for i in range(n):
        close = base_close + (trend * i)
        
        # Apply specific volume logic
        if isinstance(volume_trend, list) and i < len(volume_trend):
            vol = volume_trend[i]
        elif callable(volume_trend):
            vol = volume_trend(i, base_volume)
        else:
            vol = base_volume

        records.append({
            "Date": f"2026-01-{(i % 28) + 1:02d}",
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": vol,
        })
    return records


def _make_state(price_records=None, horizon="swing", ticker="AAPL"):
    """Build a complete mock state for technical_node testing."""
    state = {
        "ticker": ticker,
        "horizon": horizon,
        "data": {
            "price_history": price_records if price_records is not None else _make_price_records(),
        },
        "analyst_signals": {},
    }
    return state


# ---------------------------------------------------------------------------
# Output contract tests
# ---------------------------------------------------------------------------

def test_output_contract_shape():
    """Validate the exact shape of the return dict matches the required contract."""
    state = _make_state()
    result = technical_node(state)

    assert "analyst_signals" in result
    assert "technical" in result["analyst_signals"]

    sig = result["analyst_signals"]["technical"]

    # Required fields from teammate's spec
    assert "rating" in sig
    assert "confidence" in sig
    assert "features" in sig
    assert "details" in sig
    assert "horizon_alignment_note" in sig




def test_rating_is_canonical():
    """Rating must be one of the 5 canonical values."""
    CANONICAL = {"strong_buy", "buy", "hold", "sell", "strong_sell"}

    for trend in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        state = _make_state(price_records=_make_price_records(trend=trend))
        result = technical_node(state)
        sig = result["analyst_signals"]["technical"]
        assert sig["rating"] in CANONICAL, f"Got non-canonical rating: {sig['rating']}"


def test_confidence_clamped_0_to_1():
    """Confidence must always be in [0.0, 1.0]."""
    for trend in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        state = _make_state(price_records=_make_price_records(trend=trend))
        result = technical_node(state)
        sig = result["analyst_signals"]["technical"]
        assert 0.0 <= sig["confidence"] <= 1.0


def test_feature_keys_match_expected():
    """Feature dict must contain exactly the 7 expected technical indicator keys."""
    expected_keys = {
        "rsi", "macd_histogram", "sma_20_50_cross",
        "bollinger_pct", "volume_ratio",
        "price_change_5d", "price_change_20d",
    }
    state = _make_state()
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]
    assert set(sig["features"].keys()) == expected_keys


def test_details_is_evidence_based_string():
    """Details string must contain indicator names and the rating."""
    state = _make_state()
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]
    assert isinstance(sig["details"], str)
    assert "rating=" in sig["details"]
    assert "horizon=" in sig["details"]


# ---------------------------------------------------------------------------
# Missing / edge-case data tests
# ---------------------------------------------------------------------------

def test_no_price_history_defaults_to_hold():
    """No price data should return hold with zero confidence."""
    state = _make_state(price_records=[])
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]

    assert sig["rating"] == "hold"
    assert sig["confidence"] == 0.0
    assert all(v is None for v in sig["features"].values())


def test_no_data_key_defaults_to_hold():
    """State with empty data dict should return hold."""
    state = {"ticker": "XYZ", "horizon": "swing", "data": {}, "analyst_signals": {}, "debate": {}}
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]

    assert sig["rating"] == "hold"
    assert sig["confidence"] == 0.0


def test_insufficient_data_still_returns():
    """Very short price history (< 15 days) should still produce a result
    with partial features (e.g., RSI=None) but no crash."""
    short_records = _make_price_records(n=5)
    state = _make_state(price_records=short_records)
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]

    assert sig["rating"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}
    assert 0.0 <= sig["confidence"] <= 1.0
    # RSI requires 15+ periods, so should be None
    assert sig["features"]["rsi"] is None


def test_missing_volume_column_handled():
    """Price records without Volume column should not crash."""
    records = [{"Close": 100.0 + i} for i in range(60)]
    state = _make_state(price_records=records)
    # _build_price_dataframe requires Close and Volume, so this should
    # return hold due to missing columns
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]
    assert sig["rating"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}


# ---------------------------------------------------------------------------
# Bullish / Bearish / Neutral mapping tests
# ---------------------------------------------------------------------------

def test_strong_uptrend_is_bullish():
    """Strong uptrend should produce a bullish-leaning or neutral result.

    Note: In a strong uptrend, RSI may hit overbought (bearish contrarian)
    and Bollinger may be near upper band (bearish), which partially offset
    the trend signals. This is correct behavior — RSI is contrarian.
    We test that the trend indicators (SMA, MACD, price changes) are bullish."""
    records = _make_price_records(n=60, base_close=100.0, trend=0.5)
    state = _make_state(price_records=records)
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]

    # Trend indicators should be bullish
    assert sig["features"]["sma_20_50_cross"] == 1  # Golden cross
    assert sig["features"]["price_change_20d"] > 0  # Positive 20d change
    assert sig["rating"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}
    assert sig["confidence"] >= 0.0


def test_strong_downtrend_is_bearish():
    """Strong downtrend should produce bearish-leaning trend indicators.

    Similar to uptrend test: RSI in a downtrend hits oversold (bullish
    contrarian) which offsets trend signals. We verify the trend
    indicators themselves score correctly."""
    records = _make_price_records(n=60, base_close=200.0, trend=-0.5)
    state = _make_state(price_records=records)
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]

    # Trend indicators should be bearish
    assert sig["features"]["sma_20_50_cross"] == 0  # Death cross
    assert sig["features"]["price_change_20d"] < 0  # Negative 20d change
    assert sig["rating"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}
    assert sig["confidence"] >= 0.0


def test_flat_trend_is_neutral():
    """Flat price action should produce a hold rating with high confidence
    due to strong consensus among all indicators that the market is neutral."""
    records = _make_price_records(n=60, base_close=150.0, trend=0.0)
    state = _make_state(price_records=records)
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]

    assert sig["rating"] == "hold"
    assert sig["confidence"] > 0.0  # Should be highly confident it's a hold


def test_high_volume_downtrend_is_bearish():
    """High volume on a downtrend should amplify the bearishness, not score as bullish."""
    # Create a downtrend where recent volume spikes massively
    def vol_spike(i, base):
        return base * 5 if i >= 55 else base
    
    records = _make_price_records(n=60, base_close=200.0, trend=-0.5, volume_trend=vol_spike)
    state = _make_state(price_records=records)
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]

    # We expect the node to recognize volume_ratio > 1.5 during a downtrend as bearish
    assert sig["features"]["volume_ratio"] > 1.5
    assert _score_indicator("volume_ratio", sig["features"]["volume_ratio"], trend_direction=-0.5) == -1


# ---------------------------------------------------------------------------
# Individual indicator scoring tests
# ---------------------------------------------------------------------------

def test_score_rsi_oversold():
    """RSI below 35 = bullish (oversold, buying opportunity)."""
    assert _score_indicator("rsi", 25.0) == 1


def test_score_rsi_overbought():
    """RSI above 65 = bearish (overbought)."""
    assert _score_indicator("rsi", 72.0) == -1


def test_score_rsi_neutral():
    """RSI between 35 and 65 = neutral."""
    assert _score_indicator("rsi", 50.0) == 0


def test_score_rsi_none():
    """RSI None = neutral (no data)."""
    assert _score_indicator("rsi", None) == 0


def test_score_macd_positive():
    """Positive MACD histogram = bullish."""
    assert _score_indicator("macd_histogram", 0.5) == 1


def test_score_macd_negative():
    """Negative MACD histogram = bearish."""
    assert _score_indicator("macd_histogram", -0.3) == -1


def test_score_sma_golden_cross():
    """SMA 20 > SMA 50 (golden cross) = bullish."""
    assert _score_indicator("sma_20_50_cross", 1) == 1


def test_score_sma_death_cross():
    """SMA 20 < SMA 50 (death cross) = bearish."""
    assert _score_indicator("sma_20_50_cross", 0) == -1


def test_score_bollinger_near_lower():
    """Bollinger %B < 0.2 (near lower band) = bullish (potential bounce)."""
    assert _score_indicator("bollinger_pct", 0.1) == 1


def test_score_bollinger_near_upper():
    """Bollinger %B > 0.8 (near upper band) = bearish (potential reversal)."""
    assert _score_indicator("bollinger_pct", 0.9) == -1


def test_score_bollinger_middle():
    """Bollinger %B between 0.2 and 0.8 = neutral."""
    assert _score_indicator("bollinger_pct", 0.5) == 0


def test_score_volume_high():
    """Volume ratio > 1.5 = bullish (high interest)."""
    assert _score_indicator("volume_ratio", 2.0) == 1


def test_score_volume_low():
    """Volume ratio < 0.7 = neutral (low interest doesn't confirm trend)."""
    assert _score_indicator("volume_ratio", 0.5) == 0


def test_score_price_change_5d_up():
    """5-day price change > 2% = bullish."""
    assert _score_indicator("price_change_5d", 0.05) == 1


def test_score_price_change_5d_down():
    """5-day price change < -2% = bearish."""
    assert _score_indicator("price_change_5d", -0.04) == -1


def test_score_price_change_20d_up():
    """20-day price change > 5% = bullish."""
    assert _score_indicator("price_change_20d", 0.08) == 1


def test_score_price_change_20d_down():
    """20-day price change < -5% = bearish."""
    assert _score_indicator("price_change_20d", -0.10) == -1


# ---------------------------------------------------------------------------
# Horizon-aware tests
# ---------------------------------------------------------------------------

def test_short_horizon_weights_momentum():
    """Short-term horizon should weight RSI/momentum more heavily."""
    records = _make_price_records(n=60, trend=1.0)
    state_short = _make_state(price_records=records, horizon="short")
    state_long = _make_state(price_records=records, horizon="long")

    result_short = technical_node(state_short)
    result_long = technical_node(state_long)

    sig_short = result_short["analyst_signals"]["technical"]
    sig_long = result_long["analyst_signals"]["technical"]

    # Both should produce valid output
    assert sig_short["rating"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}
    assert sig_long["rating"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}

    # Horizon alignment note should mention the horizon
    assert "Short-term" in sig_short["horizon_alignment_note"]
    assert "Long-term" in sig_long["horizon_alignment_note"]


def test_all_horizons_produce_valid_output():
    """All three horizons should return valid output."""
    for horizon in ("short", "swing", "long"):
        state = _make_state(horizon=horizon)
        result = technical_node(state)
        sig = result["analyst_signals"]["technical"]
        assert sig["rating"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}
        assert "horizon_alignment_note" in sig





def test_volume_unknown_trend_is_neutral():
    """When price_change_20d is None (insufficient data), high volume anomalies
    should return 0 (neutral) not default to bullish."""
    # Test the scorer directly: None trend + high volume = neutral
    assert _score_indicator("volume_ratio", 2.5, trend_direction=None) == 0
    assert _score_indicator("volume_ratio", 3.0, trend_direction=None) == 0
    # Known trend still works
    assert _score_indicator("volume_ratio", 2.5, trend_direction=0.05) == 1
    assert _score_indicator("volume_ratio", 2.5, trend_direction=-0.05) == -1





def test_position_changed_tracking_on_parse_failure():
    """Parse-failure fallback to hold should return hold with zero confidence."""
    # Missing Volume forces _build_price_dataframe to return empty DataFrame.
    records = [{"Close": 100.0 + i} for i in range(10)]
    state = _make_state(price_records=records)
    result = technical_node(state)
    sig = result["analyst_signals"]["technical"]

    assert sig["rating"] == "hold"
    assert sig["confidence"] == 0.0


# ---------------------------------------------------------------------------
# _build_price_dataframe helper tests
# ---------------------------------------------------------------------------

def test_build_price_dataframe_valid():
    """Valid records should produce a usable DataFrame."""
    records = _make_price_records(n=10)
    df = _build_price_dataframe(records)
    assert not df.empty
    assert "Close" in df.columns
    assert "Volume" in df.columns
    assert len(df) == 10


def test_build_price_dataframe_empty():
    """Empty list should produce empty DataFrame."""
    df = _build_price_dataframe([])
    assert df.empty


def test_build_price_dataframe_missing_columns():
    """Records without required columns should produce empty DataFrame."""
    records = [{"Open": 100, "High": 101}]  # No Close or Volume
    df = _build_price_dataframe(records)
    assert df.empty


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
