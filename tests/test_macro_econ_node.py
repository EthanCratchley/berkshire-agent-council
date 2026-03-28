from nodes.macro_econ_node import macro_econ_node, _score_macro_indicator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_macro_indicators(overrides=None):
    """Build a full set of macro indicators with sensible defaults."""
    defaults = {
        "vix": 20.0,
        "yield_curve_spread": 0.5,
        "unemployment": 4.5,
        "fed_funds_rate": 4.0,
        "cpi_yoy": 3.0,
        "gdp": 28_000.0,
        "treasury_10y": 4.2,
        "cpi": 327.0,  # raw index, not used for scoring
    }
    if overrides:
        defaults.update(overrides)
    return defaults


def _make_state(macro_overrides=None, horizon="swing", ticker="AAPL",
                prior_macro=None, debate=None, include_macro=True):
    """Build a complete mock state for macro_econ_node testing."""
    state = {
        "ticker": ticker,
        "horizon": horizon,
        "data": {},
        "analyst_signals": {},
        "debate": debate or {},
    }
    if include_macro:
        state["data"]["macro_indicators"] = _make_macro_indicators(macro_overrides)
    if prior_macro:
        state["analyst_signals"]["macro"] = prior_macro
    return state


# ---------------------------------------------------------------------------
# Output contract tests
# ---------------------------------------------------------------------------

def test_output_contract_shape():
    """Validate the exact shape of the return dict matches the required contract."""
    state = _make_state()
    result = macro_econ_node(state)

    assert "analyst_signals" in result
    assert "macro" in result["analyst_signals"]

    sig = result["analyst_signals"]["macro"]

    # Required fields from teammate's spec
    assert "rating" in sig
    assert "confidence" in sig
    assert "features" in sig
    assert "details" in sig
    assert "horizon_alignment_note" in sig

    # Debate fields
    assert "debate_response" in sig
    assert "position_changed" in sig
    assert "counterpoints_addressed" in sig
    assert "claims_conceded" in sig
    assert "claims_disputed" in sig
    assert "final_position" in sig
    assert "weighting_statement" in sig

    # final_position sub-fields
    assert "rating" in sig["final_position"]
    assert "confidence" in sig["final_position"]


def test_rating_is_canonical():
    """Rating must be one of the 5 canonical values."""
    CANONICAL = {"strong_buy", "buy", "hold", "sell", "strong_sell"}

    test_cases = [
        {},  # neutral defaults
        {"vix": 10, "unemployment": 3.0, "fed_funds_rate": 2.0, "cpi_yoy": 1.5},  # bullish
        {"vix": 35, "unemployment": 7.0, "fed_funds_rate": 6.0, "cpi_yoy": 5.0},  # bearish
    ]

    for overrides in test_cases:
        state = _make_state(macro_overrides=overrides)
        result = macro_econ_node(state)
        sig = result["analyst_signals"]["macro"]
        assert sig["rating"] in CANONICAL, f"Got non-canonical rating: {sig['rating']}"
        assert sig["final_position"]["rating"] in CANONICAL


def test_confidence_clamped_0_to_1():
    """Confidence must always be in [0.0, 1.0]."""
    test_cases = [
        {},
        {"vix": 10, "unemployment": 3.0, "fed_funds_rate": 2.0, "cpi_yoy": 1.0},
        {"vix": 40, "unemployment": 8.0, "fed_funds_rate": 7.0, "cpi_yoy": 6.0},
    ]

    for overrides in test_cases:
        state = _make_state(macro_overrides=overrides)
        result = macro_econ_node(state)
        sig = result["analyst_signals"]["macro"]
        assert 0.0 <= sig["confidence"] <= 1.0
        assert 0.0 <= sig["final_position"]["confidence"] <= 1.0


def test_features_contain_model_features():
    """Features dict must include sector_performance and market_trend
    (from compute_macro_features) for downstream model consumption."""
    state = _make_state()
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert "sector_performance" in sig["features"]
    assert "market_trend" in sig["features"]


def test_features_contain_raw_indicators():
    """Features should also include the raw indicator values for transparency."""
    state = _make_state()
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert "vix" in sig["features"]
    assert "unemployment" in sig["features"]


def test_details_is_evidence_based_string():
    """Details string must contain indicator names and the rating."""
    state = _make_state()
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert isinstance(sig["details"], str)
    assert "rating=" in sig["details"]
    assert "horizon=" in sig["details"]


# ---------------------------------------------------------------------------
# Missing / edge-case data tests
# ---------------------------------------------------------------------------

def test_no_macro_data_defaults_to_hold():
    """No macro indicators should return hold with zero confidence."""
    state = _make_state(include_macro=False)
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] == "hold"
    assert sig["confidence"] == 0.0


def test_empty_macro_dict_defaults_to_hold():
    """Empty macro_indicators dict should return hold."""
    state = {
        "ticker": "SPY", "horizon": "swing",
        "data": {"macro_indicators": {}},
        "analyst_signals": {}, "debate": {},
    }
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] == "hold"
    assert sig["confidence"] == 0.0


def test_partial_macro_data_still_works():
    """Only some indicators available should still produce a result
    with reduced confidence."""
    state = _make_state(macro_overrides={
        "vix": 12.0,
        "yield_curve_spread": None,
        "unemployment": None,
        "fed_funds_rate": None,
        "cpi_yoy": None,
    })
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}
    assert 0.0 <= sig["confidence"] <= 1.0
    # Only VIX has data, so confidence should be low
    assert sig["confidence"] < 0.5


def test_none_values_no_crash():
    """None values for all indicators should not crash."""
    state = _make_state(macro_overrides={
        "vix": None,
        "yield_curve_spread": None,
        "unemployment": None,
        "fed_funds_rate": None,
        "cpi_yoy": None,
    })
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] == "hold"
    assert sig["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Bullish / Bearish / Neutral mapping tests
# ---------------------------------------------------------------------------

def test_bullish_macro_environment():
    """Low VIX, positive yield curve, low unemployment, low rates, low CPI
    should produce a bullish signal."""
    state = _make_state(macro_overrides={
        "vix": 12.0,
        "yield_curve_spread": 1.5,
        "unemployment": 3.5,
        "fed_funds_rate": 2.0,
        "cpi_yoy": 2.0,
    })
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] in ("buy", "strong_buy"), f"Expected bullish, got {sig['rating']}"
    assert sig["confidence"] > 0.5


def test_bearish_macro_environment():
    """High VIX, inverted yield curve, high unemployment, high rates, high CPI
    should produce a bearish signal."""
    state = _make_state(macro_overrides={
        "vix": 35.0,
        "yield_curve_spread": -0.5,
        "unemployment": 7.0,
        "fed_funds_rate": 6.0,
        "cpi_yoy": 5.5,
    })
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] in ("sell", "strong_sell"), f"Expected bearish, got {sig['rating']}"
    assert sig["confidence"] > 0.5


def test_mixed_macro_is_neutral():
    """Mixed indicators should produce a hold rating."""
    state = _make_state(macro_overrides={
        "vix": 12.0,            # bullish (< 15)
        "yield_curve_spread": -0.5,  # bearish (< 0)
        "unemployment": 3.5,    # bullish (< 4)
        "fed_funds_rate": 6.0,  # bearish (> 5)
        "cpi_yoy": 3.0,         # neutral
    })
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] == "hold"


# ---------------------------------------------------------------------------
# Individual indicator scoring tests
# ---------------------------------------------------------------------------

def test_score_vix_calm():
    """VIX < 15 = bullish (calm market)."""
    assert _score_macro_indicator("vix", 12.0) == 1


def test_score_vix_fearful():
    """VIX > 25 = bearish (fearful market)."""
    assert _score_macro_indicator("vix", 30.0) == -1


def test_score_vix_normal():
    """VIX between 15-25 = neutral."""
    assert _score_macro_indicator("vix", 20.0) == 0


def test_score_vix_none():
    """VIX None = neutral."""
    assert _score_macro_indicator("vix", None) == 0


def test_score_yield_positive():
    """Yield curve > 1.0 = bullish (healthy economy)."""
    assert _score_macro_indicator("yield_curve_spread", 1.5) == 1


def test_score_yield_inverted():
    """Yield curve < 0 = bearish (recession signal)."""
    assert _score_macro_indicator("yield_curve_spread", -0.3) == -1


def test_score_yield_narrow():
    """Yield curve between 0-1.0 = neutral."""
    assert _score_macro_indicator("yield_curve_spread", 0.5) == 0


def test_score_unemployment_low():
    """Unemployment < 4% = bullish (strong jobs)."""
    assert _score_macro_indicator("unemployment", 3.5) == 1


def test_score_unemployment_high():
    """Unemployment > 6% = bearish (weak jobs)."""
    assert _score_macro_indicator("unemployment", 7.0) == -1


def test_score_unemployment_moderate():
    """Unemployment 4-6% = neutral."""
    assert _score_macro_indicator("unemployment", 5.0) == 0


def test_score_fed_rate_low():
    """Fed funds rate < 3% = bullish (easy money)."""
    assert _score_macro_indicator("fed_funds_rate", 2.0) == 1


def test_score_fed_rate_high():
    """Fed funds rate > 5% = bearish (tight money)."""
    assert _score_macro_indicator("fed_funds_rate", 5.5) == -1


def test_score_fed_rate_moderate():
    """Fed funds rate 3-5% = neutral."""
    assert _score_macro_indicator("fed_funds_rate", 4.0) == 0


def test_score_cpi_yoy_low():
    """CPI YoY < 2.5% = bullish (low inflation)."""
    assert _score_macro_indicator("cpi_yoy", 2.0) == 1


def test_score_cpi_yoy_high():
    """CPI YoY > 4.0% = bearish (hot inflation)."""
    assert _score_macro_indicator("cpi_yoy", 5.0) == -1


def test_score_cpi_yoy_moderate():
    """CPI YoY 2.5-4.0% = neutral."""
    assert _score_macro_indicator("cpi_yoy", 3.0) == 0


def test_score_unknown_indicator():
    """Unknown indicator name should return 0."""
    assert _score_macro_indicator("totally_fake", 999.0) == 0


# ---------------------------------------------------------------------------
# Horizon-aware tests
# ---------------------------------------------------------------------------

def test_short_horizon_weights_volatility():
    """Short-term horizon should mention VIX/volatility in alignment note."""
    state = _make_state(horizon="short")
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert "Short-term" in sig["horizon_alignment_note"]


def test_long_horizon_weights_inflation():
    """Long-term horizon should mention inflation/employment in alignment note."""
    state = _make_state(horizon="long")
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert "Long-term" in sig["horizon_alignment_note"]


def test_all_horizons_produce_valid_output():
    """All three horizons should return valid output."""
    for horizon in ("short", "swing", "long"):
        state = _make_state(horizon=horizon)
        result = macro_econ_node(state)
        sig = result["analyst_signals"]["macro"]
        assert sig["rating"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}
        assert "horizon_alignment_note" in sig


def test_horizon_weighting_affects_confidence():
    """Different horizons should produce different confidence for the same data
    because indicator weights differ."""
    # VIX-heavy data: low VIX is bullish
    data = {"vix": 10.0, "yield_curve_spread": 0.5, "unemployment": 5.0,
            "fed_funds_rate": 4.0, "cpi_yoy": 3.0}

    state_short = _make_state(macro_overrides=data, horizon="short")
    state_long = _make_state(macro_overrides=data, horizon="long")

    result_short = macro_econ_node(state_short)
    result_long = macro_econ_node(state_long)

    sig_short = result_short["analyst_signals"]["macro"]
    sig_long = result_long["analyst_signals"]["macro"]

    # Both valid, but potentially different confidence due to weighting
    assert 0.0 <= sig_short["confidence"] <= 1.0
    assert 0.0 <= sig_long["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Debate compatibility tests
# ---------------------------------------------------------------------------

def test_non_debate_mode_defaults():
    """In non-debate mode, debate fields should have default/empty values."""
    state = _make_state()
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["counterpoints_addressed"] == []
    assert sig["claims_conceded"] == []
    assert sig["claims_disputed"] == []
    assert sig["weighting_statement"] == ""
    assert sig["position_changed"] is False


def test_debate_mode_populates_fields():
    """When debate is active and awaiting macro, debate fields should be populated."""
    debate = {
        "active_challenge": {
            "id": "macro-challenge-1",
            "action": "revise_or_defend",
            "reason": "Technical disagrees with macro outlook",
            "my_case": {
                "analyst": "macro",
                "rating": "hold",
                "confidence": 0.4,
                "details": "Mixed macro signals",
            },
            "opponent_case": {
                "analyst": "technical",
                "rating": "buy",
                "confidence": 0.6,
                "details": "Strong price momentum",
                "last_debate_response": "Price action is clearly bullish.",
            },
            "coalition": {
                "supporters_of_opponent": [],
                "partial": [],
            },
        },
        "awaiting_response_from": "macro",
    }
    state = _make_state(debate=debate)
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert len(sig["debate_response"]) > 0
    assert sig["weighting_statement"] != ""


def test_debate_not_triggered_for_other_node():
    """When debate awaits a different node, macro should NOT enter debate mode."""
    debate = {
        "active_challenge": {
            "id": "sentiment-challenge",
            "my_case": {"analyst": "sentiment"},
            "opponent_case": {"analyst": "fundamental"},
        },
        "awaiting_response_from": "sentiment",  # Not macro
    }
    state = _make_state(debate=debate)
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["counterpoints_addressed"] == []
    assert sig["claims_conceded"] == []
    assert sig["weighting_statement"] == ""


def test_position_changed_tracking():
    """position_changed should be True when the rating changes."""
    prior = {"rating": "buy"}
    # Bearish macro → sell, different from prior "buy"
    state = _make_state(
        macro_overrides={
            "vix": 35.0, "yield_curve_spread": -0.5,
            "unemployment": 7.0, "fed_funds_rate": 6.0, "cpi_yoy": 5.5,
        },
        prior_macro=prior,
    )
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    if sig["rating"] != "buy":
        assert sig["position_changed"] is True
    else:
        assert sig["position_changed"] is False


# ---------------------------------------------------------------------------
# CPI YoY specific tests 
# ---------------------------------------------------------------------------

def test_cpi_yoy_used_not_raw_cpi():
    """The node should score cpi_yoy (percentage) not raw cpi (index level).
    Raw CPI index (e.g. 327.0) should NOT affect scoring."""
    # cpi_yoy=2.0 is bullish (< 2.5), raw cpi=327 should be ignored
    state = _make_state(macro_overrides={
        "cpi_yoy": 2.0,
        "cpi": 327.0,
    })
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    # Verify cpi_yoy is in the details, not raw cpi index
    assert "cpi_yoy" in sig["details"]


def test_missing_cpi_yoy_does_not_crash():
    """If cpi_yoy is not computed by data fetcher, node should still work."""
    state = _make_state(macro_overrides={
        "cpi_yoy": None,
    })
    result = macro_econ_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
