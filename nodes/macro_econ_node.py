from shared.state_schema import BerkshireState
from shared.horizon import normalize_horizon, horizon_label
from shared.feature_engineering import compute_macro_features
from shared.stance import Rating, rating_from_aggregate_score, rating_to_score, rating_to_signal


# Scoring thresholds for individual macro indicators
MACRO_THRESHOLDS = {
    "vix": {"bullish_below": 15, "bearish_above": 25},
    "yield_curve_spread": {"bullish_above": 1.0, "bearish_below": 0.0},
    "unemployment": {"bullish_below": 4.0, "bearish_above": 6.0},
    "fed_funds_rate": {"bullish_below": 3.0, "bearish_above": 5.0},
    "cpi_yoy": {"bullish_below": 2.5, "bearish_above": 4.0},
}

# Horizon-based indicator weights
# Short-term: VIX/volatility focused; Long-term: GDP/inflation/employment focused
HORIZON_INDICATOR_WEIGHTS = {
    "short": {
        "vix": 1.5,
        "yield_curve_spread": 1.25,
        "unemployment": 0.75,
        "fed_funds_rate": 1.0,
        "cpi_yoy": 0.75,
    },
    "swing": {
        "vix": 1.0,
        "yield_curve_spread": 1.0,
        "unemployment": 1.0,
        "fed_funds_rate": 1.0,
        "cpi_yoy": 1.0,
    },
    "long": {
        "vix": 0.75,
        "yield_curve_spread": 1.0,
        "unemployment": 1.25,
        "fed_funds_rate": 1.25,
        "cpi_yoy": 1.5,
    },
}


def _score_macro_indicator(name: str, value) -> int:
    """Score a single macro indicator as +1 (bullish), 0 (neutral), or -1 (bearish)."""
    if value is None:
        return 0

    thresholds = MACRO_THRESHOLDS.get(name, {})

    # "lower is better" indicators: VIX, unemployment, fed_funds_rate, cpi_yoy
    if name in ("vix", "unemployment", "fed_funds_rate", "cpi_yoy"):
        bullish_below = thresholds.get("bullish_below")
        bearish_above = thresholds.get("bearish_above")
        if bullish_below is not None and value < bullish_below:
            return 1
        if bearish_above is not None and value > bearish_above:
            return -1
        return 0

    # "higher is better" indicators: yield_curve_spread
    if name == "yield_curve_spread":
        bullish_above = thresholds.get("bullish_above")
        bearish_below = thresholds.get("bearish_below")
        if bullish_above is not None and value > bullish_above:
            return 1
        if bearish_below is not None and value < bearish_below:
            return -1
        return 0

    return 0


def macro_econ_node(state: BerkshireState):
    """
    Node for macroeconomic analysis.

    Reads FRED macro indicators from state["data"]["macro_indicators"]
    (VIX, yield curve spread, unemployment, fed funds rate, CPI, GDP,
    treasury 10y), scores each against fixed thresholds with
    horizon-aware weighting, and derives a rating.

    Also computes sector_performance and market_trend features via
    shared.feature_engineering.compute_macro_features() for
    downstream model consumption.
    """
    ticker = state.get("ticker", "UNKNOWN")
    selected_horizon = normalize_horizon(state.get("horizon", "swing"))
    data = state.get("data", {})
    macro_indicators = data.get("macro_indicators", {})
    # --- Evaluation ---

    # --- Edge case: no macro data ---
    if not macro_indicators:
        print(f"[Macro] No macro indicators for {ticker}. Defaulting to hold.")
        return {
            "analyst_signals": {
                "macro": {
                    "rating": Rating.HOLD.value,
                    "confidence": 0.0,
                    "features": {"sector_performance": None, "market_trend": None},
                    "details": "No macroeconomic data available. Defaulting to hold.",
                    "horizon_alignment_note": f"No macro data for {horizon_label(selected_horizon)}.",
                }
            }
        }

    # --- Compute model features (sector_performance, market_trend) ---
    model_features = compute_macro_features(macro_indicators)

    # --- Score individual macro indicators ---
    scorable_indicators = {
        "vix": macro_indicators.get("vix"),
        "yield_curve_spread": macro_indicators.get("yield_curve_spread"),
        "unemployment": macro_indicators.get("unemployment"),
        "fed_funds_rate": macro_indicators.get("fed_funds_rate"),
        "cpi_yoy": macro_indicators.get("cpi_yoy"),
    }

    weights = HORIZON_INDICATOR_WEIGHTS.get(selected_horizon, HORIZON_INDICATOR_WEIGHTS["swing"])
    raw_scores = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for name, value in scorable_indicators.items():
        raw_score = _score_macro_indicator(name, value)
        raw_scores[name] = raw_score
        if value is not None:
            w = weights.get(name, 1.0)
            weighted_sum += raw_score * w
            total_weight += w

    # Map to rating
    if total_weight > 0:
        normalized_score = weighted_sum / total_weight
        score_sum = round(normalized_score * 3)
    else:
        score_sum = 0

    rating = rating_from_aggregate_score(score_sum)
    stance_score = rating_to_score(rating)
    signal = rating_to_signal(rating)


    # --- Confidence ---
    metrics_with_data = sum(1 for v in scorable_indicators.values() if v is not None)
    if metrics_with_data == 0:
        confidence = 0.0
    else:
        weight_by_signal = {1: 0.0, 0: 0.0, -1: 0.0}
        for name, score in raw_scores.items():
            if scorable_indicators.get(name) is not None:
                weight_by_signal[score] += weights.get(name, 1.0)
        
        data_coverage = metrics_with_data / len(MACRO_THRESHOLDS)
        agreement = max(weight_by_signal.values()) / total_weight if total_weight > 0 else 0.0
        confidence = round(data_coverage * agreement, 2)

    # --- Merge features: model features + raw indicators for transparency ---
    all_features = dict(model_features)
    for name, value in scorable_indicators.items():
        if value is not None:
            all_features[name] = round(value, 4) if isinstance(value, float) else value

    # --- Build details string ---
    detail_parts = []
    for name, value in scorable_indicators.items():
        if value is not None:
            score_label = {1: "bullish", 0: "neutral", -1: "bearish"}[raw_scores[name]]
            detail_parts.append(f"{name}={value:.2f} ({score_label})")
    if model_features.get("sector_performance") is not None:
        detail_parts.append(f"sector_performance={model_features['sector_performance']:.4f}")
    if model_features.get("market_trend") is not None:
        detail_parts.append(f"market_trend={model_features['market_trend']}")
    detail_parts.append(f"rating={rating.value}")
    detail_parts.append(f"horizon={selected_horizon}")
    details = "; ".join(detail_parts) if detail_parts else "No macro data available."

    # --- Prepare return ---

    # --- Print result for visibility ---
    label = signal.upper()
    print(
        f"\n[Macro] {ticker}: {label} / {rating.value.upper()} "
        f"(stance_score={stance_score:+d}, confidence: {confidence})"
    )
    for name, value in scorable_indicators.items():
        if value is not None:
            print(f"[Macro]   {name}: {value:.2f} -> {raw_scores[name]:+d}")
    if model_features.get("sector_performance") is not None:
        print(f"[Macro]   sector_performance: {model_features['sector_performance']:.4f}")
    if model_features.get("market_trend") is not None:
        print(f"[Macro]   market_trend: {model_features['market_trend']}")
    print(f"[Macro] Indicators available: {metrics_with_data}/5, weighted_sum: {weighted_sum:.2f}")

    # --- Write to state ---
    return {
        "analyst_signals": {
            "macro": {
                "rating": rating.value,
                "confidence": confidence,
                "features": all_features,
                "details": details,
                "horizon_alignment_note": (
                    f"Macro signals prioritized for {horizon_label(selected_horizon)}: "
                    f"{'VIX and yield curve weighted heavily' if selected_horizon == 'short' else ''}"
                    f"{'balanced macro weighting' if selected_horizon == 'swing' else ''}"
                    f"{'unemployment, CPI, and rates weighted heavily' if selected_horizon == 'long' else ''}."
                ),
            }
        }
    }
