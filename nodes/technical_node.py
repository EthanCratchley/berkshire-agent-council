import pandas as pd

from shared.state_schema import BerkshireState
from shared.horizon import normalize_horizon, horizon_label
from shared.feature_engineering import compute_technical_features
from shared.stance import Rating, rating_from_aggregate_score, rating_to_score, rating_to_signal


# Scoring thresholds for each technical indicator
# Format: (bullish_threshold, bearish_threshold)
INDICATOR_THRESHOLDS = {
    "rsi": {"bullish_below": 35, "bearish_above": 65},
    "macd_histogram": {"bullish_above": 0.0, "bearish_below": 0.0},
    "sma_20_50_cross": {"bullish_eq": 1, "bearish_eq": 0},
    "bollinger_pct": {"bullish_below": 0.2, "bearish_above": 0.8},
    "volume_ratio": {"bullish_above": 1.5, "bearish_below": 0.7},
    "price_change_5d": {"bullish_above": 0.02, "bearish_below": -0.02},
    "price_change_20d": {"bullish_above": 0.05, "bearish_below": -0.05},
}

# Horizon-based indicator weights
# Short-term: momentum/oscillator heavy; Long-term: trend heavy
HORIZON_INDICATOR_WEIGHTS = {
    "short": {
        "rsi": 1.5, "macd_histogram": 1.25, "sma_20_50_cross": 0.75,
        "bollinger_pct": 1.25, "volume_ratio": 1.25,
        "price_change_5d": 1.5, "price_change_20d": 0.75,
    },
    "swing": {
        "rsi": 1.0, "macd_histogram": 1.0, "sma_20_50_cross": 1.0,
        "bollinger_pct": 1.0, "volume_ratio": 1.0,
        "price_change_5d": 1.0, "price_change_20d": 1.0,
    },
    "long": {
        "rsi": 0.75, "macd_histogram": 0.75, "sma_20_50_cross": 1.5,
        "bollinger_pct": 0.75, "volume_ratio": 0.75,
        "price_change_5d": 0.75, "price_change_20d": 1.5,
    },
}


def _score_indicator(name: str, value, trend_direction: float = 0.0) -> int:
    """Score a single technical indicator as +1 (bullish), 0 (neutral), or -1 (bearish)."""
    if value is None:
        return 0

    thresholds = INDICATOR_THRESHOLDS.get(name, {})

    if name == "sma_20_50_cross":
        if value == thresholds.get("bullish_eq"):
            return 1
        return -1

    if name == "rsi":
        if value < thresholds.get("bullish_below", 35):
            return 1  # Oversold = buying opportunity
        if value > thresholds.get("bearish_above", 65):
            return -1  # Overbought = sell signal
        return 0

    if name == "bollinger_pct":
        if value < thresholds.get("bullish_below", 0.2):
            return 1  # Near lower band = potential bounce
        if value > thresholds.get("bearish_above", 0.8):
            return -1  # Near upper band = potential reversal
        return 0

    # Standard above/below thresholds (MACD, volume, price changes)
    bullish_above = thresholds.get("bullish_above")
    bearish_below = thresholds.get("bearish_below")

    if name == "volume_ratio":
        if bullish_above is not None and value > bullish_above:
            if trend_direction is None:
                return 0  # Unknown trend — volume anomaly is ambiguous
            return 1 if trend_direction >= 0 else -1
        return 0

    if bullish_above is not None and value > bullish_above:
        return 1
    if bearish_below is not None and value < bearish_below:
        return -1
    return 0


def _build_price_dataframe(price_history: list) -> pd.DataFrame:
    """Convert price_history records from data fetcher into a DataFrame."""
    if not price_history:
        return pd.DataFrame()

    df = pd.DataFrame(price_history)

    # Ensure required columns exist
    required = {"Close", "Volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    return df


def technical_node(state: BerkshireState):
    """
    Node for technical analysis of stock data.

    Computes 7 technical indicators from price history using
    shared.feature_engineering, scores each indicator, and
    derives a rating (strong_buy/buy/hold/sell/strong_sell)
    with horizon-aware weighting.

    Indicators:
        - RSI (14-period)
        - MACD Histogram
        - SMA 20/50 Cross (golden/death)
        - Bollinger %B
        - Volume Ratio (vs 20-day avg)
        - 5-Day Price Change
        - 20-Day Price Change
    """
    ticker = state.get("ticker", "UNKNOWN")
    selected_horizon = normalize_horizon(state.get("horizon", "swing"))
    data = state.get("data", {})
    price_history = data.get("price_history", [])
    debate = state.get("debate", {})
    prior_technical = state.get("analyst_signals", {}).get("technical", {})
    prior_rating = prior_technical.get("rating")

    # --- Debate context ---
    is_debate_turn = False
    debate_context_str = ""
    active_challenge = debate.get("active_challenge", {})
    awaiting = debate.get("awaiting_response_from")
    if isinstance(active_challenge, dict) and awaiting == "technical":
        is_debate_turn = True
        my_case = active_challenge.get("my_case", {})
        opponent_case = active_challenge.get("opponent_case", {})
        debate_context_str = (
            f"Challenged by {opponent_case.get('analyst', 'opponent')} "
            f"(rating={opponent_case.get('rating', '')}, "
            f"confidence={opponent_case.get('confidence', '')}). "
            f"Reason: {active_challenge.get('reason', 'N/A')}."
        )

    # --- Edge case: no price data ---
    if not price_history:
        print(f"[Technical] No price history for {ticker}. Defaulting to hold.")
        position_changed = prior_rating not in (None, Rating.HOLD.value)
        return {
            "analyst_signals": {
                "technical": {
                    "rating": Rating.HOLD.value,
                    "confidence": 0.0,
                    "features": {k: None for k in INDICATOR_THRESHOLDS},
                    "details": "No price history available for technical analysis.",
                    "debate_response": "No price data available to revise stance; defaulting to hold.",
                    "position_changed": position_changed,
                    "counterpoints_addressed": [],
                    "claims_conceded": [],
                    "claims_disputed": [],
                    "final_position": {
                        "rating": Rating.HOLD.value,
                        "confidence": 0.0,
                    },
                    "weighting_statement": "No evidence available.",
                    "horizon_alignment_note": (
                        f"No technical data for {horizon_label(selected_horizon)}."
                    ),
                }
            }
        }

    # --- Compute features ---
    price_df = _build_price_dataframe(price_history)
    if price_df.empty:
        print(f"[Technical] Could not build price DataFrame for {ticker}. Defaulting to hold.")
        position_changed = prior_rating not in (None, Rating.HOLD.value)
        return {
            "analyst_signals": {
                "technical": {
                    "rating": Rating.HOLD.value,
                    "confidence": 0.0,
                    "features": {k: None for k in INDICATOR_THRESHOLDS},
                    "details": "Price data could not be parsed for technical analysis.",
                    "debate_response": "Price data unparseable; holding neutral stance.",
                    "position_changed": position_changed,
                    "counterpoints_addressed": [],
                    "claims_conceded": [],
                    "claims_disputed": [],
                    "final_position": {
                        "rating": Rating.HOLD.value,
                        "confidence": 0.0,
                    },
                    "weighting_statement": "No usable evidence.",
                    "horizon_alignment_note": (
                        f"Technical analysis unavailable for {horizon_label(selected_horizon)}."
                    ),
                }
            }
        }

    features = compute_technical_features(price_df)

    # --- Score each indicator with horizon weighting ---
    weights = HORIZON_INDICATOR_WEIGHTS.get(selected_horizon, HORIZON_INDICATOR_WEIGHTS["swing"])
    raw_scores = {}
    weighted_sum = 0.0
    total_weight = 0.0

    trend_val = features.get("price_change_20d")
    trend_direction = trend_val  # None stays None — don't assume direction

    for name, value in features.items():
        raw_score = _score_indicator(name, value, trend_direction)
        raw_scores[name] = raw_score
        if value is not None:
            w = weights.get(name, 1.0)
            weighted_sum += raw_score * w
            total_weight += w

    # Round weighted sum to integer for rating lookup
    if total_weight > 0:
        normalized_score = weighted_sum / total_weight
        # Scale to the aggregate score range expected by rating_from_aggregate_score
        score_sum = round(normalized_score * 3)  # scale factor: max ±3
    else:
        score_sum = 0

    rating = rating_from_aggregate_score(score_sum)
    stance_score = rating_to_score(rating)
    signal = rating_to_signal(rating)
    position_changed = prior_rating is not None and prior_rating != rating.value

    # --- Confidence ---
    metrics_with_data = sum(1 for v in features.values() if v is not None)
    if metrics_with_data == 0:
        confidence = 0.0
    else:
        weight_by_signal = {1: 0.0, 0: 0.0, -1: 0.0}
        for name, score in raw_scores.items():
            if features.get(name) is not None:
                weight_by_signal[score] += weights.get(name, 1.0)
        
        data_coverage = metrics_with_data / len(INDICATOR_THRESHOLDS)
        agreement = max(weight_by_signal.values()) / total_weight if total_weight > 0 else 0.0
        confidence = round(data_coverage * agreement, 2)

    # --- Build details string ---
    detail_parts = []
    for name, value in features.items():
        if value is not None:
            score_label = {1: "bullish", 0: "neutral", -1: "bearish"}[raw_scores[name]]
            if isinstance(value, float):
                detail_parts.append(f"{name}={value:.4f} ({score_label})")
            else:
                detail_parts.append(f"{name}={value} ({score_label})")
    detail_parts.append(f"rating={rating.value}")
    detail_parts.append(f"horizon={selected_horizon}")
    details = "; ".join(detail_parts) if detail_parts else "No technical data available."

    # --- Debate response ---
    if is_debate_turn and debate_context_str:
        bullish_indicators = [n for n, s in raw_scores.items() if s == 1 and features[n] is not None]
        bearish_indicators = [n for n, s in raw_scores.items() if s == -1 and features[n] is not None]
        debate_response = (
            f"Technical indicators support a {signal} stance. "
            f"Bullish signals: {', '.join(bullish_indicators) or 'none'}. "
            f"Bearish signals: {', '.join(bearish_indicators) or 'none'}. "
            f"Weighted score={score_sum:+d} for {horizon_label(selected_horizon)}. "
            f"{debate_context_str}"
        )
        opponent_rating = opponent_case.get("rating", "unknown") if opponent_case else "unknown"
        claims_conceded = []
        claims_disputed = []
        if signal == "bullish":
            if bearish_indicators:
                claims_conceded.append(f"Acknowledged bearish signals ({', '.join(bearish_indicators)}), but {len(bullish_indicators)} bullish indicator(s) dominate.")
            if bullish_indicators:
                claims_disputed.append(f"{len(bullish_indicators)} bullish indicator(s) contradict the opponent's {opponent_rating} thesis.")
        elif signal == "bearish":
            if bullish_indicators:
                claims_conceded.append(f"Acknowledged bullish signals ({', '.join(bullish_indicators)}), but {len(bearish_indicators)} bearish indicator(s) dominate.")
            if bearish_indicators:
                claims_disputed.append(f"{len(bearish_indicators)} bearish indicator(s) contradict the opponent's {opponent_rating} thesis.")
        else:
            if bullish_indicators or bearish_indicators:
                claims_conceded.append(f"Acknowledged mixed signals ({len(bullish_indicators)} bullish, {len(bearish_indicators)} bearish) which prevent a strong stance.")
            claims_disputed.append("The overall lack of consensus among indicators contradicts any strong directional thesis.")
    else:
        debate_response = f"Maintaining {signal} stance based on {metrics_with_data} technical indicators."
        claims_conceded = []
        claims_disputed = []

    weighting_statement = (
        f"Horizon={selected_horizon}: momentum/oscillator signals weighted "
        f"{'heavily' if selected_horizon == 'short' else 'less'} vs trend signals "
        f"weighted {'heavily' if selected_horizon == 'long' else 'normally'}."
    )

    # --- Print result for visibility ---
    label = signal.upper()
    print(
        f"\n[Technical] {ticker}: {label} / {rating.value.upper()} "
        f"(stance_score={stance_score:+d}, confidence: {confidence})"
    )
    for name, value in features.items():
        if value is not None:
            fmt_val = f"{value:.4f}" if isinstance(value, float) else str(value)
            print(f"[Technical]   {name}: {fmt_val} -> {raw_scores[name]:+d}")
    print(f"[Technical] Indicators available: {metrics_with_data}/7, weighted_sum: {weighted_sum:.2f}")

    # --- Write to state ---
    return {
        "analyst_signals": {
            "technical": {
                "rating": rating.value,
                "confidence": confidence,
                "features": features,
                "details": details,
                "debate_response": (
                    debate_response
                    if is_debate_turn
                    else f"Maintaining {signal} stance based on current technical indicators."
                ),
                "position_changed": position_changed,
                "counterpoints_addressed": claims_disputed if is_debate_turn else [],
                "claims_conceded": claims_conceded if is_debate_turn else [],
                "claims_disputed": claims_disputed if is_debate_turn else [],
                "final_position": {
                    "rating": rating.value,
                    "confidence": confidence,
                },
                "weighting_statement": weighting_statement if is_debate_turn else "",
                "horizon_alignment_note": (
                    f"Technical signals prioritized for {horizon_label(selected_horizon)}: "
                    f"{'momentum/oscillators weighted heavily' if selected_horizon == 'short' else ''}"
                    f"{'balanced indicator weighting' if selected_horizon == 'swing' else ''}"
                    f"{'trend indicators (SMA, 20d change) weighted heavily' if selected_horizon == 'long' else ''}."
                ),
            }
        }
    }
