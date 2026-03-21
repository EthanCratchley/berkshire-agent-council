from shared.state_schema import BerkshireState
from shared.feature_engineering import compute_fundamental_features


# Sector-relative scoring thresholds
# Format: {metric: (bullish_below, bearish_above)} for P/E and D/E
# Format: {metric: (bearish_below, bullish_above)} for margins, growth, EPS
SECTOR_THRESHOLDS = {
    "Technology": {
        "pe_ratio": (25, 40),
        "debt_to_equity": (50, 100),
        "profit_margin": (0.05, 0.15),
        "revenue_growth": (0.0, 0.15),
        "eps": (0.0, 3.0),
    },
    "Financial Services": {
        "pe_ratio": (12, 20),
        "debt_to_equity": (150, 300),
        "profit_margin": (0.05, 0.20),
        "revenue_growth": (0.0, 0.08),
        "eps": (0.0, 4.0),
    },
    "Healthcare": {
        "pe_ratio": (20, 35),
        "debt_to_equity": (80, 200),
        "profit_margin": (0.0, 0.15),
        "revenue_growth": (0.0, 0.10),
        "eps": (0.0, 3.0),
    },
    "Energy": {
        "pe_ratio": (10, 18),
        "debt_to_equity": (60, 150),
        "profit_margin": (0.02, 0.10),
        "revenue_growth": (-0.05, 0.05),
        "eps": (0.0, 4.0),
    },
    "Consumer Cyclical": {
        "pe_ratio": (18, 30),
        "debt_to_equity": (80, 200),
        "profit_margin": (0.03, 0.10),
        "revenue_growth": (0.0, 0.08),
        "eps": (0.0, 3.0),
    },
    "Utilities": {
        "pe_ratio": (15, 22),
        "debt_to_equity": (120, 250),
        "profit_margin": (0.05, 0.12),
        "revenue_growth": (-0.02, 0.03),
        "eps": (0.0, 2.0),
    },
}

DEFAULT_THRESHOLDS = {
    "pe_ratio": (15, 25),
    "debt_to_equity": (50, 150),
    "profit_margin": (0.05, 0.20),
    "revenue_growth": (0.0, 0.10),
    "eps": (0.0, 5.0),
}


def _extract_features(data: dict) -> dict:
    """Extract fundamental features using shared feature_engineering module."""
    return compute_fundamental_features(
        company_info=data.get("company_info", {}),
        basic_financials=data.get("basic_financials", {}),
        income_statement=data.get("income_statement", {}),
        balance_sheet=data.get("balance_sheet", {}),
        earnings_surprises=data.get("earnings_surprises", []),
    )


def _score_feature(name: str, value, thresholds: dict) -> int:
    """Score a single feature as +1 (bullish), 0 (neutral), or -1 (bearish)."""
    if value is None:
        return 0

    low, high = thresholds[name]

    if name in ("pe_ratio", "debt_to_equity"):
        # Lower is better
        if value < low:
            return 1
        elif value > high:
            return -1
        return 0
    else:
        # Higher is better (profit_margin, revenue_growth, eps)
        if value > high:
            return 1
        elif value < low:
            return -1
        return 0


def fundamental_node(state: BerkshireState):
    """
    Node for fundamental analysis.

    Extracts financial ratios from state data, scores them using
    sector-relative thresholds, and returns a bullish/bearish/neutral
    signal with numerical features for RF/KNN models.
    """
    ticker = state.get("ticker", "UNKNOWN")
    data = state.get("data", {})

    if not data:
        print(f"[Fundamental] No data available for {ticker}. Defaulting to neutral.")
        return {
            "analyst_signals": {
                "fundamental": {
                    "signal": "neutral",
                    "confidence": 0.0,
                    "features": {k: None for k in DEFAULT_THRESHOLDS},
                    "details": "No data available for fundamental analysis.",
                }
            }
        }

    try:
        features = _extract_features(data)

        # Select sector-relative thresholds
        sector = data.get("company_info", {}).get("sector", "")
        thresholds = SECTOR_THRESHOLDS.get(sector, DEFAULT_THRESHOLDS)

        # Score each feature
        scores = {}
        for name in features:
            scores[name] = _score_feature(name, features[name], thresholds)

        metrics_with_data = sum(1 for v in features.values() if v is not None)
        score_sum = sum(s for name, s in scores.items() if features[name] is not None)

        # Determine signal
        if score_sum > 1:
            signal = "bullish"
        elif score_sum < -1:
            signal = "bearish"
        else:
            signal = "neutral"

        # Calculate confidence
        if metrics_with_data == 0:
            confidence = 0.0
        else:
            data_coverage = metrics_with_data / 5
            agreement = min(abs(score_sum) / metrics_with_data, 1.0)
            confidence = round(data_coverage * agreement, 2)

        # Build details string
        detail_parts = []
        if sector:
            detail_parts.append(f"Sector: {sector}")
        for name, value in features.items():
            if value is not None:
                score_label = {1: "bullish", 0: "neutral", -1: "bearish"}[scores[name]]
                detail_parts.append(f"{name}={value:.2f} ({score_label})")
        details = "; ".join(detail_parts) if detail_parts else "No fundamental data available."

        # Debug output
        label = signal.upper()
        print(f"\n[Fundamental] {ticker}: {label} (confidence: {confidence})")
        if sector:
            print(f"[Fundamental] Sector: {sector} (using sector-relative thresholds)")
        for name, value in features.items():
            if value is not None:
                print(f"[Fundamental]   {name}: {value:.2f} -> {scores[name]:+d}")
        print(f"[Fundamental] Metrics available: {metrics_with_data}/5, score sum: {score_sum}")

        return {
            "analyst_signals": {
                "fundamental": {
                    "signal": signal,
                    "confidence": confidence,
                    "features": features,
                    "details": details,
                }
            }
        }

    except Exception as e:
        print(f"[Fundamental] Unexpected error for {ticker}: {e}")
        return {
            "analyst_signals": {
                "fundamental": {
                    "signal": "neutral",
                    "confidence": 0.0,
                    "features": {k: None for k in DEFAULT_THRESHOLDS},
                    "details": f"Error during fundamental analysis: {str(e)}",
                }
            }
        }
