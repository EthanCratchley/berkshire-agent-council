from shared.state_schema import BerkshireState


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


def _get_most_recent_value(statement: dict, row_key: str):
    """Extract the most recent column value from a financial statement dict."""
    row = statement.get(row_key, {})
    if not row:
        return None
    sorted_dates = sorted(row.keys(), reverse=True)
    for date in sorted_dates:
        val = row[date]
        if val is not None:
            return float(val)
    return None


def _extract_features(data: dict) -> dict:
    """Extract the 5 fundamental features from state data with fallbacks."""
    company_info = data.get("company_info", {})
    basic_financials = data.get("basic_financials", {})
    income_statement = data.get("income_statement", {})
    balance_sheet = data.get("balance_sheet", {})
    earnings_surprises = data.get("earnings_surprises", [])

    metric = basic_financials.get("metric", {}) if isinstance(basic_financials, dict) else {}

    features = {
        "pe_ratio": None,
        "debt_to_equity": None,
        "profit_margin": None,
        "revenue_growth": None,
        "eps": None,
    }

    # P/E ratio
    try:
        val = company_info.get("trailingPE")
        if val is None:
            val = metric.get("peRatio")
        if val is not None:
            features["pe_ratio"] = float(val)
    except (TypeError, ValueError):
        pass

    # Debt-to-equity
    try:
        val = company_info.get("debtToEquity")
        if val is not None:
            features["debt_to_equity"] = float(val)
        else:
            total_debt = _get_most_recent_value(balance_sheet, "Total Debt")
            equity = _get_most_recent_value(balance_sheet, "Stockholders Equity")
            if total_debt is not None and equity is not None and equity != 0:
                features["debt_to_equity"] = total_debt / equity
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    # Profit margin
    try:
        val = company_info.get("profitMargins")
        if val is not None:
            features["profit_margin"] = float(val)
        else:
            net_income = _get_most_recent_value(income_statement, "Net Income")
            revenue = _get_most_recent_value(income_statement, "Total Revenue")
            if net_income is not None and revenue is not None and revenue != 0:
                features["profit_margin"] = net_income / revenue
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    # Revenue growth
    try:
        val = company_info.get("revenueGrowth")
        if val is not None:
            features["revenue_growth"] = float(val)
        else:
            rev_row = income_statement.get("Total Revenue", {})
            if len(rev_row) >= 2:
                sorted_dates = sorted(rev_row.keys(), reverse=True)
                recent = float(rev_row[sorted_dates[0]])
                prior = float(rev_row[sorted_dates[1]])
                if prior != 0:
                    features["revenue_growth"] = (recent - prior) / abs(prior)
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    # EPS
    try:
        val = company_info.get("trailingEps")
        if val is not None:
            features["eps"] = float(val)
        elif earnings_surprises and isinstance(earnings_surprises, list):
            actual = earnings_surprises[0].get("actual")
            if actual is not None:
                features["eps"] = float(actual)
    except (TypeError, ValueError, IndexError):
        pass

    return features


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
