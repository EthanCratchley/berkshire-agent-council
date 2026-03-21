import pandas as pd


# Sectors used for one-hot encoding. Order matters — must be stable.
SECTORS = [
    "Technology",
    "Financial Services",
    "Healthcare",
    "Energy",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Industrials",
    "Utilities",
    "Communication Services",
]

# Canonical feature order — single source of truth for model input shape.
# Both build_dataset.py and orchestrator use this to guarantee consistent ordering.
FEATURE_ORDER = [
    # Technical (7)
    "rsi",
    "macd_histogram",
    "sma_20_50_cross",
    "bollinger_pct",
    "volume_ratio",
    "price_change_5d",
    "price_change_20d",
    # Sentiment (2)
    "sentiment_score",
    "news_volume",
    # Fundamental (5)
    "pe_ratio",
    "debt_to_equity",
    "profit_margin",
    "revenue_growth",
    "eps",
    # Macro (2)
    "sector_performance",
    "market_trend",
    # Sector one-hot (9)
] + [f"sector_{s.lower().replace(' ', '_')}" for s in SECTORS]


def compute_technical_features(price_df: pd.DataFrame) -> dict:
    """Compute technical indicators from price history."""
    features = {
        "rsi": None,
        "macd_histogram": None,
        "sma_20_50_cross": None,
        "bollinger_pct": None,
        "volume_ratio": None,
        "price_change_5d": None,
        "price_change_20d": None,
    }

    if price_df is None or len(price_df) < 2:
        return features

    close = price_df["Close"].astype(float)
    volume = price_df["Volume"].astype(float) if "Volume" in price_df.columns else None

    # RSI (14-period)
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = gain.rolling(window=14, min_periods=14).mean().iloc[-1]
        avg_loss = loss.rolling(window=14, min_periods=14).mean().iloc[-1]
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            features["rsi"] = round(100 - (100 / (1 + rs)), 2)
        else:
            features["rsi"] = 100.0

    # MACD histogram: (EMA12 - EMA26) - Signal(EMA9 of MACD line)
    if len(close) >= 35:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        features["macd_histogram"] = round(float(macd_line.iloc[-1] - signal_line.iloc[-1]), 4)

    # SMA 20/50 cross: 1 = golden cross (SMA20 > SMA50), 0 = death cross
    if len(close) >= 50:
        sma20 = close.rolling(window=20).mean().iloc[-1]
        sma50 = close.rolling(window=50).mean().iloc[-1]
        features["sma_20_50_cross"] = 1 if sma20 > sma50 else 0

    # Bollinger %B: (Close - lower) / (upper - lower), 20-period, 2 std
    if len(close) >= 20:
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        band_width = upper.iloc[-1] - lower.iloc[-1]
        if band_width != 0:
            features["bollinger_pct"] = round(
                float((close.iloc[-1] - lower.iloc[-1]) / band_width), 4
            )

    # Volume ratio: current volume / 20-day average volume
    if volume is not None and len(volume) >= 20:
        avg_vol = volume.rolling(window=20).mean().iloc[-1]
        if avg_vol != 0:
            features["volume_ratio"] = round(float(volume.iloc[-1] / avg_vol), 4)

    # Price change 5d
    if len(close) >= 6:
        prev = close.iloc[-6]
        if prev != 0:
            features["price_change_5d"] = round(float((close.iloc[-1] - prev) / prev), 4)

    # Price change 20d
    if len(close) >= 21:
        prev = close.iloc[-21]
        if prev != 0:
            features["price_change_20d"] = round(float((close.iloc[-1] - prev) / prev), 4)

    return features


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


def compute_fundamental_features(
    company_info: dict,
    basic_financials: dict,
    income_statement: dict,
    balance_sheet: dict,
    earnings_surprises: list,
) -> dict:
    """Extract pe_ratio, debt_to_equity, profit_margin, revenue_growth, eps.

    Uses multiple fallback sources for each metric. Returns None for any
    metric that cannot be determined.
    """
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


def compute_macro_features(macro_indicators: dict) -> dict:
    """Derive sector_performance and market_trend from FRED macro data.

    sector_performance: composite score in [-1.0, 1.0] based on VIX,
                        yield curve spread, and GDP trend.
    market_trend: binary (1=bullish, 0=bearish) based on macro conditions.
    """
    features = {
        "sector_performance": None,
        "market_trend": None,
    }

    if not macro_indicators:
        return features

    vix = macro_indicators.get("vix")
    yield_spread = macro_indicators.get("yield_curve_spread")
    unemployment = macro_indicators.get("unemployment")

    # sector_performance: composite of inverted VIX + yield curve + low unemployment
    score_components = []
    if vix is not None:
        # VIX < 15 = calm (+1), 15-25 = normal (0), > 25 = fearful (-1)
        if vix < 15:
            score_components.append(1.0)
        elif vix > 25:
            score_components.append(-1.0)
        else:
            score_components.append((20 - vix) / 10.0)  # linear scale around 20

    if yield_spread is not None:
        # Positive spread = healthy economy, negative = recession signal
        if yield_spread > 1.0:
            score_components.append(1.0)
        elif yield_spread < -0.5:
            score_components.append(-1.0)
        else:
            score_components.append(yield_spread / 1.0)

    if unemployment is not None:
        # < 4% = strong (+1), 4-6% = moderate (0), > 6% = weak (-1)
        if unemployment < 4.0:
            score_components.append(1.0)
        elif unemployment > 6.0:
            score_components.append(-1.0)
        else:
            score_components.append((5.0 - unemployment) / 2.0)

    if score_components:
        features["sector_performance"] = round(
            sum(score_components) / len(score_components), 4
        )

    # market_trend: bullish if yield curve positive AND VIX reasonable AND low unemployment
    bullish_conditions = 0
    total_conditions = 0

    if yield_spread is not None:
        total_conditions += 1
        if yield_spread > 0:
            bullish_conditions += 1

    if vix is not None:
        total_conditions += 1
        if vix < 25:
            bullish_conditions += 1

    if unemployment is not None:
        total_conditions += 1
        if unemployment < 5.0:
            bullish_conditions += 1

    if total_conditions > 0:
        features["market_trend"] = 1 if bullish_conditions >= (total_conditions / 2) else 0

    return features


def compute_sentiment_features(sentiment_score: float, news_volume: int) -> dict:
    """Passthrough wrapper for sentiment features.

    In the live pipeline, these come from the LLM-based sentiment_node.
    In the training pipeline, these are approximated (default neutral).
    """
    return {
        "sentiment_score": sentiment_score,
        "news_volume": news_volume,
    }


def compute_sector_features(sector: str) -> dict:
    """One-hot encode the sector so the RF can learn sector-specific patterns.
    """
    features = {f"sector_{s.lower().replace(' ', '_')}": 0 for s in SECTORS}
    key = f"sector_{sector.lower().replace(' ', '_')}" if sector else ""
    if key in features:
        features[key] = 1
    return features


def assemble_feature_vector(all_features: dict) -> list:
    """Assemble a feature vector in canonical FEATURE_ORDER"""
    vector = []
    for name in FEATURE_ORDER:
        val = all_features.get(name)
        vector.append(float(val) if val is not None else 0.0)
    return vector
