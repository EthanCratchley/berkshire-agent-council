"""
Build the cached training dataset for RF model training.

Fetches 2 years of historical data for 50+ S&P 500 tickers, computes
all 16 features per trading day, labels with 5-day forward returns,
and saves to data/cached_dataset.csv.
"""

import os
import sys
import time
import logging

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

# Add project root to path so we can import shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.feature_engineering import (
    FEATURE_ORDER,
    compute_technical_features,
    compute_fundamental_features,
    compute_macro_features,
    compute_sentiment_features,
    compute_sector_features,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Suppress yfinance noise
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cached_dataset.csv")

# 50+ S&P 500 tickers across sectors
TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO", "ORCL", "CRM", "AMD", "INTC",
    # Financial Services
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "AXP", "C",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX",
    # Consumer Cyclical
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
    # Consumer Staples
    "PG", "KO", "PEP", "WMT", "COST", "CL",
    # Industrials
    "CAT", "BA", "HON", "UPS", "GE", "RTX",
    # Utilities
    "NEE", "DUK", "SO", "D",
    # Communication
    "DIS", "NFLX", "CMCSA", "T",
]

# 5-class label thresholds based on 5-day forward return
LABEL_THRESHOLDS = [
    (0.05, "STRONG BUY"),   # > +5%
    (0.02, "BUY"),          # +2% to +5%
    (-0.02, "HOLD"),        # -2% to +2%
    (-0.05, "SELL"),        # -5% to -2%
]


def classify_return(forward_return: float) -> str:
    """Map a 5-day forward return to one of 5 class labels."""
    if forward_return > 0.05:
        return "STRONG BUY"
    elif forward_return > 0.02:
        return "BUY"
    elif forward_return > -0.02:
        return "HOLD"
    elif forward_return > -0.05:
        return "SELL"
    else:
        return "STRONG SELL"


def fetch_fred_macro_series() -> pd.DataFrame:
    """Fetch 2 years of FRED macro indicators, forward-filled to daily."""
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        logger.warning("FRED_API_KEY not set. Macro features will be None.")
        return pd.DataFrame()

    try:
        fred = Fred(api_key=fred_key)

        series_ids = {
            "gdp": "GDP",
            "cpi": "CPIAUCSL",
            "unemployment": "UNRATE",
            "fed_funds_rate": "FEDFUNDS",
            "treasury_10y": "GS10",
            "vix": "VIXCLS",
        }

        frames = {}
        for name, series_id in series_ids.items():
            try:
                s = fred.get_series(series_id, observation_start="2023-01-01")
                frames[name] = s
            except Exception as e:
                logger.warning(f"Failed to fetch FRED series {series_id}: {e}")

        if not frames:
            return pd.DataFrame()

        macro_df = pd.DataFrame(frames)
        macro_df = macro_df.asfreq("D").ffill()  # Forward-fill to daily

        # Compute yield curve spread (10Y - 2Y proxy: 10Y - fed_funds)
        if "treasury_10y" in macro_df.columns and "fed_funds_rate" in macro_df.columns:
            macro_df["yield_curve_spread"] = macro_df["treasury_10y"] - macro_df["fed_funds_rate"]

        return macro_df

    except Exception as e:
        logger.warning(f"Failed to initialize FRED: {e}")
        return pd.DataFrame()


def get_macro_for_date(macro_df: pd.DataFrame, date) -> dict:
    """Look up macro indicators for a specific date (nearest available)."""
    if macro_df.empty:
        return {}

    date = pd.Timestamp(date).tz_localize(None)  # Strip timezone for comparison with FRED
    # Find the most recent available date <= target date
    available = macro_df.index[macro_df.index <= date]
    if len(available) == 0:
        return {}

    row = macro_df.loc[available[-1]]
    return {k: float(v) if pd.notna(v) else None for k, v in row.items()}


def build_ticker_data(ticker: str, macro_df: pd.DataFrame) -> list[dict]:
    """Build feature rows for a single ticker."""
    rows = []

    try:
        stock = yf.Ticker(ticker)
        price_hist = stock.history(period="2y")

        if price_hist.empty or len(price_hist) < 60:
            logger.warning(f"{ticker}: insufficient price history ({len(price_hist)} rows). Skipping.")
            return []

        info = {}
        try:
            info = stock.info or {}
        except Exception:
            pass

        # Fetch financials for fundamental features
        basic_financials = {}
        income_statement = {}
        balance_sheet = {}
        earnings_surprises = []

        try:
            inc = stock.income_stmt
            if inc is not None and not inc.empty:
                income_statement = inc.to_dict()
        except Exception:
            pass

        try:
            bs = stock.balance_sheet
            if bs is not None and not bs.empty:
                balance_sheet = bs.to_dict()
        except Exception:
            pass

        # Compute fundamental features once (static per ticker)
        fund_features = compute_fundamental_features(
            company_info=info,
            basic_financials=basic_financials,
            income_statement=income_statement,
            balance_sheet=balance_sheet,
            earnings_surprises=earnings_surprises,
        )

        # Sentiment defaults (no historical LLM calls)
        sent_features = compute_sentiment_features(sentiment_score=5.0, news_volume=0)

        # Sector one-hot encoding (static per ticker)
        sector = info.get("sector", "")
        sector_features = compute_sector_features(sector)

        dates = price_hist.index.tolist()

        # Start from day 50 (need lookback for technical indicators)
        # Stop 5 days before end (need forward return for label)
        for i in range(50, len(dates) - 5):
            date = dates[i]

            # Technical features from price slice up to current day
            price_slice = price_hist.iloc[: i + 1]
            tech_features = compute_technical_features(price_slice)

            # Macro features for this date
            macro_indicators = get_macro_for_date(macro_df, date)
            macro_features = compute_macro_features(macro_indicators)

            # 5-day forward return for label
            current_close = price_hist.iloc[i]["Close"]
            future_close = price_hist.iloc[i + 5]["Close"]
            forward_return = (future_close - current_close) / current_close
            label = classify_return(forward_return)

            # Merge all features
            all_features = {}
            all_features.update(tech_features)
            all_features.update(sent_features)
            all_features.update(fund_features)
            all_features.update(macro_features)
            all_features.update(sector_features)

            row = {
                "ticker": ticker,
                "date": date.strftime("%Y-%m-%d"),
            }
            for feat_name in FEATURE_ORDER:
                row[feat_name] = all_features.get(feat_name)
            row["label"] = label

            rows.append(row)

    except Exception as e:
        logger.error(f"{ticker}: unexpected error: {e}")

    return rows


def build_dataset():
    """Main entry point: build the full dataset and save to CSV."""
    logger.info(f"Building dataset for {len(TICKERS)} tickers...")

    # Fetch FRED macro data once (shared across all tickers)
    logger.info("Fetching FRED macro series...")
    macro_df = fetch_fred_macro_series()
    if not macro_df.empty:
        logger.info(f"FRED data: {len(macro_df)} daily rows, columns: {list(macro_df.columns)}")
    else:
        logger.warning("No FRED data available. Macro features will be None.")

    all_rows = []
    failed_tickers = []

    for idx, ticker in enumerate(TICKERS, 1):
        logger.info(f"[{idx}/{len(TICKERS)}] Processing {ticker}...")
        try:
            rows = build_ticker_data(ticker, macro_df)
            if rows:
                all_rows.extend(rows)
                logger.info(f"  {ticker}: {len(rows)} samples generated")
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            logger.error(f"  {ticker}: failed with {e}")
            failed_tickers.append(ticker)

        # Rate limiting
        if idx < len(TICKERS):
            time.sleep(0.5)

    if not all_rows:
        logger.error("No data generated. Check API keys and network connection.")
        return

    df = pd.DataFrame(all_rows)

    # Sort by date for reproducibility
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    # Summary
    logger.info(f"\nDataset saved to {OUTPUT_PATH}")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Tickers processed: {len(TICKERS) - len(failed_tickers)}/{len(TICKERS)}")
    if failed_tickers:
        logger.info(f"Failed tickers: {failed_tickers}")
    logger.info(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")
    logger.info(f"\nFeature null counts:\n{df[FEATURE_ORDER].isnull().sum().to_string()}")


if __name__ == "__main__":
    build_dataset()
