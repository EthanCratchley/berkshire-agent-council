"""
Unit tests for nodes/data_fetcher.py

All external API calls (yfinance, Finnhub, FRED) are mocked.
Run with: pytest tests/ -v
"""

import json
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

# Helpers: build minimal mock return values that satisfy the real code paths
def _make_yfinance_ticker_mock():
    """Return a MagicMock that mimics a yfinance.Ticker object."""
    mock_ticker = MagicMock()

    # price history DataFrame
    history_df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "Open": [185.0, 186.0],
        "High": [187.0, 188.0],
        "Low": [184.0, 185.0],
        "Close": [186.5, 187.5],
        "Volume": [55_000_000, 60_000_000],
    })
    mock_ticker.history.return_value = history_df

    # company info
    mock_ticker.info = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "marketCap": 2_900_000_000_000,
        "trailingPE": 28.5,
    }

    # income statement
    income_df = pd.DataFrame(
        {"TotalRevenue": [383_000_000_000.0]},
        index=pd.to_datetime(["2023-09-30"]),
    ).T
    mock_ticker.financials = income_df

    # balance sheet
    balance_df = pd.DataFrame(
        {"TotalAssets": [352_000_000_000.0]},
        index=pd.to_datetime(["2023-09-30"]),
    ).T
    mock_ticker.balance_sheet = balance_df

    # cash flow
    cashflow_df = pd.DataFrame(
        {"OperatingCashFlow": [110_000_000_000.0]},
        index=pd.to_datetime(["2023-09-30"]),
    ).T
    mock_ticker.cashflow = cashflow_df

    # analyst recommendations
    recs_df = pd.DataFrame({
        "Firm": ["Goldman Sachs", "Morgan Stanley"],
        "To Grade": ["Buy", "Overweight"],
        "Action": ["main", "main"],
    })
    mock_ticker.recommendations = recs_df

    return mock_ticker


def _make_finnhub_client_mock():
    """Return a MagicMock that mimics a finnhub.Client object (free-tier endpoints)."""
    mock_client = MagicMock()

    # company_news — free tier
    sample_ts = int(datetime(2024, 3, 1, 12, 0, 0).timestamp())
    mock_client.company_news.return_value = [
        {
            "headline": "Apple beats earnings expectations",
            "summary": "Apple reported Q1 earnings of $2.18 per share, beating analyst estimates of $2.10, driven by strong iPhone and services revenue growth.",
            "source": "Reuters",
            "url": "https://reuters.com/apple-earnings",
            "datetime": sample_ts,
        },
        {
            "headline": "Apple Vision Pro ships",
            "summary": "Apple began shipping its mixed-reality Vision Pro headset at $3,499, marking the company's first new product category since the Apple Watch in 2015.",
            "source": "Bloomberg",
            "url": "https://bloomberg.com/apple-vision",
            "datetime": sample_ts,
        },
    ]

    # basic_financials — free tier
    mock_client.company_basic_financials.return_value = {
        "metric": {
            "peRatio": 28.5,
            "52WeekHigh": 199.62,
            "52WeekLow": 124.17,
            "beta": 1.12,
            "roeTTM": 147.25,
            "marketCapitalization": 3_000_000.0,
        },
        "series": {},
    }

    # recommendation_trends — free tier
    mock_client.recommendation_trends.return_value = [
        {
            "period": "2024-03-01",
            "strongBuy": 11,
            "buy": 22,
            "hold": 7,
            "sell": 1,
            "strongSell": 0,
            "symbol": "AAPL",
        }
    ]

    # company_earnings (earnings surprises) — free tier (last 4 quarters)
    mock_client.company_earnings.return_value = [
        {
            "actual": 2.18,
            "estimate": 2.10,
            "period": "2024-03-31",
            "quarter": 2,
            "surprise": 0.08,
            "surprisePercent": 3.81,
            "symbol": "AAPL",
            "year": 2024,
        }
    ]

    # stock_insider_sentiment — free tier
    mock_client.stock_insider_sentiment.return_value = {
        "data": [
            {"symbol": "AAPL", "year": 2024, "month": 3, "change": 5540, "mspr": 12.2}
        ],
        "symbol": "AAPL",
    }

    return mock_client


def _make_fred_series(value: float = 5.33):
    """Return a minimal pandas Series mimicking a FRED API response."""
    return pd.Series([value], index=pd.to_datetime(["2024-02-01"]))


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_successful_fetch_populates_all_keys():
    """Mock all three sources. Assert all expected keys are present, no error keys."""
    mock_yf_ticker = _make_yfinance_ticker_mock()
    mock_fh_client = _make_finnhub_client_mock()
    fred_series = _make_fred_series()

    with patch("nodes.data_fetcher.yfinance.Ticker", return_value=mock_yf_ticker), \
         patch("nodes.data_fetcher.finnhub.Client", return_value=mock_fh_client), \
         patch("nodes.data_fetcher.fredapi.Fred") as MockFred:

        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = fred_series
        MockFred.return_value = mock_fred_instance

        from nodes.data_fetcher import data_fetcher
        result = data_fetcher({"ticker": "AAPL", "data": {}, "analyst_signals": {}})

    expected_keys = [
        "price_history", "company_info", "income_statement", "balance_sheet",
        "cash_flow", "analyst_recommendations",
        "news_articles", "basic_financials", "recommendation_trends",
        "earnings_surprises", "insider_sentiment",
        "macro_indicators",
    ]
    error_keys = ["yfinance_error", "finnhub_error", "fred_error"]

    for key in expected_keys:
        assert key in result["data"], f"Missing key: {key}"
    for ekey in error_keys:
        assert ekey not in result["data"], f"Unexpected error key present: {ekey}"


def test_yfinance_failure_isolated():
    """yfinance raises; Finnhub and FRED succeed. Assert yfinance_error present, others not."""
    mock_fh_client = _make_finnhub_client_mock()
    fred_series = _make_fred_series()

    with patch("nodes.data_fetcher.yfinance.Ticker", side_effect=Exception("yfinance network error")), \
         patch("nodes.data_fetcher.finnhub.Client", return_value=mock_fh_client), \
         patch("nodes.data_fetcher.fredapi.Fred") as MockFred:

        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = fred_series
        MockFred.return_value = mock_fred_instance

        from nodes.data_fetcher import data_fetcher
        result = data_fetcher({"ticker": "AAPL", "data": {}, "analyst_signals": {}})

    assert "yfinance_error" in result["data"], "yfinance_error key missing"
    assert "news_articles" in result["data"], "news_articles should still be present"
    assert "macro_indicators" in result["data"], "macro_indicators should still be present"
    assert "finnhub_error" not in result["data"], "Unexpected finnhub_error"
    assert "fred_error" not in result["data"], "Unexpected fred_error"


def test_finnhub_failure_isolated():
    """Finnhub raises; yfinance and FRED succeed. Assert finnhub_error present, yfinance keys present."""
    mock_yf_ticker = _make_yfinance_ticker_mock()
    fred_series = _make_fred_series()

    with patch("nodes.data_fetcher.yfinance.Ticker", return_value=mock_yf_ticker), \
         patch("nodes.data_fetcher.finnhub.Client", side_effect=Exception("finnhub API error")), \
         patch("nodes.data_fetcher.fredapi.Fred") as MockFred:

        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = fred_series
        MockFred.return_value = mock_fred_instance

        from nodes.data_fetcher import data_fetcher
        result = data_fetcher({"ticker": "AAPL", "data": {}, "analyst_signals": {}})

    assert "finnhub_error" in result["data"], "finnhub_error key missing"
    assert "price_history" in result["data"], "price_history should still be present"
    assert "company_info" in result["data"], "company_info should still be present"
    assert "macro_indicators" in result["data"], "macro_indicators should still be present"
    assert "yfinance_error" not in result["data"], "Unexpected yfinance_error"
    assert "fred_error" not in result["data"], "Unexpected fred_error"


def test_fred_failure_isolated():
    """FRED raises; yfinance and Finnhub succeed. Assert fred_error present, yfinance + finnhub keys present."""
    mock_yf_ticker = _make_yfinance_ticker_mock()
    mock_fh_client = _make_finnhub_client_mock()

    with patch("nodes.data_fetcher.yfinance.Ticker", return_value=mock_yf_ticker), \
         patch("nodes.data_fetcher.finnhub.Client", return_value=mock_fh_client), \
         patch("nodes.data_fetcher.fredapi.Fred", side_effect=Exception("FRED connection refused")):

        from nodes.data_fetcher import data_fetcher
        result = data_fetcher({"ticker": "AAPL", "data": {}, "analyst_signals": {}})

    assert "fred_error" in result["data"], "fred_error key missing"
    assert "price_history" in result["data"], "price_history should still be present"
    assert "news_articles" in result["data"], "news_articles should still be present"
    assert "basic_financials" in result["data"], "basic_financials should still be present"
    assert "recommendation_trends" in result["data"], "recommendation_trends should still be present"
    assert "earnings_surprises" in result["data"], "earnings_surprises should still be present"
    assert "insider_sentiment" in result["data"], "insider_sentiment should still be present"
    assert "yfinance_error" not in result["data"], "Unexpected yfinance_error"
    assert "finnhub_error" not in result["data"], "Unexpected finnhub_error"


def test_output_is_json_serializable():
    """Mock all sources. Pass result['data'] through json.dumps() — assert no exception raised."""
    mock_yf_ticker = _make_yfinance_ticker_mock()
    mock_fh_client = _make_finnhub_client_mock()
    fred_series = _make_fred_series()

    with patch("nodes.data_fetcher.yfinance.Ticker", return_value=mock_yf_ticker), \
         patch("nodes.data_fetcher.finnhub.Client", return_value=mock_fh_client), \
         patch("nodes.data_fetcher.fredapi.Fred") as MockFred:

        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = fred_series
        MockFred.return_value = mock_fred_instance

        from nodes.data_fetcher import data_fetcher
        result = data_fetcher({"ticker": "AAPL", "data": {}, "analyst_signals": {}})

    # Will raise TypeError/ValueError if not serializable — pytest catches and fails the test
    json.dumps(result["data"])


# Runner (for direct execution outside pytest)
if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
