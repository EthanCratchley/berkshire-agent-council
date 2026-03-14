import os
from datetime import datetime, timedelta, timezone

import yfinance
import finnhub
import fredapi
import pandas as pd
from dotenv import load_dotenv

from shared.state_schema import BerkshireState

load_dotenv()


def _ts_to_iso(value) -> str:
    """Convert a pandas Timestamp or datetime to ISO string."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _safe_float(value) -> float:
    """Cast numpy/pandas numeric types to Python float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _stringify_keys(d: dict) -> dict:
    """Recursively convert Timestamp/datetime keys and numpy values to JSON-safe types."""
    result = {}
    for k, v in d.items():
        # Convert Timestamp keys to strings
        if hasattr(k, "isoformat"):
            k = k.isoformat()
        else:
            k = str(k)

        # Recurse on nested dicts
        if isinstance(v, dict):
            v = _stringify_keys(v)
        elif isinstance(v, list):
            v = [_stringify_keys(i) if isinstance(i, dict) else i for i in v]
        elif hasattr(v, "isoformat"):
            v = v.isoformat()
        else:
            # Attempt to cast numpy scalar types to Python native
            try:
                import numpy as np
                if isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (np.floating,)):
                    v = float(v)
                elif isinstance(v, (np.bool_,)):
                    v = bool(v)
            except ImportError:
                pass

        result[k] = v
    return result


def data_fetcher(state: BerkshireState):
    """
    Node to fetch financial data for the given ticker.

    Fetches data from three sources:
    - yfinance: price history, company info, financials, analyst recommendations
    - Finnhub (free tier): company news, basic financials, recommendation trends,
                           earnings surprises, insider sentiment
    - FRED: macroeconomic indicators (GDP, CPI, unemployment, etc.)

    Returns a dict with key "data" containing all fetched data.
    Error keys (*_error) are included only when the respective source fails.
    Each Finnhub sub-call is isolated so one failure does not drop other keys.
    """
    ticker_symbol: str = state["ticker"]
    data: dict = {}

    try:
        ticker = yfinance.Ticker(ticker_symbol)
    except Exception as e:
        print(f"[Data Fetcher] yfinance.Ticker() failed: {e}")
        ticker = None
        data["yfinance_error"] = str(e)

    if ticker is not None:
        # Price history
        try:
            history_df = ticker.history(period="1y").reset_index()
            price_history_records = []
            for record in history_df.to_dict(orient="records"):
                safe_record = {}
                for k, v in record.items():
                    k = _ts_to_iso(k) if hasattr(k, "isoformat") else str(k)
                    if hasattr(v, "isoformat"):
                        v = v.isoformat()
                    else:
                        try:
                            import numpy as np
                            if isinstance(v, (np.integer,)):
                                v = int(v)
                            elif isinstance(v, (np.floating,)):
                                v = float(v)
                        except ImportError:
                            pass
                    safe_record[k] = v
                price_history_records.append(safe_record)
            data["price_history"] = price_history_records
        except Exception as e:
            print(f"[Data Fetcher] Error fetching price history: {e}")
            data["price_history"] = []

        # Company info
        try:
            data["company_info"] = ticker.info
        except Exception:
            data["company_info"] = {}

        # Income Statement
        try:
            financials_df = ticker.financials
            if financials_df is not None and not financials_df.empty:
                data["income_statement"] = _stringify_keys(financials_df.to_dict())
            else:
                data["income_statement"] = {}
        except Exception:
            data["income_statement"] = {}

        # Balance Sheet
        try:
            balance_df = ticker.balance_sheet
            if balance_df is not None and not balance_df.empty:
                data["balance_sheet"] = _stringify_keys(balance_df.to_dict())
            else:
                data["balance_sheet"] = {}
        except Exception:
            data["balance_sheet"] = {}

        # Cash Flow
        try:
            cashflow_df = ticker.cashflow
            if cashflow_df is not None and not cashflow_df.empty:
                data["cash_flow"] = _stringify_keys(cashflow_df.to_dict())
            else:
                data["cash_flow"] = {}
        except Exception:
            data["cash_flow"] = {}

        # Analyst recommendations
        try:
            recs = ticker.recommendations
            if recs is not None and not recs.empty:
                data["analyst_recommendations"] = recs.reset_index().to_dict(orient="records")
            else:
                data["analyst_recommendations"] = []
        except Exception:
            data["analyst_recommendations"] = []
    else:
        data["price_history"] = []
        data["company_info"] = {}
        data["income_statement"] = {}
        data["balance_sheet"] = {}
        data["cash_flow"] = {}
        data["analyst_recommendations"] = []

    # Finnhub block  (free-tier endpoints only)
    # Each sub-call is isolated so one failure does not drop other keys.
    try:
        finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")
        client = finnhub.Client(api_key=finnhub_api_key)

        # Date range helpers
        today = datetime.now(timezone.utc)
        from_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        from_date_1y = (today - timedelta(days=365)).strftime("%Y-%m-%d")

        # News articles (up to 20) — free tier
        try:
            raw_news = client.company_news(ticker_symbol, _from=from_date, to=to_date)
            news_articles = []
            for article in (raw_news or [])[:20]:
                ts = article.get("datetime", 0)
                try:
                    date_iso = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
                except (TypeError, ValueError, OSError):
                    date_iso = str(ts)
                news_articles.append({
                    "headline": article.get("headline", ""),
                    "summary": article.get("summary", ""),
                    "source": article.get("source", ""),
                    "url": article.get("url", ""),
                    "datetime": date_iso,
                })
            data["news_articles"] = news_articles
        except Exception as e:
            data["news_articles"] = []
            data["finnhub_news_error"] = str(e)

        # Basic financials: P/E, 52-week high/low, margins, beta, etc. — free tier
        try:
            basic = client.company_basic_financials(ticker_symbol, "all")
            data["basic_financials"] = basic.get("metric", {}) if basic else {}
        except Exception as e:
            data["basic_financials"] = {}
            data["finnhub_basic_financials_error"] = str(e)

        # Recommendation trends: buy / hold / sell consensus — free tier
        try:
            trends = client.recommendation_trends(ticker_symbol)
            # Keep the 3 most recent periods
            data["recommendation_trends"] = (trends or [])[:3]
        except Exception as e:
            data["recommendation_trends"] = []
            data["finnhub_rec_trends_error"] = str(e)

        # Earnings surprises: actual vs estimate EPS (last 4 quarters) — free tier
        try:
            surprises = client.company_earnings(ticker_symbol, limit=4)
            data["earnings_surprises"] = surprises or []
        except Exception as e:
            data["earnings_surprises"] = []
            data["finnhub_earnings_error"] = str(e)

        # Insider sentiment: monthly share purchase ratio (MSPR) — free tier
        try:
            insider = client.stock_insider_sentiment(
                ticker_symbol, from_date_1y, to_date
            )
            data["insider_sentiment"] = (insider or {}).get("data", [])
        except Exception as e:
            data["insider_sentiment"] = []
            data["finnhub_insider_error"] = str(e)

    except Exception as e:
        data["finnhub_error"] = str(e)

    # FRED block
    try:
        fred_api_key = os.getenv("FRED_API_KEY", "")
        fred = fredapi.Fred(api_key=fred_api_key)

        series_map = {
            "gdp": "GDP",
            "cpi": "CPIAUCSL",
            "unemployment": "UNRATE",
            "fed_funds_rate": "FEDFUNDS",
            "treasury_10y": "GS10",
            "vix": "VIXCLS",
            "yield_curve_spread": "T10Y2Y",
        }

        macro_indicators = {}
        for key, series_id in series_map.items():
            try:
                series = fred.get_series(series_id)
                latest_value = float(series.tail(1).iloc[0])
                macro_indicators[key] = latest_value
            except Exception:
                macro_indicators[key] = None

        data["macro_indicators"] = macro_indicators

    except Exception as e:
        data["fred_error"] = str(e)
    
    print(f"\n[Data Fetcher] Intel gathered for {ticker_symbol}:")
    
    # Check Price Data
    if data.get("price_history"): print("Price History")
    else: print("Price History (Missing)")
        
    # Check Fundamentals (Company Info, Balance Sheet, etc.)
    if data.get("company_info") and data.get("balance_sheet"): 
        print("Fundamentals & Financials")
    else: 
        print("Fundamentals (Missing/Partial - Expected for ETFs)")
        
    # Check News
    if data.get("news_articles"): print(f"News Articles ({len(data['news_articles'])} found)")
    else: print("News Articles (None found)")
        
    print("----------------------------------------\n")

    return {"data": data}
