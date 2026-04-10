# CMPT 310 - Project Write-Up: Berkshire Agent Council

**Group Members:** Ethan Cratchley, Jayden Truong, Xavier Rahman
**Date:** April 12, 2026

---

## 1. System Explanation

### 1.1 Overview

The Berkshire Agent Council is a multi-agent stock analysis system that generates investment recommendations (strong buy, buy, hold, sell, strong sell) for any publicly traded stock. Given a ticker symbol and a time horizon (short-term, swing, or long-term), the system fetches real-time market data from three independent APIs, runs four specialized AI analyst agents that each evaluate the stock from a different perspective, facilitates structured debate between analysts that disagree, and then combines the debate outcome with pre-trained Random Forest and KNN classifiers in a 3-way vote to produce a final recommendation with an explainable rationale.

The problem this solves is the challenge of integrating diverse, often contradictory signals in financial analysis. A stock might have strong technical momentum but weak fundamentals, or positive news sentiment but deteriorating macro conditions. Rather than relying on a single model or a simple weighted average, our approach uses an LLM-powered debate protocol where analysts with conflicting views must defend or revise their positions with evidence, combined with classical ML models trained on 2 years of historical data across 128 tickers. This multi-method ensemble produces more robust recommendations than any single approach alone, and the debate transcript provides transparency into how the recommendation was reached.

### 1.2 AI Methods

**LLM-Based Analyst Agents (Google Gemini 2.5 Flash):** Four analyst agents (sentiment, technical, fundamental, macro-economic) each produce a rating with a confidence score. The sentiment and debate nodes use Gemini to analyze news headlines and argue positions. Gemini was chosen for its fast inference speed and generous free-tier API limits, which are important for a system that makes multiple LLM calls per analysis (initial ratings + multiple debate turns).

**Orchestrated Debate (LangGraph State Machine):** The orchestrator detects contradictions between analysts by computing severity = score_distance x min_confidence. When severity exceeds a threshold (1.0), it identifies the outlier analyst and dispatches a debate turn where that analyst must respond to the opposing case. Rating changes are bounded to +/-1 step per turn to prevent radical swings. Stagnation detection (2 unchanged turns) prevents infinite debate loops.

**Random Forest Classifier (scikit-learn):** Trained per-horizon on 25 features with `class_weight="balanced"` to handle label imbalance. RF was chosen for its strong performance on mixed-type tabular data (continuous indicators + binary sector flags), interpretable feature importances, and robustness without feature scaling.

**K-Nearest Neighbors Classifier (scikit-learn):** Trained per-horizon with distance-weighted voting and StandardScaler normalization. K is selected via grid search over {5, 7, 9, 11, 15, 21}. KNN was included as a complementary model to RF because it uses a fundamentally different decision mechanism (instance-based vs. ensemble tree-based), which increases diversity in the 3-way vote.

**3-Way Voting with Coverage Gates:** The synthesizer combines LLM debate, RF, and KNN predictions. If 2+ agree, the majority wins. If all disagree, scores are averaged. Coverage gates downgrade extreme recommendations when fewer than 2 analysts have confidence >= 0.30, and unresolved contradiction penalties can force a HOLD when severity is high.

### 1.3 Pipeline

```
User Input (ticker, horizon)
        |
        v
+------------------+
|   Data Fetcher   |  <-- yfinance (price, fundamentals)
|                  |  <-- Finnhub (news, earnings, insider)
|                  |  <-- FRED (GDP, VIX, unemployment, rates)
+------------------+
        |
        v
+------------------+       +-------------------+
|   Orchestrator   | <---> | Analyst Agents    |
|  (debate ctrl)   |       |  - Sentiment (LLM)|
|                  |       |  - Technical       |
|  Detects contra- |       |  - Fundamental     |
|  dictions, sends |       |  - Macro-Economic  |
|  debate turns    |       +-------------------+
+------------------+       +-------------------+
        |            <---> | Debate Nodes (LLM)|
        |                  |  - Tech Debate     |
        |                  |  - Fund Debate     |
        |                  |  - Macro Debate    |
        |                  +-------------------+
        v
+------------------+
| Classical Models |  <-- RF (from rf_{horizon}.pkl)
|                  |  <-- KNN (from knn_{horizon}.pkl)
+------------------+
        |
        v
+------------------+
|   Synthesizer    |  3-way vote: LLM + RF + KNN
|                  |  Coverage gates & penalties
|                  |  LLM narrative generation
+------------------+
        |
        v
  Final Recommendation
  (rating, rationale, debate transcript,
   model probabilities, reliability notes)
```

**Data flow detail:**
1. **Data Fetcher** collects raw market data from 3 APIs (error-isolated; partial data is acceptable)
2. **Orchestrator** dispatches each analyst in sequence (sentiment, technical, fundamental, macro)
3. Each **Analyst Agent** computes features from raw data and produces a rating + confidence
4. **Orchestrator** scans for contradictions (score distance >= 2 and severity >= 1.0)
5. If contradictions exist, **Debate Nodes** argue positions with bounded rating updates (max 3 rounds)
6. Once resolved or stagnated, **Classical Models Node** assembles the 25-feature vector from analyst outputs, imputes missing values, and runs RF/KNN inference
7. **Synthesizer** performs 3-way vote, applies coverage gates, generates LLM narrative, and outputs final report

### 1.4 Limitations & Mitigations

**Classification accuracy is modest (~38% exact, ~59% directional).** With 5 classes and noisy financial data, exact label prediction is inherently difficult. We mitigated this by focusing on directional accuracy (bullish/neutral/bearish), which is more actionable for investment decisions, and by using the 3-way vote to filter out low-confidence signals. In backtesting, all strategies outperform buy-and-hold despite the modest exact accuracy.

**LLM debate is not backtestable.** The debate system requires live Gemini API calls per ticker, so it cannot be included in offline backtesting. We mitigated this by backtesting the classical models independently (RF, KNN, 2-way vote) and validating that they provide value on their own, so the LLM debate is additive rather than load-bearing.

**Sentiment defaults to neutral in training data.** The dataset builder cannot call the LLM for every historical sample (cost + rate limits), so `sentiment_score` and `news_volume` are set to neutral defaults during training. This means the RF/KNN models cannot learn from sentiment features. We mitigated this by ensuring the LLM sentiment agent contributes through the debate/vote path rather than through the classical models.

**API rate limits and latency.** Finnhub's free tier has a 60-call/minute limit, and each analysis makes multiple API calls. Running batch analyses on many tickers can hit rate limits. We mitigated this by building a pre-cached dataset (`data/cached_dataset.csv`) for training, so live API calls are only needed for single-ticker analysis at runtime.

**Label imbalance.** HOLD is overrepresented in the dataset (~37%) while STRONG SELL/STRONG BUY are underrepresented (~7-14%). Without mitigation, models default to predicting HOLD. We addressed this with `class_weight="balanced"` in Random Forest, which inversely weights classes by frequency, and distance-weighted KNN, which reduces the bias toward majority-class neighbors.

**ETFs and tickers missing fundamentals.** Some tickers (e.g., SPY, QQQ) lack P/E ratios, debt-to-equity, or earnings data. The fundamental analyst would produce unreliable ratings with missing inputs. We mitigated this by returning None for unavailable metrics, using median imputation in the classical models, and having the confidence calculation penalize data-sparse analysts (confidence = data_coverage x agreement), which causes the orchestrator and synthesizer to weight their opinions less.

---

## 2. Feature Table

| Description | Platform | Completeness | Code | Author(s) | Notes |
|---|---|---|---|---|---|
| **Data Fetcher** - Collects price history, fundamentals, news, and macro data from 3 APIs (yfinance, Finnhub, FRED) | Local | 5 | Python | Ethan, Xavier | Error-isolated per source; handles ETFs missing fundamentals; fully tested |
| **Sentiment Analyst** - LLM-based news sentiment scoring (1-10) with Gemini; supports debate mode | Local | 5 | Python | JJ, Ethan | Handles missing news (defaults HOLD); JSON repair for malformed LLM responses; tested |
| **Technical Analyst** - Computes 7 technical indicators (RSI, MACD, SMA, Bollinger, volume, price changes) with horizon-weighted scoring | Local | 5 | Python | Ethan | Purely quantitative (no LLM); graceful degradation with insufficient price history; 16 tests |
| **Fundamental Analyst** - Analyzes P/E, debt/equity, margin, revenue growth, EPS with sector-specific thresholds for 6 sectors | Local | 5 | Python | Xavier | Multiple fallback sources per metric (yfinance -> Finnhub -> computed); tested |
| **Macro-Economic Analyst** - Scores VIX, yield curve, unemployment, fed funds, CPI with horizon-adjusted weights | Local | 5 | Python | Xavier | Computes composite sector_performance and binary market_trend; tested |
| **Debate System** - Orchestrator detects contradictions, assigns debate turns to outliers, tracks stagnation | Local | 5 | Python | Ethan | Bounded rating updates (+/-1 step); stagnation limit = 2 turns; coalition context for debaters; tested |
| **Technical Debate Node** - LLM-powered defense/revision of technical position during debate | Local | 4 | Python | JJ | Works well but LLM occasionally returns malformed JSON requiring repair prompt |
| **Fundamental Debate Node** - LLM-powered defense/revision of fundamental position during debate | Local | 4 | Python | JJ | Same JSON robustness caveat as technical debate |
| **Macro Debate Node** - LLM-powered defense/revision of macro position during debate | Local | 4 | Python | JJ | Same JSON robustness caveat as technical debate |
| **Feature Engineering** - 25-feature canonical pipeline (technical, sentiment, fundamental, macro, sector one-hot) shared by training and inference | Local | 5 | Python | Ethan | Single source of truth (FEATURE_ORDER); ensures model/inference consistency; tested |
| **Dataset Builder** - Generates training data from 128 S&P 500 tickers across 2 years with forward-return labels per horizon | Local | 5 | Python | Ethan | Cached as CSV (~7.9 MB, ~23K samples); 3 label horizons (short/swing/long) |
| **Random Forest Training** - Per-horizon RF with balanced class weights, temporal 80/20 split, feature importance output | Local | 5 | Python | Ethan | 100 trees, max_depth=10; median imputation; tested |
| **KNN Training** - Per-horizon KNN with distance weighting, k-selection grid search, StandardScaler | Local | 5 | Python | Ethan | Best k found via search over {5,7,9,11,15,21}; tested |
| **Model Evaluation** - Per-horizon accuracy, weighted F1, confusion matrix, head-to-head RF vs KNN comparison | Local | 5 | Python | Ethan | Cross-horizon summary table; per-class F1 breakdown; tested |
| **Backtesting** - Simulated trading returns using position sizing by prediction; directional accuracy and win rate metrics | Local | 5 | Python | Ethan | Compares RF, KNN, 2-way vote, and buy-and-hold baseline; tested |
| **3-Way Voting Synthesizer** - Combines LLM debate + RF + KNN; applies coverage gates and contradiction penalties; generates LLM narrative | Local | 5 | Python | Ethan | Majority vote or score-average fallback; downgrades extreme calls on low coverage; tested |
| **State Schema** - Custom LangGraph reducers: read-only data lock, non-destructive signal merge, debate state protection | Local | 5 | Python | Ethan | Prevents LLM hallucination overwrites of raw data; tested |
| **Horizon Configuration** - Short/swing/long definitions with per-horizon analyst weights and label thresholds | Local | 5 | Python | Ethan | Weights emphasize momentum for short-term, fundamentals for long-term |
| **Rating/Stance Utilities** - Rating enum, score conversion, alias parsing ("outperform" -> BUY), aggregate scoring | Local | 5 | Python | Ethan | Handles free-form LLM rating strings; tested |
| **Interactive CLI** - Ticker validation, horizon selection, result display loop | Local | 5 | Python | Ethan | Validates tickers via yfinance before analysis; clean exit handling |
| **Test Suite** - 19 test files covering all nodes, orchestrator, models, schema, and utilities | Local | 5 | Python | All | Mocked API calls; no live API dependencies in tests |

---

## 3. External Tools & Libraries

### 3.1 Frameworks & Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| langgraph | latest | State machine graph for multi-agent workflow orchestration |
| langchain-google-genai | latest | Integration layer for Google Gemini LLM (sentiment analysis, debate, narrative generation) |
| pandas | latest | DataFrame operations for price history, dataset building, feature computation |
| numpy | latest | Numerical operations in feature engineering and backtesting |
| scikit-learn | latest | Random Forest, KNN classifiers, SimpleImputer, StandardScaler, evaluation metrics |
| joblib | latest | Serialization of trained model artifacts (.pkl files) |
| yfinance | latest | Yahoo Finance API wrapper for price history, company info, financial statements |
| finnhub-python | latest | Finnhub API client for news articles, earnings surprises, insider sentiment, recommendation trends |
| fredapi | latest | Federal Reserve FRED API client for macro indicators (GDP, CPI, unemployment, VIX, yield curve) |
| python-dotenv | latest | Loading API keys from .env file |
| pytest | 9.0.2 | Test framework with unittest.mock for mocking API calls |

### 3.2 Datasets & APIs

| Source | Type | Usage | Link |
|--------|------|-------|------|
| Yahoo Finance (yfinance) | API | 1-year price history (OHLCV), company info, income statements, balance sheets, cash flow, analyst recommendations | https://pypi.org/project/yfinance/ |
| Finnhub | API (free tier) | Company news (30 days), basic financials, recommendation trends, earnings surprises (4 quarters), insider sentiment (MSPR) | https://finnhub.io/ |
| FRED (Federal Reserve) | API | GDP, CPI, unemployment rate, federal funds rate, 10-year Treasury yield, VIX; used for macro-economic analysis | https://fred.stlouisfed.org/ |
| Google Gemini 2.5 Flash | API | LLM for sentiment analysis, debate argumentation, and final narrative generation | https://ai.google.dev/ |
| Self-built dataset | CSV | `data/cached_dataset.csv` — 128 S&P 500 tickers, 2 years history, 25 features, 3-horizon labels; built with `data/build_dataset.py` | Generated locally |

### 3.3 Reused Code

All code in this project is original. No open-source code was adapted or copied. Standard library APIs (scikit-learn, pandas, LangGraph) were used according to their documentation. The project is licensed under MIT (Copyright 2026 Ethan Cratchley).

### 3.4 Installation & Dependencies

**Platform:** Python 3.11+ on macOS, Linux, or Windows

**Setup:**
```bash
# Clone and enter project
git clone <repository-url>
cd 310-project

# Create virtual environment
python3 -m venv venv              # macOS/Linux
# python -m venv venv             # Windows

# Activate
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env              # macOS/Linux
# copy .env.example .env          # Windows
# Edit .env with your API keys (GOOGLE_API_KEY, FINNHUB_API_KEY, FRED_API_KEY)
```

**Running:**
```bash
python main.py                    # Interactive analysis CLI
python models/train_models.py     # Train RF/KNN models (all horizons)
python models/evaluate.py         # Evaluate model performance
python models/backtest.py         # Run backtesting simulation
pytest tests/ -v                  # Run test suite
```

All dependencies are listed in `requirements.txt`. The pre-trained model files (`models/*.pkl`) and cached dataset (`data/cached_dataset.csv`) are included in the repository, so training is optional unless features are modified.
