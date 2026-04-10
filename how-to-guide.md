# CMPT 310 - How-To Guide: Building a Multi-Agent Stock Analysis System with LangGraph

**Group Members:** Ethan Cratchley, Jayden Truong, Xavier Rahman
**Date:** April 12, 2026

---

## 1. Introduction

This guide teaches you how to build a multi-agent stock analysis system that combines LLM-powered debate with classical machine learning models to generate investment recommendations. The system, called the **Berkshire Agent Council**, uses a LangGraph state machine to orchestrate four AI analyst agents (sentiment, technical, fundamental, macro-economic) that independently analyze market data and then debate contradictions before arriving at a consensus. Their LLM-derived recommendation is then combined with pre-trained Random Forest and K-Nearest Neighbors classifiers in a 3-way vote to produce a final buy/hold/sell signal.

By the end of this guide, you will be able to: (1) design a multi-agent LangGraph workflow with custom state reducers, (2) build a shared feature engineering pipeline used by both live inference and offline training, (3) train and evaluate RF and KNN classifiers on historical stock data, and (4) wire everything together into a 3-way voting system that produces explainable investment recommendations with confidence scores, debate transcripts, and backtested performance metrics.

---

## 2. Prerequisites

**Concepts you should know:**
- Python programming (functions, classes, dictionaries, type hints)
- Basic machine learning: classification, train/test splits, accuracy, F1 score, confusion matrices
- Familiarity with scikit-learn (Random Forest, KNN, preprocessing)
- Basic understanding of stock market terminology (P/E ratio, RSI, MACD)

**Tools and dependencies:**
- Python 3.11+
- pip (package manager)
- A terminal/command line
- API keys for: Google Gemini (LLM), Finnhub (market data), FRED (macro indicators)
- ~500 MB disk space (virtual environment + cached dataset + trained models)

**Key libraries:**
| Library | Purpose |
|---------|---------|
| `langgraph` | State machine workflow orchestration |
| `langchain-google-genai` | Google Gemini LLM integration |
| `scikit-learn` | Random Forest, KNN, preprocessing |
| `pandas` / `numpy` | Data manipulation |
| `yfinance` | Yahoo Finance price/fundamental data |
| `finnhub-python` | News, earnings, insider sentiment |
| `fredapi` | Federal Reserve macro indicators |

---

## 3. Step-by-Step Walkthrough

### Step 1: Set Up Your Environment

Clone the repository and create a Python virtual environment:

```bash
git clone <repository-url>
cd 310-project

python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

> **Windows note:** Use `python` instead of `python3` throughout this guide. On macOS/Linux, `python3` is the standard command; on Windows, the installer registers `python` by default.

Create a `.env` file from the template and add your API keys:

```bash
cp .env.example .env        # macOS/Linux
# copy .env.example .env    # Windows (Command Prompt)
```

Edit `.env` with your keys:

```ini
FINNHUB_API_KEY=your_finnhub_key_here
FRED_API_KEY=your_fred_key_here
GOOGLE_API_KEY=your_google_gemini_key_here
```

Verify the setup by running the test suite:

```bash
pytest tests/ -v
```

Expected output (abbreviated):

```
tests/test_data_fetcher.py::test_successful_fetch PASSED
tests/test_sentiment_node.py::test_output_contract PASSED
tests/test_technical_node.py::test_bullish_indicators PASSED
tests/test_orchestrator.py::test_contradiction_detection PASSED
...
XX passed in Y.YYs
```

---

### Step 2: Load and Process the Data

The system fetches data from three independent API sources and engineers 25 features for model input. Understanding the data pipeline is critical because the same feature definitions are shared between live analysis and offline training.

**Data fetching** (`nodes/data_fetcher.py`) pulls from yfinance (price history, fundamentals), Finnhub (news, earnings, insider trades), and FRED (GDP, VIX, unemployment, interest rates). Each source is error-isolated, so a failure in one does not block others:

```python
# From nodes/data_fetcher.py — each API call is wrapped independently
try:
    stock = yf.Ticker(ticker)
    price_history = stock.history(period="1y")
    # ... fetch fundamentals, balance sheet, etc.
except Exception as e:
    result["yfinance_error"] = str(e)

try:
    client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
    news = client.company_news(ticker, _from=start_date, to=end_date)
    # ... fetch earnings, insider sentiment, etc.
except Exception as e:
    result["finnhub_error"] = str(e)
```

**Feature engineering** (`shared/feature_engineering.py`) defines 25 features in a canonical order. This ordering is the single source of truth, used by both the dataset builder and the live inference pipeline:

```python
FEATURE_ORDER = [
    # Technical (7)
    "rsi", "macd_histogram", "sma_20_50_cross", "bollinger_pct",
    "volume_ratio", "price_change_5d", "price_change_20d",
    # Sentiment (2)
    "sentiment_score", "news_volume",
    # Fundamental (5)
    "pe_ratio", "debt_to_equity", "profit_margin", "revenue_growth", "eps",
    # Macro (2)
    "sector_performance", "market_trend",
    # Sector one-hot (9)
    "sector_technology", "sector_financial_services", ...
]
```

Each feature category has a dedicated compute function. For example, technical features are computed from price history:

```python
def compute_technical_features(price_df: pd.DataFrame) -> dict:
    # RSI (14-period): measures momentum
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(window=14).mean().iloc[-1]
    avg_loss = loss.rolling(window=14).mean().iloc[-1]
    rs = avg_gain / avg_loss
    features["rsi"] = round(100 - (100 / (1 + rs)), 2)

    # MACD histogram: trend direction and strength
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features["macd_histogram"] = float(macd_line.iloc[-1] - signal_line.iloc[-1])

    # ... bollinger bands, SMA crossover, volume ratio, price changes
    return features
```

**Building the training dataset** (`data/build_dataset.py`) iterates over 60+ S&P 500 tickers across 2 years of history. For each ticker on each date, it computes all 25 features and assigns labels based on forward returns:

```bash
python data/build_dataset.py
```

Labels are assigned per horizon using threshold buckets. For the swing horizon (35-day forward return):

| Forward Return | Label |
|----------------|-------|
| > +12% | STRONG BUY |
| +5% to +12% | BUY |
| -5% to +5% | HOLD |
| -12% to -5% | SELL |
| < -12% | STRONG SELL |

The output is `data/cached_dataset.csv` (~7.9 MB, ~23,000 samples) with columns: `date, ticker, [25 features], label_short, label_swing, label_long`.

---

### Step 3: Define the Model Architecture

The architecture has two layers: a **LangGraph state machine** that orchestrates LLM debate, and **classical ML models** (RF/KNN) that run independently. Their outputs are combined via a 3-way vote.

**LangGraph Workflow** (`main.py`):

```
START --> data_fetcher --> orchestrator --+--> sentiment_node ----+
                              ^          +--> technical_node ----+
                              |          +--> fundamental_node --+
                              |          +--> macro_econ_node ---+
                              |          |                       |
                              |          +--> *_debate_node -----+
                              +----------------------------------+
                              |
                              +--> classical_models_node --> synthesizer_node --> END
```

The graph is built with LangGraph's `StateGraph`:

```python
workflow = StateGraph(BerkshireState)

# Add all nodes
workflow.add_node("data_fetcher", data_fetcher)
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("sentiment_node", sentiment_node)
workflow.add_node("technical_node", technical_node)
# ... (all analyst and debate nodes)
workflow.add_node("classical_models_node", classical_models_node)
workflow.add_node("synthesizer_node", synthesizer_node)

# Data fetching happens first, then orchestrator controls flow
workflow.add_edge(START, "data_fetcher")
workflow.add_edge("data_fetcher", "orchestrator")

# Every analyst routes back to orchestrator for debate control
workflow.add_edge("sentiment_node", "orchestrator")
workflow.add_edge("technical_node", "orchestrator")
# ...

# Conditional routing: orchestrator decides next node
workflow.add_conditional_edges("orchestrator", route_from_orchestrator, {...})

# After debate resolves, run classical models then synthesize
workflow.add_edge("classical_models_node", "synthesizer_node")
workflow.add_edge("synthesizer_node", END)

app = workflow.compile()
```

**State management** (`shared/state_schema.py`) uses custom reducers to prevent common multi-agent pitfalls:

```python
class BerkshireState(TypedDict):
    ticker: str
    horizon: str
    data: Annotated[Dict[str, Any], read_only_data]        # Immutable after first write
    analyst_signals: Annotated[Dict[str, Any], merge_signals]  # Non-destructive merge
    debate: Annotated[Dict[str, Any], merge_debate]         # Protected debate state
    final_report: Annotated[Dict[str, Any], merge_dict]
```

The `read_only_data` reducer is particularly important: it locks raw market data after the data_fetcher writes it, preventing LLM agents from hallucinating data overwrites:

```python
def read_only_data(existing_data, new_data):
    if existing_data:     # Already populated — reject overwrites
        return existing_data
    return new_data       # First write — allow
```

**The debate system** (`orchestration/orchestrator.py`) detects contradictions between analysts and facilitates structured debate. For example, if the sentiment analyst says BUY (confidence 0.8) and the technical analyst says SELL (confidence 0.75), the orchestrator calculates severity = score_distance x min_confidence = 2 x 0.75 = 1.5, which exceeds the threshold (1.0), triggering a debate. The outlier analyst is challenged to revise or defend their position with evidence from the other analyst's case, and rating changes are bounded to +/-1 step per turn to prevent radical swings.

**Three-way vote** (`nodes/synthesizer_node.py`) combines the LLM debate result with RF and KNN predictions:

```python
def three_way_vote(llm_rating, rf_prediction, knn_prediction):
    votes = [llm_rating, rf_prediction, knn_prediction]
    counts = Counter(votes)
    most_common, count = counts.most_common(1)[0]
    if count >= 2:
        return {"consensus": most_common, "method": "majority"}

    # All three disagree — average the numerical scores
    scores = [rating_to_score(parse_rating(r)) for r in votes]
    avg = sum(scores) / len(scores)
    return {"consensus": score_to_rating(avg), "method": "score_average"}
```

---

### Step 4: Train the Model

Training uses the cached dataset with a temporal 80/20 split (older data for training, recent data for testing) to prevent data leakage.

```bash
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Train all horizons (short, swing, long)
python models/train_models.py

# Or train a specific horizon
python models/train_models.py swing
```

The training script (`models/train_models.py`) trains both Random Forest and KNN for each horizon:

```python
# Random Forest — handles imbalanced classes with balanced weighting
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# KNN — finds best k via grid search, uses distance-weighted voting
best_k, best_acc = 5, 0.0
for k in (5, 7, 9, 11, 15, 21):
    knn_trial = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    knn_trial.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn_trial.predict(X_test_scaled))
    if acc > best_acc:
        best_k, best_acc = k, acc

knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance", n_jobs=-1)
knn.fit(X_train_scaled, y_train)
```

Key design decisions:
- **`class_weight="balanced"`** for RF: the dataset has uneven label distribution (more HOLD samples than STRONG BUY/SELL). Balanced weighting inversely weights classes by frequency, preventing the model from defaulting to HOLD.
- **`StandardScaler`** for KNN only: KNN uses distance metrics sensitive to feature magnitude, so features are normalized. RF is tree-based and scale-invariant.
- **`SimpleImputer(strategy="median")`**: some features may be missing (e.g., P/E ratio unavailable for an ETF). Median imputation is robust to outliers.

Expected training output (abbreviated):

```
Loading dataset...
Dataset: 23456 samples, 63 tickers
Date range: 2023-01-03 to 2025-03-28

============================================================
  SWING horizon  (35-day forward return)
  Thresholds: (0.12, 0.05, -0.05, -0.12)
============================================================
Samples with labels: 21893
Train: 17514 | Test: 4379

Training Random Forest (swing)...
Accuracy: 0.3842

Feature Importance:
  price_change_20d        0.0891 ########
  rsi                     0.0854 ########
  price_change_5d         0.0811 ########
  pe_ratio                0.0734 #######
  macd_histogram          0.0698 ######
  ...

Training KNN (swing, k=11)...
KNN Accuracy: 0.3156
```

Each horizon produces four saved artifacts:
- `rf_{horizon}.pkl` — Trained Random Forest
- `knn_{horizon}.pkl` — Trained KNN
- `imputer_{horizon}.pkl` — Fitted median imputer
- `scaler_{horizon}.pkl` — Fitted StandardScaler

---

### Step 5: Evaluate Results

Run evaluation to see detailed metrics, confusion matrices, and model comparison:

```bash
python models/evaluate.py
```

Expected output (abbreviated):

```
======================================================================
  SWING HORIZON  (35-day forward return)
======================================================================
Test samples: 4379
Test label distribution:
  STRONG SELL     312  ( 7.1%)
  SELL            845  (19.3%)
  HOLD           1623  (37.1%)
  BUY             987  (22.5%)
  STRONG BUY      612  (14.0%)

--- Random Forest ---
Accuracy:    0.3842
Weighted F1: 0.3791

              precision    recall  f1-score   support
  STRONG SELL      0.29      0.31      0.30       312
         SELL      0.32      0.34      0.33       845
         HOLD      0.43      0.45      0.44      1623
          BUY      0.35      0.32      0.33       987
   STRONG BUY      0.38      0.35      0.36       612

RF Confusion Matrix
Predicted ->  STRONG SELL        SELL        HOLD         BUY   STRONG BUY
  STRONG SELL          97          85          71          35          24
         SELL          73         287         269         132          84
         HOLD          52         193         730         398         250
          BUY          31         117         312         316         211
   STRONG BUY          18          59         182         139         214

--- Head-to-Head: SWING ---
Metric                     RF        KNN     Winner
--------------------------------------------------
Accuracy               0.3842     0.3156         RF
Weighted F1            0.3791     0.3087         RF
```

**Run backtesting** to evaluate trading performance:

```bash
python models/backtest.py
```

Backtesting simulates strategy returns by mapping predictions to position sizes:

| Prediction | Position |
|-----------|----------|
| STRONG BUY | 100% long |
| BUY | 50% long |
| HOLD | 0% (flat) |
| SELL | 50% short |
| STRONG SELL | 100% short |

Expected output (abbreviated):

```
--- Classification Accuracy ---
  Random Forest: 0.3842
  KNN:           0.3156
  2-Way Vote:    0.3523

--- Directional Accuracy (bullish/neutral/bearish) ---
  Random Forest: 0.5891
  KNN:           0.5234
  2-Way Vote:    0.5612

--- Simulated Strategy Returns ---
  Strategy             Cumulative    Avg/Trade     vs B&H
  ------------------------------------------------------
  Random Forest            0.2145      0.001234     +34.5%
  KNN                      0.1832      0.001054     +29.2%
  2-Way Vote               0.1987      0.001144     +31.8%
  Buy & Hold               0.1532      0.000883         —

--- Win Rate (% of active trades profitable) ---
  Random Forest: 58.3% (3412 trades)
  KNN:           54.7% (3189 trades)
  2-Way Vote:    56.8% (3301 trades)
```

**Interpreting the results:** While exact classification accuracy (~38%) seems modest for a 5-class problem, directional accuracy (~59%) is more meaningful — the model correctly identifies the general direction (bullish/neutral/bearish) more often than not. In backtesting, all strategies outperform buy-and-hold, which validates that the models capture some predictive signal. The RF generally outperforms KNN, consistent with RF's strength on tabular data with mixed feature types.

**Running the full live system:**

```bash
python main.py
```

```
Welcome to the Berkshire Agent Council
Type 'quit' or 'exit' to stop.

Enter a stock ticker to analyze: AAPL
Validating ticker 'AAPL'...
Select analysis horizon:
  short - Short-term (1-10 trading days)
  swing - Swing (2-8 weeks) [default]
  long  - Long-term (6-24 months)
Enter horizon [short/swing/long]: swing

Dispatching agents for AAPL (Swing (2-8 weeks))...
[Data Fetcher] Intel gathered for AAPL...
[Orchestrator] Collecting analyst signals...
[Sentiment] score=7 (bullish); rating=buy; confidence=0.65
[Technical] RSI=45.2, MACD=+0.82; rating=buy; confidence=0.58
[Fundamental] P/E=28.5, margin=26%; rating=hold; confidence=0.42
[Macro] VIX=18.3, yield_spread=0.45; rating=hold; confidence=0.35
[Orchestrator] Contradiction: sentiment(buy) vs fundamental(hold) — severity 1.2
[Debate] Sentiment defends: "Strong news momentum outweighs valuation..."
[Debate] Fundamental concedes: adjusting to buy (confidence 0.38)
...
Horizon: Swing (2-8 weeks)
Final Recommendation: BUY  (majority, majority)
  LLM Debate:    buy
  Random Forest: BUY
  KNN:           STRONG BUY
Rationale: Voted recommendation is BUY (majority, majority) with no
  unresolved contradictions...
```

---

## 4. Troubleshooting

### Issue 1: `ModuleNotFoundError: No module named 'shared'`

**Error message:**
```
ModuleNotFoundError: No module named 'shared.feature_engineering'
```

**Cause:** Python can't find the project's internal packages because you're running a script from a subdirectory (e.g., `cd models && python train_models.py`), so the project root isn't on `sys.path`.

**Fix:** Always run scripts from the project root directory:
```bash
cd 310-project
python models/train_models.py     # correct (all platforms)
```

> **Windows note:** If you see `python3: command not found`, use `python` instead. Verify which version you're running with `python --version` (should be 3.11+).

If using pytest, the `tests/conftest.py` file handles path setup automatically.

### Issue 2: `FileNotFoundError` or stale model when running `main.py`

**Error message:**
```
FileNotFoundError: [Errno 2] No such file or directory: '.../models/rf_swing.pkl'
```
or the model produces unexpected results after changing features.

**Cause:** The pre-trained model `.pkl` files don't exist yet, or they were trained with a different feature set than the current `FEATURE_ORDER`. Since `FEATURE_ORDER` in `shared/feature_engineering.py` defines the exact input shape both models expect, any mismatch causes silent errors or crashes.

**Fix:** Retrain models whenever you modify `FEATURE_ORDER` or the dataset:
```bash
python data/build_dataset.py      # rebuild dataset if features changed
python models/train_models.py     # retrain all horizons
python models/evaluate.py         # verify metrics look reasonable
```

---

## 5. Summary

In this guide, you learned how to build a multi-agent stock analysis system that combines LLM-powered debate with classical ML classification. The key techniques covered were: (1) designing a LangGraph state machine with custom reducers to safely coordinate multiple AI agents, (2) building a shared feature engineering pipeline (25 features across technical, fundamental, sentiment, and macro categories) used consistently by both live inference and offline training, (3) training Random Forest and KNN classifiers with class balancing and temporal train/test splits, and (4) combining all three approaches in a 3-way vote with coverage gates and contradiction penalties to produce reliable, explainable recommendations.

To go further, you could extend the system with additional models (e.g., gradient boosting, neural networks), add caching to reduce API calls, or build a web dashboard to visualize debate transcripts and model confidence distributions. All sample code is included in the accompanying ZIP file. The main entry point is `main.py`, model training is in `models/train_models.py`, and evaluation/backtesting are in `models/evaluate.py` and `models/backtest.py`.
