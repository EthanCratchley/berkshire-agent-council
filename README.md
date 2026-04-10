# Berkshire Agent Council

Multi-agent stock analysis system built with LangGraph. Four AI analyst agents (sentiment, technical, fundamental, macro) debate contradictions, then a 3-way vote combines the LLM debate result with pre-trained Random Forest and KNN classifiers to produce a final recommendation. CMPT 310 course project.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
cp .env.example .env              # add your API keys (Google Gemini, Finnhub, FRED)
```

## Usage

```bash
python main.py                    # interactive stock analysis CLI
python models/train_models.py     # retrain RF/KNN models (optional — pre-trained .pkl files included)
python models/evaluate.py         # model evaluation metrics
python models/backtest.py         # backtesting simulation
pytest tests/ -v                  # run test suite (188 tests)
```

## Project Structure

```
├── main.py                        # LangGraph workflow entry point & CLI
├── requirements.txt
├── .env.example                   # API key template
│
├── nodes/
│   ├── data_fetcher.py            # Market data collection (yfinance, Finnhub, FRED)
│   ├── sentiment_node.py          # LLM-based news sentiment analysis
│   ├── technical_node.py          # Technical indicator scoring (RSI, MACD, SMA, etc.)
│   ├── fundamental_node.py        # Fundamental ratio analysis (P/E, D/E, margins)
│   ├── macro_econ_node.py         # Macro-economic indicator scoring
│   ├── technical_debate_node.py   # LLM debate agent for technical disputes
│   ├── fundamental_debate_node.py # LLM debate agent for fundamental disputes
│   ├── macro_debate_node.py       # LLM debate agent for macro disputes
│   ├── classical_models_node.py   # RF/KNN inference on live features
│   └── synthesizer_node.py        # 3-way vote, coverage gates, final narrative
│
├── orchestration/
│   └── orchestrator.py            # Contradiction detection, debate routing, stagnation control
│
├── shared/
│   ├── state_schema.py            # BerkshireState TypedDict + custom reducers
│   ├── feature_engineering.py     # 25-feature pipeline (FEATURE_ORDER)
│   ├── horizon.py                 # Short/swing/long config, analyst weights, thresholds
│   └── stance.py                  # Rating enum, score conversion, alias parsing
│
├── models/
│   ├── train_models.py            # RF/KNN training per horizon
│   ├── evaluate.py                # Accuracy, F1, confusion matrix, head-to-head
│   ├── backtest.py                # Simulated trading returns vs buy-and-hold
│   └── *.pkl                      # Pre-trained models, imputers, scalers
│
├── data/
│   ├── build_dataset.py           # Historical dataset builder (128 tickers, 2 years)
│   └── cached_dataset.csv         # Pre-built training data (~23K samples)
│
├── tests/                         # 19 test files, 188 tests, all API calls mocked
```

_Authors: Ethan Cratchley, Jayden Truong, Xavier Rahman_
