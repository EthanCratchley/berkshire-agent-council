# Berkshire Agent Council

A multi-agent stock analysis system built with LangGraph.

## Project Structure

```
├── main.py                  # Entry point — builds and runs the analysis graph
├── nodes/                   # Agent logic
│   ├── data_fetcher.py      # Fetches market data (yfinance, Finnhub, FRED)
│   ├── technical_node.py    # Technical analysis agent
│   ├── fundamental_node.py  # Fundamental analysis agent
│   ├── sentiment_node.py    # Sentiment analysis agent
│   └── macro_econ_node.py   # Macro-economic analysis agent
├── orchestration/
│   └── orchestrator.py      # Graph flow and conditional routing
├── shared/
│   └── state_schema.py      # Shared state schemas and utilities
├── tests/                   # Test suite
│   ├── conftest.py          # Pytest config and shared fixtures
│   ├── test_data_fetcher.py # Data fetcher unit tests
│   └── test_schema.py       # State schema tests
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the analysis graph:
   ```bash
   python main.py
   ```

## Running Tests

```bash
# activate venv if not already active
source venv/bin/activate

# run full suite
pytest tests/ -v
```
