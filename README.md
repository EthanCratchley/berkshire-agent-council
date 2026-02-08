# Berkshire Agent Council

A multi-agent stock analysis system built with LangGraph.

## Project Structure

- `nodes/`: Contains the logic for different analysis agents (Technical, Sentiment, Fundamental, Macro).
- `orchestration/`: Logic for graph flow and conditional routing.
- `shared/`: Shared state schemas and utilities.
- `main.py`: Entry point to build and run the analysis graph.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the skeleton:
   ```bash
   python main.py
   ```
