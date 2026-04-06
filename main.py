from langgraph.graph import StateGraph, START, END
import logging

import yfinance as yf

from shared.state_schema import BerkshireState, make_initial_debate_state
from shared.horizon import normalize_horizon, horizon_label, horizon_day_range
from nodes.data_fetcher import data_fetcher
from nodes.technical_node import technical_node
from nodes.sentiment_node import sentiment_node
from nodes.fundamental_node import fundamental_node
from nodes.fundamental_debate_node import fundamental_debate_node
from nodes.macro_econ_node import macro_econ_node
from nodes.macro_debate_node import macro_debate_node
from nodes.technical_debate_node import technical_debate_node
from nodes.synthesizer_node import synthesizer_node
from nodes.classical_models_node import classical_models_node
from orchestration.orchestrator import orchestrator


# Must yfinance logging to CRITICAL to avoid terminal clutter for expected warning cases.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def is_valid_ticker(ticker_symbol: str) -> bool:
    """Fast validation check for market ticker existence."""
    try:
        stock = yf.Ticker(ticker_symbol)
        return not stock.history(period="1d").empty
    except Exception:
        return False


def route_from_orchestrator(state: BerkshireState):
    next_node = (state.get("debate", {}) or {}).get("next_node")
    if next_node in {
        "sentiment_node",
        "fundamental_node",
        "fundamental_debate_node",
        "technical_node",
        "technical_debate_node",
        "macro_econ_node",
        "macro_debate_node",
    }:
        return next_node
    # When debate is done, run classical models before synthesis.
    return "classical_models_node"


def prompt_for_horizon() -> str:
    print("Select analysis horizon:")
    print("  short - Short-term (1-10 trading days)")
    print("  swing - Swing (2-8 weeks) [default]")
    print("  long  - Long-term (6-24 months)")
    raw = input("Enter horizon [short/swing/long]: ").strip().lower()
    return normalize_horizon(raw if raw else "swing")


# Build graph.
workflow = StateGraph(BerkshireState)
workflow.add_node("data_fetcher", data_fetcher)
workflow.add_node("sentiment_node", sentiment_node)
workflow.add_node("fundamental_node", fundamental_node)
workflow.add_node("fundamental_debate_node", fundamental_debate_node)
workflow.add_node("technical_node", technical_node)
workflow.add_node("technical_debate_node", technical_debate_node)
workflow.add_node("macro_econ_node", macro_econ_node)
workflow.add_node("macro_debate_node", macro_debate_node)
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("classical_models_node", classical_models_node)
workflow.add_node("synthesizer_node", synthesizer_node)

# Data loading first, then orchestrator controls the full flow.
workflow.add_edge(START, "data_fetcher")
workflow.add_edge("data_fetcher", "orchestrator")

# Any analyst turn routes back to orchestrator.
workflow.add_edge("sentiment_node", "orchestrator")
workflow.add_edge("fundamental_node", "orchestrator")
workflow.add_edge("fundamental_debate_node", "orchestrator")
workflow.add_edge("technical_node", "orchestrator")
workflow.add_edge("technical_debate_node", "orchestrator")
workflow.add_edge("macro_econ_node", "orchestrator")
workflow.add_edge("macro_debate_node", "orchestrator")

# Classical models run, then synthesis exits.
workflow.add_edge("classical_models_node", "synthesizer_node")
workflow.add_edge("synthesizer_node", END)

workflow.add_conditional_edges(
    "orchestrator",
    route_from_orchestrator,
    {
        "sentiment_node": "sentiment_node",
        "fundamental_node": "fundamental_node",
        "fundamental_debate_node": "fundamental_debate_node",
        "technical_node": "technical_node",
        "technical_debate_node": "technical_debate_node",
        "macro_econ_node": "macro_econ_node",
        "macro_debate_node": "macro_debate_node",
        "classical_models_node": "classical_models_node",
    },
)

app = workflow.compile()


if __name__ == "__main__":
    print("Welcome to the Berkshire Agent Council")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("Enter a stock ticker to analyze: ").strip().upper()

        if user_input in ["QUIT", "EXIT"]:
            print("Shutting down council...")
            break

        if not user_input:
            continue

        print(f"Validating ticker '{user_input}'...")
        if not is_valid_ticker(user_input):
            print(f"{user_input}' is not a valid stock ticker. Please try again.\n")
            continue
        selected_horizon = prompt_for_horizon()
        horizon_min_days, horizon_max_days = horizon_day_range(selected_horizon)

        initial_state = {
            "ticker": user_input,
            "horizon": selected_horizon,
            "horizon_days": {
                "min": horizon_min_days,
                "max": horizon_max_days,
            },
            "data": {},
            "analyst_signals": {},
            "debate": make_initial_debate_state(max_rounds=3),
            "final_report": {},
        }

        print(
            f"\nDispatching agents for {user_input} "
            f"({horizon_label(selected_horizon)})..."
        )
        final_state = app.invoke(initial_state)

        final_report = final_state.get("final_report", {})
        if final_report:
            print(f"Horizon: {final_report.get('horizon_label', horizon_label(selected_horizon))}")
            vote = final_report.get("vote", {})
            print(f"Final Recommendation: {final_report.get('recommendation', 'N/A').upper()}"
                  f"  ({vote.get('method', '?')}, {vote.get('agreement', '?')})")
            print(f"  LLM Debate:    {final_report.get('llm_recommendation', 'N/A')}")
            cm = final_report.get("classical_models", {})
            if cm.get("rf"):
                print(f"  Random Forest: {cm['rf'].get('prediction', 'N/A')}")
            if cm.get("knn"):
                print(f"  KNN:           {cm['knn'].get('prediction', 'N/A')}")
            print(f"Rationale: {final_report.get('rationale', 'No rationale generated.')}")
            detailed_narrative = final_report.get("detailed_narrative", "")
            if detailed_narrative:
                print(f"Market Summary: {detailed_narrative}")

        print("\n" + "=" * 40 + "\n")
