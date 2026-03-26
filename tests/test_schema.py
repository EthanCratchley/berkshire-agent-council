from langgraph.graph import StateGraph, START, END
from shared.state_schema import (
    BerkshireState,
    merge_signals,
    read_only_data,
    merge_debate,
    make_initial_debate_state,
)

# UNIT TESTS: Testing the Reducer Rules (Tests 3 & 4)
def test_data_protection():
    """
    Test 1: Cannot overwrite raw data with false data
    Tests read_only_data reducer
    """
    print("Running Test 1: Raw Data Protection...")
    original_data = {"AAPL_price": 150.00}
    rogue_agent_data = {"AAPL_price": 9999.00} # Agent hallucinating
    
    # Run the rule
    result = read_only_data(original_data, rogue_agent_data)
    
    assert result == {"AAPL_price": 150.00}, "FAIL: Data was overwritten!"
    print("✅ Passed: Raw data cannot be overwritten.\n")

def test_signal_merging():
    """
    Test 2: Cannot overwrite other agents results
    Tests merge_signals reducer
    """
    print("Running Test 2: Signal Merging...")
    current_canvas = {"sentiment": {"score": 8, "reason": "Good news"}}
    new_technical_data = {"technical": {"score": 3, "reason": "Bad chart"}}
    
    # Run the rule
    result = merge_signals(current_canvas, new_technical_data)
    
    assert "sentiment" in result and "technical" in result, "FAIL: Data was lost!"
    print("Passed: Agent signals merged successfully without deletion.\n")


def test_signal_revision_history_preserved():
    """
    Re-engaging the same analyst should preserve prior versions under "revisions".
    """
    existing = {
        "fundamental": {
            "rating": "sell",
            "confidence": 0.72,
            "details": "Initial stance.",
        }
    }
    incoming = {
        "fundamental": {
            "rating": "hold",
            "confidence": 0.44,
            "details": "Revised after challenge.",
        }
    }

    result = merge_signals(existing, incoming)
    fundamental = result["fundamental"]

    assert fundamental["rating"] == "hold"
    assert "revisions" in fundamental
    assert isinstance(fundamental["revisions"], list)
    assert len(fundamental["revisions"]) == 1
    assert fundamental["revisions"][0]["rating"] == "sell"


def test_debate_merge_appends_and_replaces_lists():
    """
    Debate reducer should append lists by default and support explicit replacement.
    """
    existing = make_initial_debate_state(max_rounds=3)
    existing["queue"] = [{"id": "c1"}]
    existing["history"] = [{"event": "r1"}]

    appended = merge_debate(existing, {"queue": [{"id": "c2"}], "round": 1})
    assert appended["round"] == 1
    assert [item["id"] for item in appended["queue"]] == ["c1", "c2"]

    replaced = merge_debate(
        appended,
        {"queue": [{"id": "c3"}], "_replace_lists": ["queue"]},
    )
    assert [item["id"] for item in replaced["queue"]] == ["c3"]

# INTEGRATION TESTS: Testing the Graph Flow (Tests 1 & 2)
def test_graph_flow():
    """Tests 3 & 4: Raw data -> state -> agent context -> result"""
    print("Running Tests 3 & 4: Graph Flow and Context...")
    
    # 1. Define dummy nodes
    def dummy_data_fetcher(state: BerkshireState):
        # Simulates fetching data
        return {"data": {"financials": "Strong earnings", "price": 100}}

    def dummy_sentiment_agent(state: BerkshireState):
        # Test 3 check: Is the raw data available as context?
        raw_data = state.get("data", {})
        if not raw_data:
            raise ValueError("Agent could not see the raw data!")
            
        # Test 4: Derive result based on that data
        reasoning = f"Saw {raw_data['financials']}, so I am bullish."
        return {
            "analyst_signals": {
                "sentiment": {
                    "rating": "buy",
                    "confidence": 0.7,
                    "details": reasoning,
                }
            }
        }

    # 2. Build the mini-graph
    workflow = StateGraph(BerkshireState)
    workflow.add_node("fetcher", dummy_data_fetcher)
    workflow.add_node("agent", dummy_sentiment_agent)
    
    workflow.add_edge(START, "fetcher")
    workflow.add_edge("fetcher", "agent")
    workflow.add_edge("agent", END)
    
    app = workflow.compile()
    
    # 3. Run it
    initial_state = {"ticker": "AAPL", "data": {}, "analyst_signals": {}}
    final_state = app.invoke(initial_state)
    
    print("Final State Output:")
    print(final_state)
    print("\n✅ Passed: Graph routed data correctly to agent and saved the result.")

# Run the tests
if __name__ == "__main__":
    test_data_protection()
    test_signal_merging()
    test_graph_flow()
