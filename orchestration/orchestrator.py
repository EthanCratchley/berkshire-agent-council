from shared.state_schema import BerkshireState

def should_continue(state: BerkshireState):
    """
    Conditional routing logic to decide the next step in the graph.
    """
    # For now just implementing a simple output formatter for this node just for ITERATION 1
    print("\n---ORCHESTRATOR REVIEW ---")
        
    # Safely get the signals dictionary
    signals = state.get("analyst_signals", {})
        
    # For Iteration 1, we just print exactly what the Sentiment Agent wrote
    if "sentiment" in signals:
        sentiment_data = signals["sentiment"]
        print(f"Sentiment Agent Output: {sentiment_data}")
    else:
        print("Warning: No sentiment data found on the canvas.")
            
    print("------------------------------\n")
        
    # Returning an empty dict means "I am not updating the state, just passing it along"
    return {}
