from langgraph.graph import StateGraph
from shared.state_schema import BerkshireState
from nodes.data_fetcher import data_fetcher
from nodes.technical_node import technical_node
from nodes.sentiment_node import sentiment_node
from nodes.fundamental_node import fundamental_node
from nodes.macro_econ_node import macro_econ_node
from orchestration.orchestrator import orchestrator
from langgraph.graph import StateGraph, START, END

# Init the graph with the BerkshireState schema (steady state)
workflow = StateGraph(BerkshireState)

workflow.add_node("data_fetcher", data_fetcher)
# workflow.add_node("technical_node", technical_node)
workflow.add_node("sentiment_node", sentiment_node)
# workflow.add_node("fundamental_node", fundamental_node)
# workflow.add_node("macro_econ_node", macro_econ_node)
workflow.add_node("orchestrator", orchestrator)

# Iteration 1 edges and simple CLI loop
workflow.add_edge(START, "data_fetcher")
workflow.add_edge("data_fetcher", "sentiment_node")
workflow.add_edge("sentiment_node", "orchestrator")
workflow.add_edge("orchestrator", END)

# Compile the graph here
app = workflow.compile()

if __name__ == "__main__":
    print("Welcome to the Berkshire Agent Council (Iteration 1)")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        # Get the ticker from the terminal
        user_input = input("Enter a stock ticker to analyze: ").strip().upper()
        
        # Escape hatch to exit the program
        if user_input in ['QUIT', 'EXIT']:
            print("Shutting down council...")
            break
            
        # Catch empty inputs if you accidentally hit Enter
        if not user_input:
            continue
            
        # Package the initial state baton (Analogy: the state is like passing the baton in a relay race)
        initial_state = {
            "ticker": user_input, # Pass the ticker the user chooses in the state
            "data": {},
            "analyst_signals": {}
        }
        
        print(f"\nDispatching agents for {user_input}...")
        
        # Hand the baton to LangGraph and start the run
        app.invoke(initial_state) # It will follow the flow we defined above
        
        print("\n" + "="*40 + "\n")

