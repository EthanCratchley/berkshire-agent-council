from langgraph.graph import StateGraph
from shared.state_schema import BerkshireState
from nodes.data_fetcher import data_fetcher
from nodes.technical_node import technical_node
from nodes.sentiment_node import sentiment_node
from nodes.fundamental_node import fundamental_node
from nodes.macro_econ_node import macro_econ_node
from orchestration.orchestrator import orchestrator

workflow = StateGraph(BerkshireState)

workflow.add_node("data_fetcher", data_fetcher)
workflow.add_node("technical_node", technical_node)
workflow.add_node("sentiment_node", sentiment_node)
workflow.add_node("fundamental_node", fundamental_node)
workflow.add_node("macro_econ_node", macro_econ_node)
workflow.add_node("orchestrator", orchestrator)

# TODO: Team to define edges and entry point
