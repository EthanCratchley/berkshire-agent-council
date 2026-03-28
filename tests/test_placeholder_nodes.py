from nodes.macro_econ_node import macro_econ_node
from nodes.technical_node import technical_node


def test_technical_placeholder_returns_stance_contract():
    result = technical_node({"ticker": "AAPL", "data": {}, "analyst_signals": {}})
    sig = result["analyst_signals"]["technical"]

    assert sig["rating"] == "hold"
    assert sig["confidence"] == 0.0
    assert isinstance(sig["features"], dict)


def test_macro_placeholder_returns_stance_contract():
    result = macro_econ_node({"ticker": "AAPL", "data": {}, "analyst_signals": {}})
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] == "hold"
    assert sig["confidence"] == 0.0
    assert isinstance(sig["features"], dict)
