from nodes.macro_debate_node import macro_debate_node


def test_macro_debate_node_preserves_quant_and_adds_narrative():
    state = {
        "ticker": "AAPL",
        "data": {},
        "analyst_signals": {
            "macro": {
                "rating": "buy",
                "confidence": 0.6,
                "features": {
                    "vix": 12.0,
                    "yield_curve_spread": 1.5,
                    "unemployment": 3.5,
                    "fed_funds_rate": 2.0,
                    "cpi_yoy": 2.0,
                },
                "details": "Baseline macro stance.",
            }
        },
        "debate": {
            "awaiting_response_from": "macro",
            "active_challenge": {
                "id": "macro_vs_sentiment",
                "action": "revise_or_defend",
                "reason": "Sentiment is more bullish.",
                "opponent_case": {"analyst": "sentiment", "rating": "strong_buy", "confidence": 0.8},
                "coalition": {"supporters_of_opponent": [], "partial": []},
            },
        },
        "final_report": {},
    }
    result = macro_debate_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] in ("strong_buy", "buy", "hold", "sell", "strong_sell")
    assert 0.0 <= sig["confidence"] <= 1.0
    assert isinstance(sig["narrative"], str)
    assert isinstance(sig["debate_response"], str)
    assert isinstance(sig["position_changed"], bool)
    assert isinstance(sig["counterpoints_addressed"], list)
    assert "Macroeconomic explanation" in sig["details"]
