from nodes.fundamental_debate_node import fundamental_debate_node


def test_fundamental_debate_node_preserves_quant_and_adds_narrative():
    state = {
        "ticker": "AAPL",
        "data": {},
        "analyst_signals": {
            "fundamental": {
                "rating": "buy",
                "confidence": 0.6,
                "features": {
                    "pe_ratio": 20.0,
                    "debt_to_equity": 40.0,
                    "profit_margin": 0.18,
                    "revenue_growth": 0.08,
                    "eps": 4.5,
                },
                "details": "Baseline fundamental stance.",
            }
        },
        "debate": {
            "awaiting_response_from": "fundamental",
            "active_challenge": {
                "id": "fundamental_vs_sentiment",
                "action": "revise_or_defend",
                "reason": "Sentiment is more bullish.",
                "primary_opponent": "sentiment",
                "coalition": {"supporters_of_opponent": [], "partial": []},
            },
        },
        "final_report": {},
    }
    result = fundamental_debate_node(state)
    sig = result["analyst_signals"]["fundamental"]

    assert sig["rating"] in ("strong_buy", "buy", "hold", "sell", "strong_sell")
    assert sig["confidence"] == 0.6
    assert isinstance(sig["narrative"], str)
    assert "Fundamental explanation" in sig["details"]

