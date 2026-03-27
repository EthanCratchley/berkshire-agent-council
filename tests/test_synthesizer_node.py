from nodes.synthesizer_node import synthesizer_node


def test_synthesizer_returns_final_report_shape():
    state = {
        "ticker": "AAPL",
        "data": {},
        "analyst_signals": {
            "sentiment": {"rating": "buy", "confidence": 0.8, "details": ""},
            "fundamental": {"rating": "strong_buy", "confidence": 0.7, "details": ""},
            "technical": {"rating": "hold", "confidence": 0.5, "details": ""},
            "macro": {"rating": "sell", "confidence": 0.4, "details": ""},
        },
        "debate": {
            "unresolved_contradictions": [],
            "history": [
                {
                    "event": "debater_turn_result",
                    "speaker": "sentiment",
                    "rating": "sell",
                    "confidence": 0.56,
                    "debate_response": "I still weigh near-term headwinds more heavily.",
                    "position_changed": False,
                    "counterpoints_addressed": ["custom AI chips threat"],
                    "weighting_statement": "Near-term legal and demand shocks dominate long-term upside.",
                }
            ],
        },
        "final_report": {},
    }
    result = synthesizer_node(state)
    report = result["final_report"]
    assert report["ticker"] == "AAPL"
    assert report["recommendation"] in ("strong_buy", "buy", "hold", "sell", "strong_sell")
    assert "weighted_score" in report
    assert isinstance(report["analyst_breakdown"], list)
    assert isinstance(report["debate_transcript"], list)
    assert "weighting_statement" in report["debate_transcript"][0]
    assert "rationale" in report
