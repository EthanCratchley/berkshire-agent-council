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
        "debate": {"unresolved_contradictions": []},
        "final_report": {},
    }
    result = synthesizer_node(state)
    report = result["final_report"]
    assert report["ticker"] == "AAPL"
    assert report["recommendation"] in ("strong_buy", "buy", "hold", "sell", "strong_sell")
    assert "weighted_score" in report
    assert isinstance(report["analyst_breakdown"], list)
    assert "rationale" in report

