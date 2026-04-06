from nodes.synthesizer_node import synthesizer_node, three_way_vote


def test_synthesizer_returns_final_report_shape():
    state = {
        "ticker": "AAPL",
        "horizon": "long",
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
    assert report["horizon"] == "long"
    assert report["recommendation"] in ("strong_buy", "buy", "hold", "sell", "strong_sell")
    assert "vote" in report
    assert "llm_recommendation" in report
    assert "weighted_score" in report
    assert isinstance(report["analyst_breakdown"], list)
    assert isinstance(report["debate_transcript"], list)
    assert "weighting_statement" in report["debate_transcript"][0]
    assert "rationale" in report


class TestThreeWayVote:
    def test_unanimous(self):
        result = three_way_vote("buy", "BUY", "BUY")
        assert result["consensus"] == "buy"
        assert result["method"] == "majority"
        assert result["agreement"] == "unanimous"

    def test_majority_rf_knn_agree(self):
        result = three_way_vote("hold", "BUY", "BUY")
        assert result["consensus"] == "buy"
        assert result["method"] == "majority"
        assert result["agreement"] == "majority"

    def test_majority_llm_rf_agree(self):
        result = three_way_vote("sell", "SELL", "HOLD")
        assert result["consensus"] == "sell"
        assert result["method"] == "majority"
        assert result["agreement"] == "majority"

    def test_majority_llm_knn_agree(self):
        result = three_way_vote("strong_buy", "HOLD", "STRONG BUY")
        assert result["consensus"] == "strong_buy"
        assert result["method"] == "majority"
        assert result["agreement"] == "majority"

    def test_all_disagree_averages(self):
        # buy=1, hold=0, sell=-1 → avg=0 → hold
        result = three_way_vote("buy", "HOLD", "SELL")
        assert result["consensus"] == "hold"
        assert result["method"] == "score_average"
        assert result["agreement"] == "split"

    def test_all_disagree_bullish_lean(self):
        # strong_buy=2, buy=1, hold=0 → avg=1.0 → buy
        result = three_way_vote("strong_buy", "BUY", "HOLD")
        assert result["consensus"] == "buy"
        assert result["method"] == "score_average"

    def test_missing_classical_models(self):
        result = three_way_vote("buy", None, None)
        assert result["consensus"] == "buy"
        assert result["method"] == "majority"
        assert len(result["votes"]) == 1

    def test_one_classical_agrees(self):
        result = three_way_vote("sell", "SELL", None)
        assert result["consensus"] == "sell"
        assert result["method"] == "majority"
        assert result["agreement"] == "unanimous"

    def test_no_votes(self):
        result = three_way_vote("", None, None)
        assert result["consensus"] == "hold"
        assert result["method"] == "no_votes"

    def test_votes_dict_has_all_models(self):
        result = three_way_vote("buy", "HOLD", "SELL")
        assert "llm" in result["votes"]
        assert "rf" in result["votes"]
        assert "knn" in result["votes"]
