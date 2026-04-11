from unittest.mock import MagicMock, patch

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
                "opponent_case": {"analyst": "sentiment", "rating": "strong_buy", "confidence": 0.8},
                "coalition": {"supporters_of_opponent": [], "partial": []},
            },
        },
        "final_report": {},
    }
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"explanation":"Apple fundamentals remain solid with reasonable valuation.",'
        '"claims_conceded":["sentiment momentum is strong"],'
        '"claims_disputed":["strong buy is justified by fundamentals alone"],'
        '"weighting_statement":"valuation and profitability dominate the fundamental view",'
        '"horizon_alignment_note":"Fundamental stance is calibrated for swing horizon.",'
        '"dialogue_response":"I acknowledge bullish sentiment, but valuation keeps me at buy.",'
        '"final_position":{"rating":"buy","confidence":0.6}}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("nodes.fundamental_debate_node.ChatGoogleGenerativeAI", return_value=mock_llm), patch(
        "nodes.fundamental_debate_node.os.getenv",
        return_value="fake-key",
    ):
        result = fundamental_debate_node(state)
    sig = result["analyst_signals"]["fundamental"]

    assert sig["rating"] in ("strong_buy", "buy", "hold", "sell", "strong_sell")
    assert 0.0 <= sig["confidence"] <= 1.0
    assert isinstance(sig["narrative"], str)
    assert isinstance(sig["debate_response"], str)
    assert isinstance(sig["position_changed"], bool)
    assert isinstance(sig["counterpoints_addressed"], list)
    assert "Fundamental explanation" in sig["details"]
    assert "deterministic_decision" in sig


def test_fundamental_debate_deterministic_policy_not_overridden_by_llm():
    state = {
        "ticker": "AAPL",
        "horizon": "short",
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
                "severity": 1.4,
                "action": "revise_or_defend",
                "reason": "Sentiment is more bullish.",
                "primary_opponent": "sentiment",
                "opponent_case": {"analyst": "sentiment", "rating": "strong_buy", "confidence": 0.8},
                "coalition": {"supporters_of_opponent": [], "partial": []},
            },
        },
        "final_report": {},
    }

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"explanation":"Fundamentals remain solid with balanced valuation.",'
        '"claims_conceded":["near-term sentiment tailwind exists"],'
        '"claims_disputed":["sentiment alone warrants strong_buy"],'
        '"weighting_statement":"profitability and valuation remain balanced",'
        '"horizon_alignment_note":"Short horizon still references financial base quality.",'
        '"dialogue_response":"I acknowledge momentum, but fundamentals still justify a measured stance.",'
        '"final_position":{"rating":"strong_sell","confidence":0.01}}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("nodes.fundamental_debate_node.ChatGoogleGenerativeAI", return_value=mock_llm), patch(
        "nodes.fundamental_debate_node.os.getenv",
        return_value="fake-key",
    ):
        result = fundamental_debate_node(state)

    sig = result["analyst_signals"]["fundamental"]
    assert sig["deterministic_decision"]["final_position"]["rating"] == sig["rating"]
    assert sig["deterministic_decision"]["final_position"]["confidence"] == sig["confidence"]
