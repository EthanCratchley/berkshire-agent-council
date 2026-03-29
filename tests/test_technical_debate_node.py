from unittest.mock import MagicMock, patch

from nodes.technical_debate_node import technical_debate_node


def test_technical_debate_node_preserves_quant_and_adds_narrative():
    state = {
        "ticker": "AAPL",
        "data": {},
        "analyst_signals": {
            "technical": {
                "rating": "buy",
                "confidence": 0.6,
                "features": {
                    "rsi": 40.0,
                    "macd_histogram": 0.5,
                    "sma_20_50_cross": 1,
                    "bollinger_pct": 0.3,
                    "volume_ratio": 1.5,
                    "price_change_5d": 0.02,
                    "price_change_20d": 0.05,
                },
                "details": "Baseline technical stance.",
            }
        },
        "debate": {
            "awaiting_response_from": "technical",
            "active_challenge": {
                "id": "technical_vs_sentiment",
                "action": "revise_or_defend",
                "reason": "Sentiment is more bullish.",
                "opponent_case": {"analyst": "sentiment", "rating": "strong_buy", "confidence": 0.8},
                "coalition": {"supporters_of_opponent": [], "partial": []},
            },
        },
        "final_report": {},
    }
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"explanation":"Technical trend remains mixed but weakening.",'
        '"claims_conceded":["short-term sentiment stress exists"],'
        '"claims_disputed":["strong buy is justified now"],'
        '"weighting_statement":"price momentum remains the dominant short-horizon signal",'
        '"horizon_alignment_note":"Technical setup is calibrated for swing horizon.",'
        '"dialogue_response":"I hear the bullish case, but current momentum still looks weak.",'
        '"final_position":{"rating":"hold","confidence":0.58}}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("nodes.technical_debate_node.ChatGoogleGenerativeAI", return_value=mock_llm), patch(
        "nodes.technical_debate_node.os.getenv",
        return_value="fake-key",
    ):
        result = technical_debate_node(state)
    sig = result["analyst_signals"]["technical"]

    assert sig["rating"] in ("strong_buy", "buy", "hold", "sell", "strong_sell")
    assert 0.0 <= sig["confidence"] <= 1.0
    assert isinstance(sig["narrative"], str)
    assert isinstance(sig["debate_response"], str)
    assert isinstance(sig["position_changed"], bool)
    assert isinstance(sig["counterpoints_addressed"], list)
    assert "Technical explanation" in sig["details"]
