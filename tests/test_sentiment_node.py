from unittest.mock import MagicMock, patch

from nodes.sentiment_node import sentiment_node


def _state_with_news():
    return {
        "ticker": "AAPL",
        "horizon": "swing",
        "data": {
            "news_articles": [
                {
                    "headline": "Apple beats earnings",
                    "summary": "Strong quarter and upbeat guidance.",
                    "source": "Reuters",
                    "datetime": "2026-03-24T00:00:00+00:00",
                }
            ]
        },
        "analyst_signals": {},
    }


def test_no_news_defaults_to_hold():
    state = {
        "ticker": "AAPL",
        "horizon": "swing",
        "data": {"news_articles": []},
        "analyst_signals": {},
    }
    result = sentiment_node(state)
    sig = result["analyst_signals"]["sentiment"]

    assert sig["rating"] == "hold"
    assert sig["confidence"] == 0.0


def test_llm_score_maps_to_actionable_rating():
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"score": 9, "reasoning": "Strong positive momentum.", '
        '"claims_conceded": [], "claims_disputed": ["short-term pullback"], '
        '"weighting_statement": "Positive momentum dominates.", '
        '"horizon_alignment_note": "Momentum is supportive over a swing window.", '
        '"dialogue_response": "I understand the pullback concern, but the broader tone is still positive.", '
        '"final_position": {"rating": "strong_buy", "confidence": 0.8}}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("nodes.sentiment_node.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = sentiment_node(_state_with_news())

    sig = result["analyst_signals"]["sentiment"]
    assert sig["rating"] == "strong_buy"
    assert sig["features"]["sentiment_score"] == 9
