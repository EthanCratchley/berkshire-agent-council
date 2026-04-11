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


def test_sentiment_debate_response_restates_opponent_and_maintains_own_perspective():
    state = _state_with_news()
    state["debate"] = {
        "awaiting_response_from": "sentiment",
        "active_challenge": {
            "id": "macro_vs_sentiment",
            "action": "revise_or_defend",
            "reason": "Macro is more neutral.",
            "my_case": {"analyst": "sentiment", "rating": "strong_buy", "confidence": 0.78},
            "opponent_case": {
                "analyst": "macro",
                "rating": "hold",
                "confidence": 1.0,
                "details": "Macro indicators are neutral in aggregate.",
            },
            "coalition": {"supporters_of_opponent": [], "partial": []},
        },
    }

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"score": 9, "reasoning": "Strong positive momentum.", '
        '"claims_conceded": ["macro is neutral"], "claims_disputed": ["neutral macro implies no upside"], '
        '"weighting_statement": "Company catalysts dominate.", '
        '"horizon_alignment_note": "Catalyst-led sentiment matters in swing horizon.", '
        '"dialogue_response": "The opponent\'s bullish assessment is correct and mirrors my own.", '
        '"final_position": {"rating": "strong_buy", "confidence": 0.8}}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("nodes.sentiment_node.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = sentiment_node(state)

    response = result["analyst_signals"]["sentiment"]["debate_response"]
    assert "Opponent (macro) argues hold" in response
    assert "I concede" in response
    assert "I maintain strong_buy" in response


def test_sentiment_primary_opponent_takes_precedence_over_case_label():
    state = _state_with_news()
    state["debate"] = {
        "awaiting_response_from": "sentiment",
        "active_challenge": {
            "id": "macro_vs_sentiment",
            "primary_opponent": "macro",
            "action": "revise_or_defend",
            "reason": "Macro is more neutral.",
            "my_case": {"analyst": "sentiment", "rating": "strong_buy", "confidence": 0.78},
            "opponent_case": {
                "analyst": "sentiment",
                "rating": "hold",
                "confidence": 1.0,
                "details": "This field is intentionally wrong to verify canonical opponent handling.",
            },
            "coalition": {"supporters_of_opponent": [], "partial": []},
        },
    }

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"score": 9, "reasoning": "Strong positive momentum.", '
        '"claims_conceded": ["macro is neutral"], "claims_disputed": ["neutral macro implies no upside"], '
        '"weighting_statement": "Company catalysts dominate.", '
        '"horizon_alignment_note": "Catalyst-led sentiment matters in swing horizon.", '
        '"dialogue_response": "The opponent\'s bullish assessment is correct and mirrors my own.", '
        '"final_position": {"rating": "strong_buy", "confidence": 0.8}}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("nodes.sentiment_node.ChatGoogleGenerativeAI", return_value=mock_llm):
        result = sentiment_node(state)

    response = result["analyst_signals"]["sentiment"]["debate_response"]
    assert "Opponent (macro) argues hold" in response
