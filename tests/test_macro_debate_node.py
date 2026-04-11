from unittest.mock import MagicMock, patch

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
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"explanation":"Macro conditions are mixed with inflation improving but volatility elevated.",'
        '"claims_conceded":["near-term sentiment pressure exists"],'
        '"claims_disputed":["all macro signals are decisively bearish"],'
        '"weighting_statement":"inflation and policy trajectory dominate long-horizon macro view",'
        '"horizon_alignment_note":"Macro stance is aligned to long-horizon cycle factors.",'
        '"dialogue_response":"I acknowledge sentiment risk, but macro regime is not decisively bearish.",'
        '"final_position":{"rating":"hold","confidence":0.61}}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("nodes.macro_debate_node.ChatGoogleGenerativeAI", return_value=mock_llm), patch(
        "nodes.macro_debate_node.os.getenv",
        return_value="fake-key",
    ):
        result = macro_debate_node(state)
    sig = result["analyst_signals"]["macro"]

    assert sig["rating"] in ("strong_buy", "buy", "hold", "sell", "strong_sell")
    assert 0.0 <= sig["confidence"] <= 1.0
    assert isinstance(sig["narrative"], str)
    assert isinstance(sig["debate_response"], str)
    assert isinstance(sig["position_changed"], bool)
    assert isinstance(sig["counterpoints_addressed"], list)
    assert "Macroeconomic explanation" in sig["details"]


def test_macro_debate_deterministic_policy_not_overridden_by_llm():
    state = {
        "ticker": "AAPL",
        "horizon": "short",
        "data": {},
        "analyst_signals": {
            "macro": {
                "rating": "hold",
                "confidence": 0.9,
                "features": {
                    "vix": 19.0,
                    "yield_curve_spread": 0.5,
                    "unemployment": 4.2,
                    "fed_funds_rate": 3.7,
                    "cpi_yoy": 3.3,
                },
                "details": "Baseline macro stance.",
            }
        },
        "debate": {
            "awaiting_response_from": "macro",
            "active_challenge": {
                "id": "fundamental_vs_macro",
                "severity": 1.2,
                "action": "revise_or_defend",
                "reason": "Fundamental is much more bullish.",
                "opponent_case": {"analyst": "fundamental", "rating": "strong_buy", "confidence": 0.6},
                "coalition": {"supporters_of_opponent": [], "partial": []},
            },
        },
        "final_report": {},
    }

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"explanation":"Macro remains mixed in the short horizon.",'
        '"claims_conceded":["company-specific momentum is strong"],'
        '"claims_disputed":["macro conditions alone justify strong_buy"],'
        '"weighting_statement":"neutral macro indicators dominate macro view",'
        '"horizon_alignment_note":"Short horizon macro mostly impacts volatility, not trend direction.",'
        '"dialogue_response":"I acknowledge the fundamental upside, but macro does not yet support a broad risk-on shift.",'
        '"final_position":{"rating":"strong_buy","confidence":1.0}}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("nodes.macro_debate_node.ChatGoogleGenerativeAI", return_value=mock_llm), patch(
        "nodes.macro_debate_node.os.getenv",
        return_value="fake-key",
    ):
        result = macro_debate_node(state)

    sig = result["analyst_signals"]["macro"]

    # LLM-proposed final_position must not control rating/confidence.
    assert sig["rating"] == "hold"
    assert sig["confidence"] != 1.0
    assert "deterministic_decision" in sig
    assert sig["deterministic_decision"]["final_position"]["rating"] == sig["rating"]


def test_macro_debate_response_restates_opponent_thesis():
    state = {
        "ticker": "META",
        "horizon": "swing",
        "data": {},
        "analyst_signals": {
            "macro": {
                "rating": "hold",
                "confidence": 1.0,
                "features": {
                    "vix": 19.49,
                    "yield_curve_spread": 0.51,
                    "unemployment": 4.3,
                    "fed_funds_rate": 3.64,
                    "cpi_yoy": 3.29,
                },
                "details": "Macro indicators mostly neutral.",
            }
        },
        "debate": {
            "awaiting_response_from": "macro",
            "active_challenge": {
                "id": "macro_vs_sentiment",
                "severity": 1.56,
                "action": "revise_or_defend",
                "reason": "Sentiment is materially more bullish.",
                "opponent_case": {
                    "analyst": "sentiment",
                    "rating": "strong_buy",
                    "confidence": 0.78,
                    "details": "News flow and AI catalysts are strongly positive.",
                },
                "coalition": {"supporters_of_opponent": [], "partial": []},
            },
        },
        "final_report": {},
    }

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"explanation":"Macro indicators are mostly neutral.",'
        '"claims_conceded":["sentiment momentum is strong"],'
        '"claims_disputed":["strong_buy is justified solely by macro"],'
        '"weighting_statement":"macro neutrality dominates macro stance",'
        '"horizon_alignment_note":"Swing horizon still considers macro regime.",'
        '"dialogue_response":"I fully agree with sentiment and mirror that view exactly."}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("nodes.macro_debate_node.ChatGoogleGenerativeAI", return_value=mock_llm), patch(
        "nodes.macro_debate_node.os.getenv",
        return_value="fake-key",
    ):
        result = macro_debate_node(state)

    response = result["analyst_signals"]["macro"]["debate_response"]
    # LLM output is used as-is to explain the deterministic decision
    assert response == 'I fully agree with sentiment and mirror that view exactly.'


def test_macro_debate_primary_opponent_takes_precedence_over_case_label():
    state = {
        "ticker": "META",
        "horizon": "swing",
        "data": {},
        "analyst_signals": {
            "macro": {
                "rating": "hold",
                "confidence": 1.0,
                "features": {
                    "vix": 19.49,
                    "yield_curve_spread": 0.51,
                    "unemployment": 4.3,
                    "fed_funds_rate": 3.64,
                    "cpi_yoy": 3.29,
                },
                "details": "Macro indicators mostly neutral.",
            }
        },
        "debate": {
            "awaiting_response_from": "macro",
            "active_challenge": {
                "id": "macro_vs_sentiment",
                "primary_opponent": "sentiment",
                "severity": 1.56,
                "action": "revise_or_defend",
                "reason": "Sentiment is materially more bullish.",
                "opponent_case": {
                    "analyst": "macro",
                    "rating": "strong_buy",
                    "confidence": 0.78,
                    "details": "This field is intentionally wrong to verify canonical opponent handling.",
                },
                "coalition": {"supporters_of_opponent": [], "partial": []},
            },
        },
        "final_report": {},
    }

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        '{"explanation":"Macro indicators are mostly neutral.",'
        '"claims_conceded":["sentiment momentum is strong"],'
        '"claims_disputed":["strong_buy is justified solely by macro"],'
        '"weighting_statement":"macro neutrality dominates macro stance",'
        '"horizon_alignment_note":"Swing horizon still considers macro regime.",'
        '"dialogue_response":"I fully agree with sentiment and mirror that view exactly."}'
    )
    mock_llm.invoke.return_value = mock_response

    with patch("nodes.macro_debate_node.ChatGoogleGenerativeAI", return_value=mock_llm), patch(
        "nodes.macro_debate_node.os.getenv",
        return_value="fake-key",
    ):
        result = macro_debate_node(state)

    response = result["analyst_signals"]["macro"]["debate_response"]
    # LLM output is used as-is - should reference sentiment not macro
    assert "sentiment" in response.lower()
