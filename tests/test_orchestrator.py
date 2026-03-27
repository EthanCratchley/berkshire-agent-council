from orchestration.orchestrator import orchestrator
from shared.state_schema import make_initial_debate_state


def _state_with_signals(signals: dict, debate_overrides=None):
    debate = make_initial_debate_state(max_rounds=3)
    if debate_overrides:
        debate.update(debate_overrides)
    return {
        "ticker": "AAPL",
        "data": {},
        "analyst_signals": signals,
        "debate": debate,
        "final_report": {},
    }


def _all_nodes_same_side():
    return {
        "sentiment": {"rating": "buy", "confidence": 0.8, "details": ""},
        "fundamental": {"rating": "strong_buy", "confidence": 0.7, "details": ""},
        "technical": {"rating": "buy", "confidence": 0.6, "details": ""},
        "macro": {"rating": "hold", "confidence": 0.7, "details": ""},
    }


def test_collects_missing_initial_analysts():
    state = _state_with_signals(
        {
            "sentiment": {"rating": "buy", "confidence": 0.8, "details": ""},
        }
    )
    result = orchestrator(state)
    debate = result["debate"]
    assert debate["status"] == "collecting_initial_analyst_stances"
    assert debate["next_node"] == "fundamental_node"
    assert debate["awaiting_response_from"] == "fundamental"


def test_resolved_routes_to_synthesizer():
    state = _state_with_signals(_all_nodes_same_side())
    result = orchestrator(state)
    debate = result["debate"]
    assert debate["status"] == "resolved"
    assert debate["next_node"] == "synthesizer_node"
    assert debate["active_challenge"] is None


def test_first_debate_turn_targets_outlier():
    state = _state_with_signals(
        {
            "sentiment": {"rating": "strong_buy", "confidence": 0.9, "details": ""},
            "fundamental": {"rating": "sell", "confidence": 0.72, "details": ""},
            "technical": {"rating": "buy", "confidence": 0.8, "details": ""},
            "macro": {"rating": "hold", "confidence": 0.7, "details": ""},
        }
    )
    result = orchestrator(state)
    debate = result["debate"]
    assert debate["status"] == "debating"
    assert debate["next_node"] == "fundamental_node"
    assert debate["awaiting_response_from"] == "fundamental"
    assert debate["active_challenge"]["primary_opponent"] == "sentiment"


def test_second_turn_dispatches_primary_opponent():
    state = _state_with_signals(
        {
            "sentiment": {"rating": "strong_buy", "confidence": 0.9, "details": ""},
            "fundamental": {"rating": "sell", "confidence": 0.72, "details": ""},
            "technical": {"rating": "buy", "confidence": 0.8, "details": ""},
            "macro": {"rating": "hold", "confidence": 0.7, "details": ""},
        },
        debate_overrides={
            "awaiting_response_from": "fundamental",
            "active_challenge": {"id": "fundamental_vs_sentiment"},
            "round": 1,
        },
    )
    result = orchestrator(state)
    debate = result["debate"]
    assert debate["next_node"] == "sentiment_node"
    assert debate["awaiting_response_from"] == "sentiment"
    assert debate["round"] == 2


def test_max_rounds_routes_to_synthesizer():
    state = _state_with_signals(
        {
            "sentiment": {"rating": "strong_buy", "confidence": 0.9, "details": ""},
            "fundamental": {"rating": "sell", "confidence": 0.72, "details": ""},
            "technical": {"rating": "buy", "confidence": 0.8, "details": ""},
            "macro": {"rating": "hold", "confidence": 0.7, "details": ""},
        },
        debate_overrides={"round": 3, "max_rounds": 3},
    )
    result = orchestrator(state)
    debate = result["debate"]
    assert debate["status"] == "max_rounds_reached"
    assert debate["next_node"] == "synthesizer_node"

