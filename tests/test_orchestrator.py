from orchestration.orchestrator import orchestrator
from shared.state_schema import make_initial_debate_state


def _state_with_signals(signals: dict):
    return {
        "ticker": "AAPL",
        "data": {},
        "analyst_signals": signals,
        "debate": make_initial_debate_state(max_rounds=3),
        "final_report": {},
    }


def test_no_contradictions_resolved_status():
    state = _state_with_signals({
        "sentiment": {"signal": "bullish", "confidence": 0.78, "details": "Positive press."},
        "fundamental": {"signal": "bullish", "confidence": 0.66, "details": "Strong margins."},
    })

    result = orchestrator(state)
    debate = result["debate"]

    assert debate["status"] == "resolved"
    assert debate["active_challenge"] is None
    assert debate["queue"] == []
    assert debate["unresolved_contradictions"] == []


def test_high_confidence_opposite_signals_create_challenge():
    state = _state_with_signals({
        "sentiment": {"signal": "bullish", "confidence": 0.91, "details": "Strong momentum."},
        "fundamental": {"signal": "bearish", "confidence": 0.72, "details": "Weak earnings quality."},
    })

    result = orchestrator(state)
    debate = result["debate"]

    assert debate["status"] == "debating"
    assert len(debate["queue"]) == 1
    challenge = debate["queue"][0]
    assert challenge["id"] == "fundamental_vs_sentiment"
    assert challenge["target"] == "fundamental"
    assert challenge["opponent"] == "sentiment"
    assert debate["active_challenge"]["id"] == challenge["id"]


def test_low_confidence_disagreement_is_not_queued():
    state = _state_with_signals({
        "sentiment": {"signal": "bullish", "confidence": 0.90, "details": "Strong momentum."},
        "fundamental": {"signal": "bearish", "confidence": 0.40, "details": "Weak earnings quality."},
    })

    result = orchestrator(state)
    debate = result["debate"]

    assert debate["status"] == "resolved"
    assert debate["queue"] == []


def test_multiple_contradictions_sorted_by_severity():
    state = _state_with_signals({
        "sentiment": {"signal": "bullish", "confidence": 0.90, "details": ""},
        "technical": {"signal": "bullish", "confidence": 0.58, "details": ""},
        "fundamental": {"signal": "bearish", "confidence": 0.80, "details": ""},
        "macro": {"signal": "bearish", "confidence": 0.60, "details": ""},
    })

    result = orchestrator(state)
    queue = result["debate"]["queue"]

    assert len(queue) == 4
    severities = [c["severity"] for c in queue]
    assert severities == sorted(severities, reverse=True)
    assert queue[0]["id"] == "fundamental_vs_sentiment"


def test_rating_only_payloads_are_supported():
    state = _state_with_signals({
        "sentiment": {"rating": "strong_buy", "confidence": 0.90, "details": ""},
        "fundamental": {"rating": "sell", "confidence": 0.80, "details": ""},
    })

    result = orchestrator(state)
    challenge = result["debate"]["active_challenge"]

    assert result["debate"]["status"] == "debating"
    assert challenge["score_distance"] == 3
    assert challenge["severity"] == 2.4  # 3 * min(0.90, 0.80)


def test_stance_score_takes_precedence_over_rating_and_signal():
    state = _state_with_signals({
        "sentiment": {
            "stance_score": 2,
            "rating": "hold",
            "signal": "neutral",
            "confidence": 0.90,
            "details": "",
        },
        "fundamental": {"stance_score": -2, "confidence": 0.80, "details": ""},
    })

    result = orchestrator(state)
    challenge = result["debate"]["active_challenge"]

    assert result["debate"]["status"] == "debating"
    assert challenge["score_distance"] == 4
    assert challenge["severity"] == 3.2  # 4 * min(0.90, 0.80)
