from itertools import combinations
from shared.state_schema import BerkshireState, make_initial_debate_state


SIGNAL_TO_INT = {
    "bullish": 1,
    "neutral": 0,
    "bearish": -1,
}
HIGH_CONFIDENCE_THRESHOLD = 0.55
RATING_TO_SCORE = {
    "strong_buy": 2,
    "buy": 1,
    "hold": 0,
    "sell": -1,
    "strong_sell": -2,
    # Common aliases
    "outperform": 1,
    "overweight": 1,
    "neutral": 0,
    "market_weight": 0,
    "underperform": -1,
    "underweight": -1,
}
SCORE_TO_SIGNAL = {
    1: "bullish",
    0: "neutral",
    -1: "bearish",
}
SCORE_TO_CANONICAL_RATING = {
    2: "strong_buy",
    1: "buy",
    0: "hold",
    -1: "sell",
    -2: "strong_sell",
}


def _normalize_rating(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _coerce_stance_score(value):
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return None
    if score < -2 or score > 2:
        return None
    return score


def _resolve_stance_score(payload: dict):
    """
    Priority:
      1) stance_score (+2..-2)
      2) rating string (strong_buy..strong_sell + aliases)
      3) legacy signal (bullish/neutral/bearish)
    """
    score = _coerce_stance_score(payload.get("stance_score"))
    if score is not None:
        return score, "stance_score"

    rating_key = _normalize_rating(payload.get("rating"))
    if rating_key in RATING_TO_SCORE:
        return RATING_TO_SCORE[rating_key], "rating"

    signal = payload.get("signal")
    if signal in SIGNAL_TO_INT:
        return SIGNAL_TO_INT[signal], "signal"

    return None, None


def _signal_from_score(score: int) -> str:
    sign = 1 if score > 0 else (-1 if score < 0 else 0)
    return SCORE_TO_SIGNAL[sign]


def _canonical_rating_from_score(score: int) -> str:
    return SCORE_TO_CANONICAL_RATING[score]


def _extract_signal_snapshot(signals: dict) -> dict:
    """
    Keep only analyst signals that have valid stance/confidence fields.
    """
    snapshots = {}
    for analyst, payload in (signals or {}).items():
        if not isinstance(payload, dict):
            continue
        stance_score, source = _resolve_stance_score(payload)
        confidence = payload.get("confidence")
        if stance_score is None:
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        snapshots[analyst] = {
            "stance_score": stance_score,
            "score_sign": 1 if stance_score > 0 else (-1 if stance_score < 0 else 0),
            "signal": payload.get("signal") or _signal_from_score(stance_score),
            "rating": _normalize_rating(payload.get("rating")) or _canonical_rating_from_score(stance_score),
            "confidence": max(0.0, min(confidence, 1.0)),
            "stance_source": source,
            "details": payload.get("details", ""),
        }
    return snapshots


def _find_contradictions(signal_snapshots: dict) -> list:
    """
    Build ranked contradiction payloads for high-confidence opposite stances.
    """
    contradictions = []
    analyst_names = sorted(signal_snapshots.keys())

    for a_name, b_name in combinations(analyst_names, 2):
        a = signal_snapshots[a_name]
        b = signal_snapshots[b_name]

        # Opposite signs only; skip neutral-involved pairs.
        if a["score_sign"] * b["score_sign"] != -1:
            continue

        # Ignore weak disagreements.
        if a["confidence"] < HIGH_CONFIDENCE_THRESHOLD or b["confidence"] < HIGH_CONFIDENCE_THRESHOLD:
            continue

        # Challenge the weaker-confidence side first.
        if a["confidence"] <= b["confidence"]:
            target, opponent = a_name, b_name
        else:
            target, opponent = b_name, a_name

        score_distance = abs(a["stance_score"] - b["stance_score"])
        severity = round(score_distance * min(a["confidence"], b["confidence"]), 2)
        contradiction = {
            "id": f"{a_name}_vs_{b_name}",
            "pair": [a_name, b_name],
            "severity": severity,
            "score_distance": score_distance,
            "target": target,
            "opponent": opponent,
            "reason": (
                f"{a_name}={a['rating']}({a['stance_score']:+d}, conf={a['confidence']:.2f}) conflicts with "
                f"{b_name}={b['rating']}({b['stance_score']:+d}, conf={b['confidence']:.2f})."
            ),
            "action": "revise_or_defend",
        }
        contradictions.append(contradiction)

    contradictions.sort(key=lambda item: (-item["severity"], item["id"]))
    return contradictions

def orchestrator(state: BerkshireState):
    """
    Orchestrator pass:
    - scans analyst outputs
    - detects contradictions
    - publishes ranked debate queue metadata

    Routing is still handled externally (no conditional loop in this step).
    """
    ticker = state.get("ticker", "UNKNOWN")
    signals = state.get("analyst_signals", {})
    debate = state.get("debate") or make_initial_debate_state()

    signal_snapshots = _extract_signal_snapshot(signals)
    contradictions = _find_contradictions(signal_snapshots)

    if contradictions:
        status = "debating"
        active_challenge = contradictions[0]
    else:
        status = "resolved" if signal_snapshots else "idle"
        active_challenge = None

    history_entry = {
        "event": "orchestrator_scan",
        "ticker": ticker,
        "round": debate.get("round", 0),
        "analyst_count": len(signal_snapshots),
        "contradictions_found": len(contradictions),
        "status": status,
    }

    print("\n---ORCHESTRATOR REVIEW ---")
    print(f"Ticker: {ticker}")
    print(f"Analysts with valid stances: {len(signal_snapshots)}")
    print(f"Contradictions found: {len(contradictions)}")
    if active_challenge:
        print(
            f"Top challenge: {active_challenge['id']} "
            f"(target={active_challenge['target']}, severity={active_challenge['severity']})"
        )
    print("------------------------------\n")

    return {
        "debate": {
            "_replace_lists": ["queue", "unresolved_contradictions"],
            "queue": contradictions,
            "unresolved_contradictions": contradictions,
            "active_challenge": active_challenge,
            "status": status,
            "history": [history_entry],
        }
    }
