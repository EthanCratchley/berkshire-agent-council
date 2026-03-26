from itertools import combinations
from shared.state_schema import BerkshireState, make_initial_debate_state
from shared.stance import (
    parse_rating,
    rating_to_score,
)


HIGH_CONFIDENCE_THRESHOLD = 0.55


def _extract_signal_snapshot(signals: dict) -> dict:
    """
    Keep only analyst signals that have valid rating/confidence fields.
    """
    snapshots = {}
    for analyst, payload in (signals or {}).items():
        if not isinstance(payload, dict):
            continue
        rating = parse_rating(payload.get("rating"))
        confidence = payload.get("confidence")
        if rating is None:
            continue
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            continue
        stance_score = rating_to_score(rating)
        snapshots[analyst] = {
            "stance_score": stance_score,
            "score_sign": 1 if stance_score > 0 else (-1 if stance_score < 0 else 0),
            "rating": rating.value,
            "confidence": max(0.0, min(confidence, 1.0)),
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
