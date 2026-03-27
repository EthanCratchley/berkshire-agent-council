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


def _distance(a_score: int, b_score: int) -> int:
    return abs(a_score - b_score)


def _compute_outlier_scores(signal_snapshots: dict) -> dict:
    """
    Outlier score per analyst = weighted average distance to all others.
    Weights use counterpart confidence to emphasize disagreement with trusted peers.
    """
    names = sorted(signal_snapshots.keys())
    outlier_scores = {}
    for name in names:
        numerator = 0.0
        denominator = 0.0
        for other in names:
            if other == name:
                continue
            d = _distance(
                signal_snapshots[name]["stance_score"],
                signal_snapshots[other]["stance_score"],
            )
            weight = signal_snapshots[other]["confidence"]
            numerator += d * weight
            denominator += weight
        outlier_scores[name] = round((numerator / denominator), 3) if denominator > 0 else 0.0
    return outlier_scores


def _pick_target_for_pair(a_name: str, b_name: str, signal_snapshots: dict, outlier_scores: dict):
    """
    Select debate target inside a conflicting pair.
    Priority:
    1) higher global outlier score
    2) lower confidence
    3) alphabetical (deterministic fallback)
    """
    a_out = outlier_scores.get(a_name, 0.0)
    b_out = outlier_scores.get(b_name, 0.0)
    if a_out > b_out:
        return a_name, b_name
    if b_out > a_out:
        return b_name, a_name

    a_conf = signal_snapshots[a_name]["confidence"]
    b_conf = signal_snapshots[b_name]["confidence"]
    if a_conf < b_conf:
        return a_name, b_name
    if b_conf < a_conf:
        return b_name, a_name

    if a_name <= b_name:
        return a_name, b_name
    return b_name, a_name


def _build_coalition_context(target: str, opponent: str, signal_snapshots: dict) -> dict:
    """
    Build coalition support buckets relative to the target/opponent pair.
    """
    supporters_of_opponent = []
    supporters_of_target = []
    partial = []

    target_score = signal_snapshots[target]["stance_score"]
    opponent_score = signal_snapshots[opponent]["stance_score"]

    for name, snap in signal_snapshots.items():
        if name in (target, opponent):
            continue
        d_target = _distance(snap["stance_score"], target_score)
        d_opp = _distance(snap["stance_score"], opponent_score)

        entry = {
            "analyst": name,
            "rating": snap["rating"],
            "confidence": snap["confidence"],
            "distance_to_target": d_target,
            "distance_to_opponent": d_opp,
        }

        # Margin-of-1 rule creates a "partial agreement bubble".
        if d_opp + 1 < d_target:
            supporters_of_opponent.append(entry)
        elif d_target + 1 < d_opp:
            supporters_of_target.append(entry)
        else:
            partial.append(entry)

    weighted_support_opponent = round(sum(e["confidence"] for e in supporters_of_opponent), 3)
    weighted_support_target = round(sum(e["confidence"] for e in supporters_of_target), 3)
    net_support_for_opponent = round(weighted_support_opponent - weighted_support_target, 3)

    return {
        "supporters_of_opponent": supporters_of_opponent,
        "supporters_of_target": supporters_of_target,
        "partial": partial,
        "weighted_support_opponent": weighted_support_opponent,
        "weighted_support_target": weighted_support_target,
        "net_support_for_opponent": net_support_for_opponent,
    }


def _find_contradictions(signal_snapshots: dict) -> list:
    """
    Build ranked contradiction payloads for high-confidence opposite stances.
    Includes outlier-aware target selection and coalition context.
    """
    contradictions = []
    analyst_names = sorted(signal_snapshots.keys())
    outlier_scores = _compute_outlier_scores(signal_snapshots)

    for a_name, b_name in combinations(analyst_names, 2):
        a = signal_snapshots[a_name]
        b = signal_snapshots[b_name]

        # Opposite signs only; skip neutral-involved pairs.
        if a["score_sign"] * b["score_sign"] != -1:
            continue

        # Ignore weak disagreements.
        if a["confidence"] < HIGH_CONFIDENCE_THRESHOLD or b["confidence"] < HIGH_CONFIDENCE_THRESHOLD:
            continue

        target, opponent = _pick_target_for_pair(a_name, b_name, signal_snapshots, outlier_scores)

        score_distance = abs(a["stance_score"] - b["stance_score"])
        severity = round(score_distance * min(a["confidence"], b["confidence"]), 2)
        coalition = _build_coalition_context(target, opponent, signal_snapshots)
        contradiction = {
            "id": f"{a_name}_vs_{b_name}",
            "pair": [a_name, b_name],
            "severity": severity,
            "score_distance": score_distance,
            "target": target,
            "primary_opponent": opponent,
            "target_outlier_score": outlier_scores.get(target, 0.0),
            "opponent_outlier_score": outlier_scores.get(opponent, 0.0),
            "coalition": coalition,
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
        coalition = active_challenge.get("coalition", {})
        print(
            "[Orchestrator] Coalition:"
            f" opp_supporters={len(coalition.get('supporters_of_opponent', []))},"
            f" target_supporters={len(coalition.get('supporters_of_target', []))},"
            f" partial={len(coalition.get('partial', []))}"
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
