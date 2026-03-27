from itertools import combinations

from shared.state_schema import BerkshireState, make_initial_debate_state
from shared.stance import parse_rating, rating_to_score


HIGH_CONFIDENCE_THRESHOLD = 0.55
ANALYST_ORDER = ["sentiment", "fundamental", "technical", "macro"]


def analyst_to_node_name(analyst: str) -> str:
    mapping = {
        "sentiment": "sentiment_node",
        "fundamental": "fundamental_node",
        "technical": "technical_node",
        "macro": "macro_econ_node",
    }
    return mapping.get(analyst, "synthesizer_node")


def analyst_to_debate_node_name(analyst: str) -> str:
    """
    Debate-time routing can differ from initial collection routing.
    """
    if analyst == "fundamental":
        return "fundamental_debate_node"
    return analyst_to_node_name(analyst)


def _extract_signal_snapshot(signals: dict) -> dict:
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
    names = sorted(signal_snapshots.keys())
    outlier_scores = {}
    for name in names:
        numerator = 0.0
        denominator = 0.0
        for other in names:
            if other == name:
                continue
            d = _distance(signal_snapshots[name]["stance_score"], signal_snapshots[other]["stance_score"])
            w = signal_snapshots[other]["confidence"]
            numerator += d * w
            denominator += w
        outlier_scores[name] = round((numerator / denominator), 3) if denominator > 0 else 0.0
    return outlier_scores


def _pick_target_for_pair(a_name: str, b_name: str, signal_snapshots: dict, outlier_scores: dict):
    a_out, b_out = outlier_scores.get(a_name, 0.0), outlier_scores.get(b_name, 0.0)
    if a_out > b_out:
        return a_name, b_name
    if b_out > a_out:
        return b_name, a_name

    a_conf, b_conf = signal_snapshots[a_name]["confidence"], signal_snapshots[b_name]["confidence"]
    if a_conf < b_conf:
        return a_name, b_name
    if b_conf < a_conf:
        return b_name, a_name

    return (a_name, b_name) if a_name <= b_name else (b_name, a_name)


def _build_coalition_context(target: str, opponent: str, signal_snapshots: dict) -> dict:
    supporters_of_opponent, supporters_of_target, partial = [], [], []
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
    contradictions = []
    names = sorted(signal_snapshots.keys())
    outlier_scores = _compute_outlier_scores(signal_snapshots)
    for a_name, b_name in combinations(names, 2):
        a, b = signal_snapshots[a_name], signal_snapshots[b_name]
        if a["score_sign"] * b["score_sign"] != -1:
            continue
        if a["confidence"] < HIGH_CONFIDENCE_THRESHOLD or b["confidence"] < HIGH_CONFIDENCE_THRESHOLD:
            continue
        target, opponent = _pick_target_for_pair(a_name, b_name, signal_snapshots, outlier_scores)
        score_distance = abs(a["stance_score"] - b["stance_score"])
        severity = round(score_distance * min(a["confidence"], b["confidence"]), 2)
        contradictions.append(
            {
                "id": f"{a_name}_vs_{b_name}",
                "pair": [a_name, b_name],
                "severity": severity,
                "score_distance": score_distance,
                "target": target,
                "primary_opponent": opponent,
                "target_outlier_score": outlier_scores.get(target, 0.0),
                "opponent_outlier_score": outlier_scores.get(opponent, 0.0),
                "coalition": _build_coalition_context(target, opponent, signal_snapshots),
                "reason": (
                    f"{a_name}={a['rating']}({a['stance_score']:+d}, conf={a['confidence']:.2f}) conflicts with "
                    f"{b_name}={b['rating']}({b['stance_score']:+d}, conf={b['confidence']:.2f})."
                ),
                "action": "revise_or_defend",
            }
        )
    contradictions.sort(key=lambda item: (-item["severity"], item["id"]))
    return contradictions


def _next_missing_analyst(signal_snapshots: dict):
    for analyst in ANALYST_ORDER:
        if analyst not in signal_snapshots:
            return analyst
    return None


def _dispatch_debate_turn(
    debate: dict,
    contradictions: list,
    closed_pairs: list,
):
    round_no = int(debate.get("round", 0))
    max_rounds = int(debate.get("max_rounds", 3))
    awaiting = debate.get("awaiting_response_from")
    prev_active = debate.get("active_challenge", {}) if isinstance(debate.get("active_challenge"), dict) else {}
    prev_id = prev_active.get("id")

    active = contradictions[0]
    active_id = active["id"]
    history = []
    unresolved_to_add = []

    # If we already heard from target and contradiction is still active, give opponent one turn.
    if awaiting and prev_id == active_id and awaiting == active["target"]:
        speaker = active["primary_opponent"]
        turn_label = "opponent_turn"
    # If both target then opponent already spoke and still unresolved, close this pair and move on.
    elif awaiting and prev_id == active_id and awaiting == active["primary_opponent"]:
        if active_id not in closed_pairs:
            closed_pairs.append(active_id)
        unresolved_to_add.append(
            {
                "id": active_id,
                "reason": "Pair remained contradictory after target and opponent turns.",
                "challenge": active,
            }
        )
        contradictions = [c for c in contradictions if c["id"] not in closed_pairs]
        if not contradictions:
            return {
                "status": "resolved_with_unresolved_pairs",
                "next_node": "synthesizer_node",
                "round": round_no,
                "active_challenge": None,
                "awaiting_response_from": None,
                "queue": [],
                "closed_pairs": closed_pairs,
                "history": history + [{"event": "pair_closed_unresolved", "id": active_id}],
                "unresolved_to_add": unresolved_to_add,
            }
        active = contradictions[0]
        speaker = active["target"]
        turn_label = "target_turn"
    else:
        speaker = active["target"]
        turn_label = "target_turn"

    if round_no >= max_rounds:
        unresolved_to_add.append(
            {
                "id": active["id"],
                "reason": f"Max rounds reached before dispatching {speaker}.",
                "challenge": active,
            }
        )
        return {
            "status": "max_rounds_reached",
            "next_node": "synthesizer_node",
            "round": round_no,
            "active_challenge": active,
            "awaiting_response_from": None,
            "queue": contradictions,
            "closed_pairs": closed_pairs,
            "history": history + [{"event": "max_rounds_reached"}],
            "unresolved_to_add": unresolved_to_add,
        }

    round_no += 1
    history.append(
        {
            "event": "debate_turn_dispatched",
            "turn_type": turn_label,
            "speaker": speaker,
            "challenge_id": active["id"],
            "round": round_no,
        }
    )
    return {
        "status": "debating",
        "next_node": analyst_to_debate_node_name(speaker),
        "round": round_no,
        "active_challenge": active,
        "awaiting_response_from": speaker,
        "queue": contradictions,
        "closed_pairs": closed_pairs,
        "history": history,
        "unresolved_to_add": unresolved_to_add,
    }


def orchestrator(state: BerkshireState):
    ticker = state.get("ticker", "UNKNOWN")
    signals = state.get("analyst_signals", {})
    debate = state.get("debate") or make_initial_debate_state()

    signal_snapshots = _extract_signal_snapshot(signals)
    missing_analyst = _next_missing_analyst(signal_snapshots)
    contradictions_all = _find_contradictions(signal_snapshots) if signal_snapshots else []
    closed_pairs = list(debate.get("closed_pairs", [])) if isinstance(debate.get("closed_pairs"), list) else []
    contradictions = [c for c in contradictions_all if c["id"] not in closed_pairs]

    history_entry = {
        "event": "orchestrator_scan",
        "ticker": ticker,
        "round": int(debate.get("round", 0)),
        "analyst_count": len(signal_snapshots),
        "contradictions_found": len(contradictions),
    }
    debate_update = {"history": [history_entry]}

    if missing_analyst:
        node_name = analyst_to_node_name(missing_analyst)
        debate_update.update(
            {
                "_replace_lists": ["queue"],
                "queue": contradictions,
                "active_challenge": None,
                "awaiting_response_from": missing_analyst,
                "next_node": node_name,
                "status": "collecting_initial_analyst_stances",
            }
        )
        print(f"\n[Orchestrator] Collecting initial stance from: {missing_analyst}")
        return {"debate": debate_update}

    if not contradictions:
        debate_update.update(
            {
                "_replace_lists": ["queue"],
                "queue": [],
                "active_challenge": None,
                "awaiting_response_from": None,
                "next_node": "synthesizer_node",
                "status": "resolved",
                "closed_pairs": closed_pairs,
            }
        )
        print(f"\n[Orchestrator] No active contradictions for {ticker}. Routing to synthesis.")
        return {"debate": debate_update}

    dispatch = _dispatch_debate_turn(debate, contradictions, closed_pairs)
    replace_lists = ["queue", "closed_pairs"]
    if dispatch.get("unresolved_to_add"):
        # append unresolved entries; do not replace historical unresolved list.
        debate_update["unresolved_contradictions"] = dispatch["unresolved_to_add"]
    debate_update.update(
        {
            "_replace_lists": replace_lists,
            "queue": dispatch["queue"],
            "closed_pairs": dispatch["closed_pairs"],
            "active_challenge": dispatch["active_challenge"],
            "awaiting_response_from": dispatch["awaiting_response_from"],
            "next_node": dispatch["next_node"],
            "status": dispatch["status"],
            "round": dispatch["round"],
            "history": dispatch["history"],
        }
    )

    print("\n---ORCHESTRATOR REVIEW ---")
    print(f"Ticker: {ticker}")
    print(f"Analysts with valid stances: {len(signal_snapshots)}")
    print(f"Contradictions found: {len(contradictions)}")
    active = dispatch.get("active_challenge")
    if active:
        print(
            f"Active challenge: {active['id']} "
            f"(target={active['target']}, opponent={active['primary_opponent']}, severity={active['severity']})"
        )
    print(f"Dispatch -> {dispatch['next_node']}")
    print("------------------------------\n")

    return {"debate": debate_update}
