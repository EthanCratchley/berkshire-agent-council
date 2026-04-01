from itertools import combinations

from shared.horizon import normalize_horizon, horizon_label
from shared.state_schema import BerkshireState, make_initial_debate_state
from shared.stance import parse_rating, rating_to_score


EFFECTIVE_CONFIDENCE_THRESHOLD = 0.30
STAGNATION_LIMIT = 2
MIN_CONTRADICTION_DISTANCE = 2
MIN_CONTRADICTION_SEVERITY = 1.0
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
    if analyst == "fundamental":
        return "fundamental_debate_node"
    if analyst == "technical":
        return "technical_debate_node"
    if analyst == "macro":
        return "macro_debate_node"
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
        snapshots[analyst] = {
            "stance_score": rating_to_score(rating),
            "score_sign": 1 if rating_to_score(rating) > 0 else (-1 if rating_to_score(rating) < 0 else 0),
            "rating": rating.value,
            "confidence": max(0.0, min(confidence, 1.0)),
            "details": payload.get("details", ""),
            "debate_response": payload.get("debate_response", ""),
            "position_changed": bool(payload.get("position_changed", False)),
            "counterpoints_addressed": payload.get("counterpoints_addressed", []),
            "claims_conceded": payload.get("claims_conceded", []),
            "claims_disputed": payload.get("claims_disputed", []),
            "final_position": payload.get("final_position", {}),
            "weighting_statement": payload.get("weighting_statement", ""),
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
            numerator += _distance(
                signal_snapshots[name]["stance_score"],
                signal_snapshots[other]["stance_score"],
            ) * signal_snapshots[other]["confidence"]
            denominator += signal_snapshots[other]["confidence"]
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
            "details": snap["details"],
            "distance_to_target": d_target,
            "distance_to_opponent": d_opp,
        }
        if d_opp + 1 < d_target:
            supporters_of_opponent.append(entry)
        elif d_target + 1 < d_opp:
            supporters_of_target.append(entry)
        else:
            partial.append(entry)

    return {
        "supporters_of_opponent": supporters_of_opponent,
        "supporters_of_target": supporters_of_target,
        "partial": partial,
        "weighted_support_opponent": round(sum(e["confidence"] for e in supporters_of_opponent), 3),
        "weighted_support_target": round(sum(e["confidence"] for e in supporters_of_target), 3),
    }


def _find_contradictions(signal_snapshots: dict) -> list:
    contradictions = []
    names = sorted(signal_snapshots.keys())
    outlier_scores = _compute_outlier_scores(signal_snapshots)

    for a_name, b_name in combinations(names, 2):
        a, b = signal_snapshots[a_name], signal_snapshots[b_name]
        if min(a["confidence"], b["confidence"]) < EFFECTIVE_CONFIDENCE_THRESHOLD:
            continue

        score_distance = abs(a["stance_score"] - b["stance_score"])
        if score_distance < MIN_CONTRADICTION_DISTANCE:
            continue

        severity = round(score_distance * min(a["confidence"], b["confidence"]), 2)
        if severity < MIN_CONTRADICTION_SEVERITY:
            continue

        target, opponent = _pick_target_for_pair(a_name, b_name, signal_snapshots, outlier_scores)
        coalition = _build_coalition_context(target, opponent, signal_snapshots)

        contradictions.append(
            {
                "id": f"{a_name}_vs_{b_name}",
                "pair": [a_name, b_name],
                "severity": severity,
                "score_distance": score_distance,
                "target": target,
                "primary_opponent": opponent,
                "target_snapshot": signal_snapshots[target],
                "opponent_snapshot": signal_snapshots[opponent],
                "my_case": {
                    "analyst": target,
                    "rating": signal_snapshots[target]["rating"],
                    "confidence": signal_snapshots[target]["confidence"],
                    "details": signal_snapshots[target]["details"],
                    "last_debate_response": signal_snapshots[target].get("debate_response", ""),
                },
                "opponent_case": {
                    "analyst": opponent,
                    "rating": signal_snapshots[opponent]["rating"],
                    "confidence": signal_snapshots[opponent]["confidence"],
                    "details": signal_snapshots[opponent]["details"],
                    "last_debate_response": signal_snapshots[opponent].get("debate_response", ""),
                },
                "target_outlier_score": outlier_scores.get(target, 0.0),
                "opponent_outlier_score": outlier_scores.get(opponent, 0.0),
                "coalition": coalition,
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


def _signature_for_pair(challenge: dict, signal_snapshots: dict):
    def _normalize_text(text: str, limit: int = 160) -> str:
        cleaned = " ".join(str(text or "").strip().lower().split())
        return cleaned[:limit]

    def _list_sig(values) -> tuple:
        if not isinstance(values, list):
            return ()
        normalized = sorted({_normalize_text(v, limit=80) for v in values if str(v or "").strip()})
        return tuple(normalized[:6])

    def _snap_sig(snap: dict):
        conf_raw = snap.get("confidence", 0.0)
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = 0.0
        conf_bucket = round(conf / 0.05) * 0.05
        return (
            snap.get("rating"),
            round(conf_bucket, 2),
            _normalize_text(snap.get("weighting_statement", ""), limit=140),
            _list_sig(snap.get("claims_conceded", [])),
            _list_sig(snap.get("claims_disputed", [])),
            _normalize_text(snap.get("debate_response", ""), limit=160),
        )

    a, b = challenge["pair"][0], challenge["pair"][1]
    snap_a = signal_snapshots.get(a, {})
    snap_b = signal_snapshots.get(b, {})
    return (
        _snap_sig(snap_a),
        _snap_sig(snap_b),
    )


def _record_debater_turn_if_available(
    debate: dict,
    signal_snapshots: dict,
):
    """
    Save human-readable transcript entry from the last dispatched speaker.
    """
    if debate.get("status") not in {"debating", "confirming_resolution"}:
        return None, None
    active = debate.get("active_challenge", {})
    if not isinstance(active, dict) or not active.get("id"):
        return None, None
    valid_speakers = {active.get("target"), active.get("primary_opponent")}
    awaiting = debate.get("awaiting_response_from")
    if not awaiting:
        return None, None
    if awaiting not in valid_speakers:
        return None, None
    speaker = awaiting
    snap = signal_snapshots.get(speaker, {})
    if not snap:
        return None, None

    entry = {
        "event": "debater_turn_result",
        "speaker": speaker,
        "rating": snap.get("rating"),
        "confidence": snap.get("confidence"),
        "debate_response": snap.get("debate_response", ""),
        "position_changed": snap.get("position_changed", False),
        "counterpoints_addressed": snap.get("counterpoints_addressed", []),
        "claims_conceded": snap.get("claims_conceded", []),
        "claims_disputed": snap.get("claims_disputed", []),
        "final_position": snap.get("final_position", {}),
        "weighting_statement": snap.get("weighting_statement", ""),
    }
    natural = str(entry.get("debate_response", "")).strip()
    if natural:
        entry["conversation_line"] = (
            f"{speaker.title()}: {natural} "
            f"(final: {entry.get('rating')}, conf={entry.get('confidence')})"
        )
    else:
        entry["conversation_line"] = (
            f"{speaker.title()}: No detailed rebuttal provided. "
            f"(final: {entry.get('rating')}, conf={entry.get('confidence')})"
        )
    return entry, speaker


def orchestrator(state: BerkshireState):
    ticker = state.get("ticker", "UNKNOWN")
    selected_horizon = normalize_horizon(state.get("horizon", "swing"))
    signals = state.get("analyst_signals", {})
    debate = state.get("debate") or make_initial_debate_state()

    signal_snapshots = _extract_signal_snapshot(signals)
    contradictions = _find_contradictions(signal_snapshots)
    effective_analysts = [
        name for name, snap in signal_snapshots.items()
        if float(snap.get("confidence", 0.0)) >= EFFECTIVE_CONFIDENCE_THRESHOLD
    ]

    history = [
        {
            "event": "orchestrator_scan",
            "ticker": ticker,
            "round": int(debate.get("round", 0)),
            "analyst_count": len(signal_snapshots),
            "effective_analyst_count": len(effective_analysts),
            "contradictions_found": len(contradictions),
        }
    ]

    # Store last speaker's dialogue response if this call follows a debate turn.
    turn_entry, last_speaker = _record_debater_turn_if_available(debate, signal_snapshots)
    if turn_entry:
        history.append(turn_entry)
        print(f"[Debate] {turn_entry.get('conversation_line', '')}")

    # Initial collection phase.
    missing = _next_missing_analyst(signal_snapshots)
    if missing:
        print(f"\n[Orchestrator] Collecting initial stance from: {missing}")
        return {
            "debate": {
                "_replace_lists": ["queue"],
                "queue": contradictions,
                "active_challenge": None,
                "awaiting_response_from": missing,
                "next_node": analyst_to_node_name(missing),
                "status": "collecting_initial_analyst_stances",
                "history": history,
            }
        }

    # No contradiction -> synthesize.
    if not contradictions:
        active_prev = debate.get("active_challenge", {}) if isinstance(debate.get("active_challenge"), dict) else {}
        previous_speaker = debate.get("awaiting_response_from")
        no_contradiction_streak = int(debate.get("no_contradiction_streak", 0))

        # Require one confirmation turn after a debate appears to converge.
        if debate.get("status") in {"debating", "confirming_resolution"} and active_prev.get("id"):
            target = active_prev.get("target")
            opponent = active_prev.get("primary_opponent")
            confirmation_speaker = None
            if previous_speaker == target and opponent:
                confirmation_speaker = opponent
            elif previous_speaker == opponent and target:
                confirmation_speaker = target

            if no_contradiction_streak < 1 and confirmation_speaker:
                print("\n[Orchestrator] No contradictions found. Dispatching one confirmation turn before synthesis.")
                return {
                    "debate": {
                        "_replace_lists": ["queue"],
                        "queue": [],
                        "active_challenge": active_prev,
                        "awaiting_response_from": confirmation_speaker,
                        "next_node": analyst_to_debate_node_name(confirmation_speaker),
                        "status": "confirming_resolution",
                        "no_contradiction_streak": no_contradiction_streak + 1,
                        "history": history + [
                            {
                                "event": "resolution_confirmation_dispatched",
                                "speaker": confirmation_speaker,
                                "challenge_id": active_prev.get("id"),
                            }
                        ],
                    }
                }

        return {
            "debate": {
                "_replace_lists": ["queue"],
                "queue": [],
                "active_challenge": None,
                "awaiting_response_from": None,
                "next_node": "synthesizer_node",
                "status": "resolved",
                "no_contradiction_streak": no_contradiction_streak + 1,
                "history": history,
            }
        }

    round_no = int(debate.get("round", 0))
    max_rounds = int(debate.get("max_rounds", 3))
    pair_stagnation = dict(debate.get("pair_stagnation", {}) or {})
    pair_last_signature = dict(debate.get("pair_last_signature", {}) or {})
    active_prev = debate.get("active_challenge", {}) if isinstance(debate.get("active_challenge"), dict) else {}
    active = contradictions[0]
    active = {
        **active,
        "selected_horizon": selected_horizon,
        "debate_question": (
            "Given the selected horizon, revise your stance or defend it with domain-specific reasoning."
        ),
    }

    # Max rounds guard.
    if round_no >= max_rounds:
        unresolved = [
            {
                "id": c["id"],
                "reason": "Max rounds reached before full convergence.",
                "challenge": c,
            }
            for c in contradictions
        ]
        return {
            "debate": {
                "_replace_lists": ["queue"],
                "queue": contradictions,
                "active_challenge": active,
                "awaiting_response_from": None,
                "next_node": "synthesizer_node",
                "status": "max_rounds_reached",
                "unresolved_contradictions": unresolved,
                "history": history + [{"event": "max_rounds_reached"}],
            }
        }

    # Stagnation check when we've just completed an opponent turn for the same pair.
    awaiting_prev = debate.get("awaiting_response_from")
    if (
        active_prev.get("id") == active["id"]
        and awaiting_prev == active["primary_opponent"]
    ):
        sig = _signature_for_pair(active, signal_snapshots)
        last_sig = pair_last_signature.get(active["id"])
        stagnation = int(pair_stagnation.get(active["id"], 0))
        if sig == last_sig:
            stagnation += 1
        else:
            stagnation = 0
        pair_stagnation[active["id"]] = stagnation
        pair_last_signature[active["id"]] = sig

        if stagnation >= STAGNATION_LIMIT:
            # Mark unresolved but continue with other contradictions if available.
            unresolved_entry = {
                "id": active["id"],
                "reason": f"Stagnation limit reached ({STAGNATION_LIMIT}) without rating change.",
                "challenge": active,
            }
            remaining = [c for c in contradictions if c["id"] != active["id"]]
            if not remaining:
                return {
                    "debate": {
                        "_replace_lists": ["queue"],
                        "queue": [],
                        "active_challenge": active,
                        "awaiting_response_from": None,
                        "next_node": "synthesizer_node",
                        "status": "resolved_with_unresolved_pairs",
                        "unresolved_contradictions": [unresolved_entry],
                        "pair_stagnation": pair_stagnation,
                        "pair_last_signature": pair_last_signature,
                        "history": history + [{"event": "pair_marked_unresolved", "id": active["id"]}],
                    }
                }
            active = remaining[0]
            contradictions = remaining

    # Decide next speaker for current active contradiction.
    awaiting_prev = debate.get("awaiting_response_from")
    if active_prev.get("id") == active["id"] and awaiting_prev == active["target"]:
        next_speaker = active["primary_opponent"]
        turn_type = "opponent_turn"
    else:
        next_speaker = active["target"]
        turn_type = "target_turn"

    round_no += 1
    history.append(
        {
            "event": "debate_turn_dispatched",
            "turn_type": turn_type,
            "speaker": next_speaker,
            "challenge_id": active["id"],
            "round": round_no,
        }
    )

    print("\n---ORCHESTRATOR REVIEW ---")
    print(f"Ticker: {ticker}")
    print(f"Horizon: {horizon_label(selected_horizon)}")
    print(f"Analysts with valid stances: {len(signal_snapshots)}")
    print(
        f"Effective analysts (confidence >= {EFFECTIVE_CONFIDENCE_THRESHOLD:.2f}): "
        f"{len(effective_analysts)}"
    )
    print(f"Contradictions found: {len(contradictions)}")
    print(
        f"Active challenge: {active['id']} "
        f"(target={active['target']}, opponent={active['primary_opponent']}, severity={active['severity']})"
    )
    print(f"Dispatch -> {analyst_to_debate_node_name(next_speaker)}")
    print("------------------------------\n")

    return {
        "debate": {
            "_replace_lists": ["queue"],
            "queue": contradictions,
            "active_challenge": active,
            "awaiting_response_from": next_speaker,
            "next_node": analyst_to_debate_node_name(next_speaker),
            "status": "debating",
            "round": round_no,
            "pair_stagnation": pair_stagnation,
            "pair_last_signature": pair_last_signature,
            "no_contradiction_streak": 0,
            "history": history,
        }
    }
