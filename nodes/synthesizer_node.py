from shared.state_schema import BerkshireState
from shared.stance import parse_rating, rating_to_score

EFFECTIVE_CONTRIBUTOR_CONFIDENCE = 0.30
MIN_EFFECTIVE_CONTRIBUTORS = 3


def _rating_from_weighted_score(weighted_score: float) -> str:
    if weighted_score >= 1.25:
        return "strong_buy"
    if weighted_score >= 0.25:
        return "buy"
    if weighted_score <= -1.25:
        return "strong_sell"
    if weighted_score <= -0.25:
        return "sell"
    return "hold"


def synthesizer_node(state: BerkshireState):
    """
    Final synthesis node.

    Produces one recommendation and explains whether there are unresolved
    analyst contradictions.
    """
    ticker = state.get("ticker", "UNKNOWN")
    signals = state.get("analyst_signals", {})
    debate = state.get("debate", {})

    weighted_sum = 0.0
    total_weight = 0.0
    analyst_breakdown = []
    effective_contributors = 0

    for analyst, payload in signals.items():
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
        confidence = max(0.0, min(confidence, 1.0))
        score = rating_to_score(rating)
        weighted_sum += score * confidence
        total_weight += confidence
        if confidence >= EFFECTIVE_CONTRIBUTOR_CONFIDENCE:
            effective_contributors += 1
        analyst_breakdown.append(
            {
                "analyst": analyst,
                "rating": rating.value,
                "confidence": confidence,
                "score": score,
            }
        )

    weighted_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
    final_rating = _rating_from_weighted_score(weighted_score)
    unresolved = debate.get("unresolved_contradictions", [])
    transcript = []
    conversation_lines = []
    for event in debate.get("history", []):
        if isinstance(event, dict) and event.get("event") == "debater_turn_result":
            turn = {
                "speaker": event.get("speaker"),
                "rating": event.get("rating"),
                "confidence": event.get("confidence"),
                "debate_response": event.get("debate_response", ""),
                "position_changed": event.get("position_changed", False),
                "counterpoints_addressed": event.get("counterpoints_addressed", []),
                "claims_conceded": event.get("claims_conceded", []),
                "claims_disputed": event.get("claims_disputed", []),
                "final_position": event.get("final_position", {}),
                "weighting_statement": event.get("weighting_statement", ""),
                "conversation_line": event.get("conversation_line", ""),
            }
            transcript.append(turn)
            line = turn.get("conversation_line") or (
                f"{str(turn.get('speaker', '')).title()}: conceded {turn.get('claims_conceded', [])}, "
                f"disputed {turn.get('claims_disputed', [])}, "
                f"weighting {turn.get('weighting_statement', '')}, "
                f"final_position {turn.get('rating')} ({turn.get('confidence')})"
            )
            conversation_lines.append(line)

    if unresolved:
        rationale = (
            f"Final recommendation is {final_rating} with unresolved analyst disagreements "
            f"({len(unresolved)} contradiction(s))."
        )
    else:
        rationale = f"Final recommendation is {final_rating} with no unresolved contradictions."

    coverage_warning = ""
    if effective_contributors < MIN_EFFECTIVE_CONTRIBUTORS:
        coverage_warning = (
            f"Coverage warning: only {effective_contributors} analyst(s) had confidence >= "
            f"{EFFECTIVE_CONTRIBUTOR_CONFIDENCE:.2f}; recommendation reliability may be reduced."
        )
        rationale = f"{rationale} {coverage_warning}"

    print("\n---FINAL SYNTHESIS ---")
    print(f"Ticker: {ticker}")
    print(f"Recommendation: {final_rating}")
    print(f"Weighted score: {weighted_score:.3f}")
    print(
        f"Effective contributors (confidence >= {EFFECTIVE_CONTRIBUTOR_CONFIDENCE:.2f}): "
        f"{effective_contributors}"
    )
    if coverage_warning:
        print(coverage_warning)
    print(f"Unresolved contradictions: {len(unresolved)}")
    print("-----------------------\n")

    return {
        "final_report": {
            "ticker": ticker,
            "recommendation": final_rating,
            "weighted_score": round(weighted_score, 3),
            "effective_contributors": effective_contributors,
            "effective_contributor_threshold": EFFECTIVE_CONTRIBUTOR_CONFIDENCE,
            "coverage_warning": coverage_warning,
            "analyst_breakdown": analyst_breakdown,
            "debate_transcript": transcript,
            "debate_conversation": conversation_lines,
            "debate_conversation_text": "\n".join(conversation_lines),
            "unresolved_contradictions": unresolved,
            "rationale": rationale,
        },
        "debate": {
            "status": "completed",
            "next_node": "end",
            "history": [
                {
                    "event": "synthesis_complete",
                    "recommendation": final_rating,
                    "weighted_score": round(weighted_score, 3),
                    "unresolved_count": len(unresolved),
                }
            ],
        },
    }
