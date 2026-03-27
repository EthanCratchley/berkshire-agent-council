from shared.state_schema import BerkshireState
from shared.stance import parse_rating, rating_to_score


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

    if unresolved:
        rationale = (
            f"Final recommendation is {final_rating} with unresolved analyst disagreements "
            f"({len(unresolved)} contradiction(s))."
        )
    else:
        rationale = f"Final recommendation is {final_rating} with no unresolved contradictions."

    print("\n---FINAL SYNTHESIS ---")
    print(f"Ticker: {ticker}")
    print(f"Recommendation: {final_rating}")
    print(f"Weighted score: {weighted_score:.3f}")
    print(f"Unresolved contradictions: {len(unresolved)}")
    print("-----------------------\n")

    return {
        "final_report": {
            "ticker": ticker,
            "recommendation": final_rating,
            "weighted_score": round(weighted_score, 3),
            "analyst_breakdown": analyst_breakdown,
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

