import json
import os

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

from shared.state_schema import BerkshireState
from shared.horizon import normalize_horizon, horizon_label, analyst_weights_for_horizon
from shared.stance import parse_rating, rating_to_score

load_dotenv()

EFFECTIVE_CONTRIBUTOR_CONFIDENCE = 0.30
MIN_EFFECTIVE_CONTRIBUTORS = 3
MIN_EFFECTIVE_FOR_STRONG = 2
MIN_EFFECTIVE_FOR_DIRECTIONAL = 2
HIGH_UNRESOLVED_SEVERITY = 1.5
VERY_HIGH_UNRESOLVED_SEVERITY = 2.5


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


def _apply_coverage_gate(final_rating: str, effective_contributors: int) -> str:
    """
    Prevent extreme recommendations when effective analyst coverage is too low.
    """
    if effective_contributors >= MIN_EFFECTIVE_FOR_STRONG:
        return final_rating
    if final_rating == "strong_buy":
        return "buy"
    if final_rating == "strong_sell":
        return "sell"
    return final_rating


def _downgrade_one_step(rating: str) -> str:
    if rating == "strong_buy":
        return "buy"
    if rating == "buy":
        return "hold"
    if rating == "strong_sell":
        return "sell"
    if rating == "sell":
        return "hold"
    return rating


def _extract_unresolved_severity(unresolved: list) -> float:
    severities = []
    for entry in unresolved or []:
        if not isinstance(entry, dict):
            continue
        challenge = entry.get("challenge", {}) if isinstance(entry.get("challenge"), dict) else {}
        sev = challenge.get("severity")
        try:
            severities.append(float(sev))
        except (TypeError, ValueError):
            continue
    return max(severities) if severities else 0.0


def _clean_json_text(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0].strip()
    return raw


def synthesizer_node(state: BerkshireState):
    """
    Final synthesis node.

    Produces one recommendation and explains whether there are unresolved
    analyst contradictions.
    """
    ticker = state.get("ticker", "UNKNOWN")
    selected_horizon = normalize_horizon(state.get("horizon", "swing"))
    selected_horizon_label = horizon_label(selected_horizon)
    analyst_weights = analyst_weights_for_horizon(selected_horizon)
    signals = state.get("analyst_signals", {})
    debate = state.get("debate", {})

    # Extract classical model predictions before analyst loop.
    classical = signals.get("classical_models", {})
    classical_rf = classical.get("rf") if isinstance(classical, dict) else None
    classical_knn = classical.get("knn") if isinstance(classical, dict) else None

    weighted_sum = 0.0
    total_weight = 0.0
    analyst_breakdown = []
    effective_contributors = 0

    for analyst, payload in signals.items():
        if analyst == "classical_models":
            continue
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
        analyst_weight = float(analyst_weights.get(analyst, 1.0))
        weighted_confidence = confidence * analyst_weight
        weighted_sum += score * weighted_confidence
        total_weight += weighted_confidence
        if confidence >= EFFECTIVE_CONTRIBUTOR_CONFIDENCE:
            effective_contributors += 1
        analyst_breakdown.append(
            {
                "analyst": analyst,
                "rating": rating.value,
                "confidence": confidence,
                "analyst_weight": analyst_weight,
                "weighted_confidence": round(weighted_confidence, 3),
                "score": score,
                "details": payload.get("details", ""),
                "horizon_alignment_note": payload.get("horizon_alignment_note", ""),
            }
        )

    weighted_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
    raw_final_rating = _rating_from_weighted_score(weighted_score)
    final_rating = _apply_coverage_gate(raw_final_rating, effective_contributors)
    coverage_gate_applied = final_rating != raw_final_rating
    coverage_gated_rating = final_rating
    unresolved = debate.get("unresolved_contradictions", [])
    highest_unresolved_severity = _extract_unresolved_severity(unresolved)
    reliability_notes = []

    if (
        effective_contributors < MIN_EFFECTIVE_FOR_DIRECTIONAL
        and final_rating in {"buy", "sell", "strong_buy", "strong_sell"}
    ):
        prior = final_rating
        final_rating = _downgrade_one_step(final_rating)
        reliability_notes.append(
            f"Directional call softened from {prior} to {final_rating} because effective contributor count "
            f"is below {MIN_EFFECTIVE_FOR_DIRECTIONAL}."
        )

    if highest_unresolved_severity >= HIGH_UNRESOLVED_SEVERITY:
        prior = final_rating
        final_rating = _downgrade_one_step(final_rating)
        if final_rating != prior:
            reliability_notes.append(
                f"Unresolved contradiction severity ({highest_unresolved_severity:.2f}) reduced conviction "
                f"from {prior} to {final_rating}."
            )
    if (
        highest_unresolved_severity >= VERY_HIGH_UNRESOLVED_SEVERITY
        and final_rating in {"buy", "sell"}
        and abs(weighted_score) < 1.1
    ):
        reliability_notes.append(
            "Severe unresolved disagreement plus moderate weighted score forced a neutral hold recommendation."
        )
        final_rating = "hold"

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
                f"{str(turn.get('speaker', '')).title()}: {turn.get('debate_response', '').strip()} "
                f"(final: {turn.get('rating')}, conf={turn.get('confidence')})"
            )
            conversation_lines.append(line)

    if unresolved:
        rationale = (
            f"Final recommendation is {final_rating} with unresolved analyst disagreements "
            f"({len(unresolved)} contradiction(s)) for {selected_horizon_label}."
        )
    else:
        rationale = (
            f"Final recommendation is {final_rating} with no unresolved contradictions "
            f"for {selected_horizon_label}."
        )

    coverage_warning = ""
    if effective_contributors < MIN_EFFECTIVE_CONTRIBUTORS:
        coverage_warning = (
            f"Coverage warning: only {effective_contributors} analyst(s) had confidence >= "
            f"{EFFECTIVE_CONTRIBUTOR_CONFIDENCE:.2f}; recommendation reliability may be reduced."
        )
        rationale = f"{rationale} {coverage_warning}"
    if coverage_gate_applied:
        rationale = (
            f"{rationale} Rating was capped from {raw_final_rating} to {coverage_gated_rating} due to "
            f"low effective coverage (< {MIN_EFFECTIVE_FOR_STRONG} analysts with confidence >= "
            f"{EFFECTIVE_CONTRIBUTOR_CONFIDENCE:.2f})."
        )
    if reliability_notes:
        rationale = f"{rationale} {' '.join(reliability_notes)}"

    ranked_contributors = sorted(
        analyst_breakdown,
        key=lambda b: (
            abs(float(b.get("score", 0.0)) * float(b.get("weighted_confidence", 0.0))),
            float(b.get("weighted_confidence", 0.0)),
        ),
        reverse=True,
    )
    top_contributors = ", ".join([b["analyst"] for b in ranked_contributors[:2]]) if ranked_contributors else "none"
    section_1_debate = (
        f"The analysts debated under {selected_horizon_label}. "
        f"{'At least one contradiction remained unresolved.' if unresolved else 'The debate converged without unresolved contradictions.'}"
    )
    section_2_decision = (
        f"The final rating is {final_rating} with weighted score {weighted_score:.3f}. "
        f"The strongest contributors were {top_contributors}."
    )
    section_3_risks = (
        f"{coverage_warning if coverage_warning else 'No major coverage warning was triggered.'} "
        f"{'Because contradictions remain, uncertainty is elevated for this horizon.' if unresolved else 'Residual uncertainty is present but comparatively lower for this horizon.'}"
    )
    detailed_narrative = (
        f"Debate Summary: {section_1_debate}\n"
        f"Decision Logic: {section_2_decision}\n"
        f"Risks & Uncertainty: {section_3_risks}"
    )
    narrative_key_drivers = []
    narrative_uncertainties = []

    try:
        if ChatGoogleGenerativeAI is not None:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            prompt = (
                "You are the final portfolio synthesis analyst. "
                "Explain this stock recommendation in clear natural language for a non-expert user.\n\n"
                f"Ticker: {ticker}\n"
                f"Horizon: {selected_horizon_label}\n"
                f"Final recommendation: {final_rating}\n"
                f"Weighted score: {weighted_score:.3f}\n"
                f"Coverage warning: {coverage_warning or 'none'}\n"
                f"Unresolved contradictions: {len(unresolved)}\n"
                f"Analyst breakdown: {analyst_breakdown}\n"
                f"Debate conversation: {conversation_lines}\n\n"
                "Return ONLY valid JSON with keys:\n"
                '{"section_1_debate":"<2-4 sentences on key debate outcomes>",'
                '"section_2_decision":"<2-4 sentences on why final recommendation was selected>",'
                '"section_3_risks":"<2-4 sentences on uncertainty and risks for this horizon>",'
                '"key_drivers":["<driver 1>","<driver 2>"],'
                '"uncertainties":["<uncertainty 1>","<uncertainty 2>"]}'
            )
            raw = _clean_json_text(llm.invoke(prompt).content)
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                llm_section_1 = parsed.get("section_1_debate", "")
                llm_section_2 = parsed.get("section_2_decision", "")
                llm_section_3 = parsed.get("section_3_risks", "")
                key_drivers = parsed.get("key_drivers", [])
                uncertainties = parsed.get("uncertainties", [])
                if all(
                    isinstance(part, str) and part.strip()
                    for part in (llm_section_1, llm_section_2, llm_section_3)
                ):
                    section_1_debate = llm_section_1.strip()
                    section_2_decision = llm_section_2.strip()
                    section_3_risks = llm_section_3.strip()
                    detailed_narrative = (
                        f"Debate Summary: {section_1_debate}\n"
                        f"Decision Logic: {section_2_decision}\n"
                        f"Risks & Uncertainty: {section_3_risks}"
                    )
                if isinstance(key_drivers, list):
                    narrative_key_drivers = [str(item).strip() for item in key_drivers if str(item).strip()]
                if isinstance(uncertainties, list):
                    narrative_uncertainties = [str(item).strip() for item in uncertainties if str(item).strip()]
    except Exception:
        # Keep deterministic fallback narrative.
        pass

    print("\n---FINAL SYNTHESIS ---")
    print(f"Ticker: {ticker}")
    print(f"Horizon: {selected_horizon_label}")
    print(f"LLM Debate Recommendation: {final_rating}")
    if classical_rf:
        print(f"Random Forest Prediction:  {classical_rf.get('prediction', 'N/A')}")
    if classical_knn:
        print(f"KNN Prediction:            {classical_knn.get('prediction', 'N/A')}")
    if coverage_gate_applied:
        print(
            f"Coverage gate applied: {raw_final_rating} -> {coverage_gated_rating} "
            f"(effective contributors: {effective_contributors})"
        )
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
            "horizon": selected_horizon,
            "horizon_label": selected_horizon_label,
            "horizon_weights": analyst_weights,
            "recommendation": final_rating,
            "raw_recommendation": raw_final_rating,
            "weighted_score": round(weighted_score, 3),
            "effective_contributors": effective_contributors,
            "effective_contributor_threshold": EFFECTIVE_CONTRIBUTOR_CONFIDENCE,
            "coverage_warning": coverage_warning,
            "reliability_notes": reliability_notes,
            "analyst_breakdown": analyst_breakdown,
            "debate_transcript": transcript,
            "debate_conversation": conversation_lines,
            "debate_conversation_text": "\n".join(conversation_lines),
            "unresolved_contradictions": unresolved,
            "highest_unresolved_severity": round(highest_unresolved_severity, 3),
            "rationale": rationale,
            "detailed_narrative": detailed_narrative,
            "summary_sections": {
                "debate_summary": section_1_debate,
                "decision_logic": section_2_decision,
                "risks_uncertainty": section_3_risks,
            },
            "narrative_key_drivers": narrative_key_drivers,
            "narrative_uncertainties": narrative_uncertainties,
            "classical_models": {
                "rf": classical_rf,
                "knn": classical_knn,
            },
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
