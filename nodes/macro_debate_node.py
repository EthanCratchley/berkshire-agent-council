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
from shared.stance import Rating, parse_rating, rating_to_score

load_dotenv()

_RATING_ORDER = [
    Rating.STRONG_SELL,
    Rating.SELL,
    Rating.HOLD,
    Rating.BUY,
    Rating.STRONG_BUY,
]

_MACRO_THRESHOLDS = {
    "vix": {"bullish_below": 15, "bearish_above": 25},
    "yield_curve_spread": {"bullish_above": 1.0, "bearish_below": 0.0},
    "unemployment": {"bullish_below": 4.0, "bearish_above": 6.0},
    "fed_funds_rate": {"bullish_below": 3.0, "bearish_above": 5.0},
    "cpi_yoy": {"bullish_below": 2.5, "bearish_above": 4.0},
}

_HORIZON_INDICATOR_WEIGHTS = {
    "short": {
        "vix": 1.5,
        "yield_curve_spread": 1.25,
        "unemployment": 0.75,
        "fed_funds_rate": 1.0,
        "cpi_yoy": 0.75,
    },
    "swing": {
        "vix": 1.0,
        "yield_curve_spread": 1.0,
        "unemployment": 1.0,
        "fed_funds_rate": 1.0,
        "cpi_yoy": 1.0,
    },
    "long": {
        "vix": 0.75,
        "yield_curve_spread": 1.0,
        "unemployment": 1.25,
        "fed_funds_rate": 1.25,
        "cpi_yoy": 1.5,
    },
}


def _deterministic_explanation(features: dict, rating: str) -> str:
    feature_bits = []
    for key in ("vix", "yield_curve_spread", "unemployment", "fed_funds_rate", "cpi_yoy"):
        value = features.get(key)
        if value is None:
            continue
        feature_bits.append(f"{key}={value:.2f}")
    metrics = ", ".join(feature_bits) if feature_bits else "limited macroeconomic metrics available"
    return (
        f"The current macroeconomic stance is {rating}. "
        f"This is based on {metrics}. Lower VIX, lower unemployment, and a positive yield curve "
        f"tend to support stronger ratings, while high inflation and tight monetary policy support weaker ratings."
    )


def _clean_json_text(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0].strip()
    return raw


def _is_string_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _valid_contract(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    required = (
        "explanation",
        "claims_conceded",
        "claims_disputed",
        "weighting_statement",
        "horizon_alignment_note",
        "dialogue_response",
    )
    if any(key not in payload for key in required):
        return False
    if not isinstance(payload.get("explanation"), str):
        return False
    if not _is_string_list(payload.get("claims_conceded")):
        return False
    if not _is_string_list(payload.get("claims_disputed")):
        return False
    if not isinstance(payload.get("weighting_statement"), str):
        return False
    if not isinstance(payload.get("horizon_alignment_note"), str):
        return False
    if not isinstance(payload.get("dialogue_response"), str):
        return False
    return True


def _bounded_rating_update(current: Rating, proposed: Rating, max_step: int = 1) -> Rating:
    current_idx = _RATING_ORDER.index(current)
    proposed_idx = _RATING_ORDER.index(proposed)
    delta = proposed_idx - current_idx
    if abs(delta) <= max_step:
        return proposed
    bounded_idx = current_idx + max_step if delta > 0 else current_idx - max_step
    bounded_idx = max(0, min(bounded_idx, len(_RATING_ORDER) - 1))
    return _RATING_ORDER[bounded_idx]


def _safe_dialogue_response(opponent_analyst: str, horizon: str, claims_disputed: list, claims_conceded: list) -> str:
    disputed_text = ", ".join(claims_disputed[:2]) if claims_disputed else "the opponent's main claim"
    conceded_text = ", ".join(claims_conceded[:2]) if claims_conceded else "limited points"
    return (
        f"I acknowledge the {opponent_analyst} concerns, but for {horizon} the macro backdrop still favors my stance. "
        f"I concede {conceded_text}, while disputing {disputed_text}."
    )


def _trim_text(value: str, limit: int = 140) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _build_guarded_debate_response(
    opponent_analyst: str,
    opponent_rating: str,
    opponent_details: str,
    claims_conceded: list,
    claims_disputed: list,
    horizon_text: str,
    final_rating: str,
) -> str:
    analyst = opponent_analyst or "opponent"
    rating = opponent_rating or "hold"
    details = _trim_text(opponent_details, 150) or "their domain-specific evidence"
    conceded = ", ".join(claims_conceded[:2]) if claims_conceded else "limited points"
    disputed = ", ".join(claims_disputed[:2]) if claims_disputed else "the main opposing thesis"
    return (
        f"Opponent ({analyst}) argues {rating} and emphasizes: {details}. "
        f"I concede {conceded} but dispute {disputed}. "
        f"For {horizon_text}, I maintain {final_rating} based on macroeconomic evidence."
    )


def _looks_natural(response: str) -> bool:
    text = " ".join(str(response or "").split()).strip()
    if len(text) < 40:
        return False
    lowered = text.lower()
    if any(phrase in lowered for phrase in ("fully agree", "perfectly mirrors", "same assessment")):
        return False
    if lowered.startswith("opponent ("):
        return False
    return True


def _resolve_opponent_analyst(active: dict, opponent_case: dict, current_analyst: str) -> str:
    current = str(current_analyst or "").strip().lower()
    candidates = [active.get("primary_opponent"), active.get("opponent"), opponent_case.get("analyst")]
    pair = active.get("pair", [])
    if isinstance(pair, list):
        for entry in pair:
            if isinstance(entry, str):
                candidates.append(entry)

    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        cleaned = candidate.strip().lower()
        if cleaned and cleaned != current:
            return cleaned
    return "unknown"


def _clamp_confidence(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(numeric, 1.0))


def _score_macro_indicator(name: str, value) -> int:
    if value is None:
        return 0
    thresholds = _MACRO_THRESHOLDS.get(name, {})

    if name in ("vix", "unemployment", "fed_funds_rate", "cpi_yoy"):
        bullish_below = thresholds.get("bullish_below")
        bearish_above = thresholds.get("bearish_above")
        if bullish_below is not None and value < bullish_below:
            return 1
        if bearish_above is not None and value > bearish_above:
            return -1
        return 0

    if name == "yield_curve_spread":
        bullish_above = thresholds.get("bullish_above")
        bearish_below = thresholds.get("bearish_below")
        if bullish_above is not None and value > bullish_above:
            return 1
        if bearish_below is not None and value < bearish_below:
            return -1
        return 0
    return 0


def _macro_feature_strength(features: dict, selected_horizon: str) -> float:
    weights = _HORIZON_INDICATOR_WEIGHTS.get(selected_horizon, _HORIZON_INDICATOR_WEIGHTS["swing"])
    weighted_sum = 0.0
    total_weight = 0.0
    for key in ("vix", "yield_curve_spread", "unemployment", "fed_funds_rate", "cpi_yoy"):
        value = features.get(key)
        if value is None:
            continue
        score = _score_macro_indicator(key, value)
        weight = float(weights.get(key, 1.0))
        weighted_sum += score * weight
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return max(-1.0, min(weighted_sum / total_weight, 1.0))


def _deterministic_policy_decision(
    selected_horizon: str,
    current_rating: Rating,
    current_confidence: float,
    features: dict,
    opponent_analyst: str,
    opponent_rating: Rating,
    opponent_confidence: float,
    contradiction_severity: float,
) -> dict:
    analyst_weights = analyst_weights_for_horizon(selected_horizon)
    own_relevance = float(analyst_weights.get("macro", 1.0))
    opponent_relevance = float(analyst_weights.get(opponent_analyst, 1.0))

    own_feature_strength = abs(_macro_feature_strength(features, selected_horizon))
    own_base_strength = (0.6 * _clamp_confidence(current_confidence)) + (0.4 * own_feature_strength)
    own_effective = own_base_strength * own_relevance

    severity_factor = max(0.4, min(1.0, contradiction_severity / 2.0))
    opponent_effective = _clamp_confidence(opponent_confidence) * opponent_relevance * severity_factor
    margin = own_effective - opponent_effective

    action = "defend"
    revised_rating = current_rating
    revised_confidence = _clamp_confidence(current_confidence)

    if margin < -0.20:
        action = "concede_one_step"
        revised_rating = _bounded_rating_update(current_rating, opponent_rating, max_step=1)
        revised_confidence = _clamp_confidence(
            (0.75 * _clamp_confidence(current_confidence))
            + (0.25 * _clamp_confidence(opponent_confidence))
            - min(0.10, contradiction_severity * 0.03)
        )
    elif margin < -0.05:
        action = "defend_reduce_confidence"
        revised_confidence = _clamp_confidence(
            _clamp_confidence(current_confidence) - min(0.12, 0.04 + (contradiction_severity * 0.02))
        )
    elif margin > 0.20:
        action = "defend_increase_confidence"
        revised_confidence = _clamp_confidence(_clamp_confidence(current_confidence) + 0.03)

    return {
        "action": action,
        "margin": round(margin, 3),
        "own_effective_strength": round(own_effective, 3),
        "opponent_effective_strength": round(opponent_effective, 3),
        "own_relevance": round(own_relevance, 3),
        "opponent_relevance": round(opponent_relevance, 3),
        "contradiction_severity": round(contradiction_severity, 3),
        "final_position": {
            "rating": revised_rating.value,
            "confidence": round(revised_confidence, 2),
        },
    }


def macro_debate_node(state: BerkshireState):
    """
    Deterministic policy-driven macro debate node.

    Rating/confidence decisions are made before LLM generation. The LLM only
    provides explanation and rebuttal language.
    """
    ticker = state.get("ticker", "UNKNOWN")
    selected_horizon = normalize_horizon(state.get("horizon", "swing"))
    signals = state.get("analyst_signals", {})
    debate = state.get("debate", {})
    active = debate.get("active_challenge", {}) if isinstance(debate.get("active_challenge"), dict) else {}
    awaiting = debate.get("awaiting_response_from")

    current = signals.get("macro", {}) if isinstance(signals.get("macro"), dict) else {}
    current_rating = parse_rating(current.get("rating")) or Rating.HOLD
    current_confidence = _clamp_confidence(current.get("confidence", 0.0))
    features = current.get("features", {})
    current_details = current.get("details", "")

    # Fallback explanation without LLM.
    explanation = _deterministic_explanation(features, current_rating.value)
    revised_rating = current_rating
    revised_confidence = current_confidence
    debate_response = "Maintaining stance from quantitative macroeconomic indicators."
    claims_conceded = []
    claims_disputed = []
    weighting_statement = ""
    horizon_alignment_note = f"Macroeconomic view aligned to {horizon_label(selected_horizon)}."
    dialogue_response = ""

    # Only include challenge context if this call is for a debate turn.
    challenge_context = ""
    opponent_analyst = "unknown"
    opponent_rating = Rating.HOLD
    opponent_confidence = 0.0
    opponent_details = ""
    contradiction_severity = 0.0
    if awaiting == "macro" and active:
        coalition = active.get("coalition", {})
        my_case = active.get("my_case", {})
        opponent_case = active.get("opponent_case", {})
        opponent_analyst = _resolve_opponent_analyst(active, opponent_case, "macro")
        opponent_rating = parse_rating(opponent_case.get("rating")) or Rating.HOLD
        opponent_confidence = _clamp_confidence(opponent_case.get("confidence", 0.0))
        opponent_details = str(opponent_case.get("details", "") or "")
        try:
            contradiction_severity = float(active.get("severity", 0.0))
        except (TypeError, ValueError):
            contradiction_severity = 0.0
        opponent_supporters = coalition.get("supporters_of_opponent", [])
        partial_supporters = coalition.get("partial", [])
        supporter_lines = []
        for entry in opponent_supporters[:2]:
            supporter_lines.append(
                f"{entry.get('analyst', 'unknown')}: rating={entry.get('rating', '')}, details={entry.get('details', '')}"
            )
        partial_lines = []
        for entry in partial_supporters[:2]:
            partial_lines.append(
                f"{entry.get('analyst', 'unknown')}: rating={entry.get('rating', '')}, details={entry.get('details', '')}"
            )
        challenge_context = (
            f"\nDEBATE CONTEXT:\n"
            f"- Selected horizon: {horizon_label(selected_horizon)}\n"
            f"- Challenge ID: {active.get('id', '')}\n"
            f"- Action requested: {active.get('action', 'revise_or_defend')}\n"
            f"- Contradiction reason: {active.get('reason', '')}\n"
            f"- Your current case: rating={my_case.get('rating', '')}, confidence={my_case.get('confidence', '')}, details={my_case.get('details', '')}\n"
            f"- Opponent case: rating={opponent_case.get('rating', '')}, confidence={opponent_case.get('confidence', '')}, details={opponent_case.get('details', '')}\n"
            f"- Opponent analyst type: {opponent_analyst}\n"
            f"- Opponent last rebuttal: {opponent_case.get('last_debate_response', '')}\n"
            f"- Opponent supporters: {len(coalition.get('supporters_of_opponent', []))}\n"
            f"- Partial-agreement analysts: {len(coalition.get('partial', []))}\n"
            f"- Supporting analyst arguments: {supporter_lines}\n"
            f"- Partial-agreement analyst arguments: {partial_lines}\n"
        )

    decision_packet = _deterministic_policy_decision(
        selected_horizon,
        current_rating,
        current_confidence,
        features if isinstance(features, dict) else {},
        opponent_analyst,
        opponent_rating,
        opponent_confidence,
        contradiction_severity,
    )
    revised_rating = parse_rating(decision_packet["final_position"]["rating"]) or current_rating
    revised_confidence = _clamp_confidence(decision_packet["final_position"]["confidence"])

    prompt = f"""You are a macroeconomic analyst.
Ticker: {ticker}
Selected horizon: {horizon_label(selected_horizon)}
Current rating: {current_rating.value}
Current confidence: {current_confidence}
Current details: {current_details}
Features: {features}
{challenge_context}

DETERMINISTIC DECISION (must honor exactly):
- action: {decision_packet['action']}
- final_position.rating: {revised_rating.value}
- final_position.confidence: {revised_confidence}
- policy_margin: {decision_packet['margin']}

Task:
1) Explain in plain English what the macroeconomic indicators imply.
2) Explicitly address the opponent claims and rebut or concede.
3) Keep your reasoning grounded in the provided macroeconomic evidence for the selected horizon.
4) Do not invent facts beyond provided features/context.
5) IMPORTANT: the opponent is {opponent_analyst}. Do not describe the opponent argument as "macroeconomic".
6) Do not say you "fully agree", "perfectly mirror", or "fully align" with the opponent. Only concede specific points if warranted.
7) You are not deciding rating/confidence. These are fixed by policy; explain and defend them.

Return ONLY valid JSON:
{{"explanation":"<3-6 concise sentences>", "claims_conceded":["<opponent claim accepted>"], "claims_disputed":["<opponent claim disputed>"], "weighting_statement":"<which factor dominated and why>", "horizon_alignment_note":"<why this macroeconomic stance fits selected horizon>", "dialogue_response":"<1-2 natural-language sentences responding directly to opponent>"}}
"""

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        is_debate_turn = awaiting == "macro" and bool(active)
        if ChatGoogleGenerativeAI is not None and api_key and is_debate_turn:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
            )
            response = llm.invoke(prompt)
            raw = _clean_json_text(response.content)
            parsed = json.loads(raw)

            if not _valid_contract(parsed):
                repair_prompt = (
                    "Return ONLY valid JSON for this schema:\n"
                    '{"explanation":"...", "claims_conceded":["..."], "claims_disputed":["..."], '
                    '"weighting_statement":"...", "horizon_alignment_note":"...", "dialogue_response":"..."}\n'
                    f"Original response:\n{raw}"
                )
                repair = llm.invoke(repair_prompt)
                repaired_raw = _clean_json_text(repair.content)
                parsed = json.loads(repaired_raw)

            if _valid_contract(parsed):
                explanation = str(parsed.get("explanation", "")).strip() or explanation
                claims_conceded = parsed.get("claims_conceded", [])
                claims_disputed = parsed.get("claims_disputed", [])
                weighting_statement = str(parsed.get("weighting_statement", "")).strip()
                horizon_alignment_note = str(parsed.get("horizon_alignment_note", "")).strip()
                dialogue_response = str(parsed.get("dialogue_response", "")).strip()
                debate_response = _build_guarded_debate_response(
                    opponent_analyst,
                    opponent_rating.value,
                    opponent_details,
                    claims_conceded,
                    claims_disputed,
                    horizon_label(selected_horizon),
                    revised_rating.value,
                )
                lowered_response = dialogue_response.lower()
                agreement_phrases = (
                    "fully agree",
                    "perfectly mirrors",
                    "fully align",
                    "fully aligned",
                    "entirely agree",
                    "same assessment",
                )
                if any(phrase in lowered_response for phrase in agreement_phrases):
                    debate_response = _build_guarded_debate_response(
                        opponent_analyst,
                        opponent_rating.value,
                        opponent_details,
                        claims_disputed,
                        claims_conceded,
                        horizon_label(selected_horizon),
                        revised_rating.value,
                    )
                elif opponent_analyst != "macro" and (
                    "macroeconomic" in lowered_response and "agree" in lowered_response
                ):
                    debate_response = _build_guarded_debate_response(
                        opponent_analyst,
                        opponent_rating.value,
                        opponent_details,
                        claims_disputed,
                        claims_conceded,
                        horizon_label(selected_horizon),
                        revised_rating.value,
                    )
    except Exception as e:
        explanation = f"{explanation} LLM refinement unavailable."
        debate_response = _build_guarded_debate_response(
            opponent_analyst,
            opponent_rating.value,
            opponent_details,
            claims_conceded,
            claims_disputed,
            horizon_label(selected_horizon),
            revised_rating.value,
        )
        weighting_statement = ""

    # Guardrail: avoid incoherent phrasing that treats non-macro opponent as macroeconomic.
    lowered = debate_response.lower()
    if opponent_analyst != "macro" and "macroeconomic case you've presented" in lowered:
        debate_response = (
            f"I acknowledge the {opponent_analyst} concerns, but for "
            f"{horizon_label(selected_horizon)} the economic environment still supports my stance."
        )
    if any(phrase in lowered for phrase in ("fully agree", "perfectly mirrors", "same assessment")):
        debate_response = _build_guarded_debate_response(
            opponent_analyst,
            opponent_rating.value,
            opponent_details,
            claims_disputed,
            claims_conceded,
            horizon_label(selected_horizon),
            revised_rating.value,
        )

    print(
        f"\n[Macroeconomic Debate] {ticker}: rating {current_rating.value} -> {revised_rating.value}"
    )
    print(f"[Macroeconomic Debate] Explanation: {explanation}")
    position_changed = revised_rating.value != current_rating.value

    return {
        "analyst_signals": {
            "macro": {
                "rating": revised_rating.value,
                "confidence": revised_confidence,
                "features": features,
                "details": f"{current_details} | Macroeconomic explanation: {explanation}".strip(),
                "narrative": explanation,
                "debate_response": debate_response,
                "position_changed": position_changed,
                "counterpoints_addressed": claims_disputed,
                "claims_conceded": claims_conceded,
                "claims_disputed": claims_disputed,
                "final_position": {
                    "rating": revised_rating.value,
                    "confidence": revised_confidence,
                },
                "deterministic_decision": decision_packet,
                "weighting_statement": weighting_statement,
                "horizon_alignment_note": horizon_alignment_note,
            }
        }
    }
