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
from shared.stance import Rating, parse_rating

load_dotenv()

_RATING_ORDER = [
    Rating.STRONG_SELL,
    Rating.SELL,
    Rating.HOLD,
    Rating.BUY,
    Rating.STRONG_BUY,
]

_INDICATOR_THRESHOLDS = {
    "rsi": {"bullish_below": 35, "bearish_above": 65},
    "macd_histogram": {"bullish_above": 0.0, "bearish_below": 0.0},
    "sma_20_50_cross": {"bullish_eq": 1, "bearish_eq": 0},
    "bollinger_pct": {"bullish_below": 0.2, "bearish_above": 0.8},
    "volume_ratio": {"bullish_above": 1.5, "bearish_below": 0.7},
    "price_change_5d": {"bullish_above": 0.02, "bearish_below": -0.02},
    "price_change_20d": {"bullish_above": 0.05, "bearish_below": -0.05},
}

_HORIZON_INDICATOR_WEIGHTS = {
    "short": {
        "rsi": 1.5,
        "macd_histogram": 1.25,
        "sma_20_50_cross": 0.75,
        "bollinger_pct": 1.25,
        "volume_ratio": 1.25,
        "price_change_5d": 1.5,
        "price_change_20d": 0.75,
    },
    "swing": {
        "rsi": 1.0,
        "macd_histogram": 1.0,
        "sma_20_50_cross": 1.0,
        "bollinger_pct": 1.0,
        "volume_ratio": 1.0,
        "price_change_5d": 1.0,
        "price_change_20d": 1.0,
    },
    "long": {
        "rsi": 0.75,
        "macd_histogram": 0.75,
        "sma_20_50_cross": 1.5,
        "bollinger_pct": 0.75,
        "volume_ratio": 0.75,
        "price_change_5d": 0.75,
        "price_change_20d": 1.5,
    },
}


def _deterministic_explanation(features: dict, rating: str) -> str:
    feature_bits = []
    for key in ("rsi", "macd_histogram", "sma_20_50_cross", "bollinger_pct", "volume_ratio", "price_change_5d", "price_change_20d"):
        value = features.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            feature_bits.append(f"{key}={value:.2f}")
        else:
            feature_bits.append(f"{key}={value}")
    metrics = ", ".join(feature_bits) if feature_bits else "limited technical metrics available"
    return (
        f"The current technical stance is {rating}. "
        f"This is based on {metrics}. Bullish momentum, positive trend changes, "
        f"and strong volume growth support stronger ratings, while bearish cross-overs and overbought indicators support weaker ones."
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


def _clamp_confidence(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(numeric, 1.0))


def _score_indicator(name: str, value, trend_direction: float = 0.0) -> int:
    if value is None:
        return 0
    thresholds = _INDICATOR_THRESHOLDS.get(name, {})

    if name == "sma_20_50_cross":
        if value == thresholds.get("bullish_eq"):
            return 1
        return -1

    if name == "rsi":
        if value < thresholds.get("bullish_below", 35):
            return 1
        if value > thresholds.get("bearish_above", 65):
            return -1
        return 0

    if name == "bollinger_pct":
        if value < thresholds.get("bullish_below", 0.2):
            return 1
        if value > thresholds.get("bearish_above", 0.8):
            return -1
        return 0

    bullish_above = thresholds.get("bullish_above")
    bearish_below = thresholds.get("bearish_below")

    if name == "volume_ratio":
        if bullish_above is not None and value > bullish_above:
            if trend_direction is None:
                return 0
            return 1 if trend_direction >= 0 else -1
        return 0

    if bullish_above is not None and value > bullish_above:
        return 1
    if bearish_below is not None and value < bearish_below:
        return -1
    return 0


def _technical_feature_strength(features: dict, selected_horizon: str) -> float:
    weights = _HORIZON_INDICATOR_WEIGHTS.get(selected_horizon, _HORIZON_INDICATOR_WEIGHTS["swing"])
    trend_direction = features.get("price_change_20d")

    weighted_sum = 0.0
    total_weight = 0.0
    for key in _INDICATOR_THRESHOLDS:
        value = features.get(key)
        if value is None:
            continue
        score = _score_indicator(key, value, trend_direction)
        weight = float(weights.get(key, 1.0))
        weighted_sum += score * weight
        total_weight += weight

    if total_weight <= 0:
        return 0.0
    return max(-1.0, min(weighted_sum / total_weight, 1.0))


def _safe_dialogue_response(opponent_analyst: str, horizon: str, claims_disputed: list, claims_conceded: list) -> str:
    disputed_text = ", ".join(claims_disputed[:2]) if claims_disputed else "the opponent's main claim"
    conceded_text = ", ".join(claims_conceded[:2]) if claims_conceded else "limited points"
    return (
        f"I acknowledge the {opponent_analyst} concerns, but for {horizon} the technical setup still supports my stance. "
        f"I concede {conceded_text}, while disputing {disputed_text}."
    )


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
    own_relevance = float(analyst_weights.get("technical", 1.0))
    opponent_relevance = float(analyst_weights.get(opponent_analyst, 1.0))

    own_feature_strength = abs(_technical_feature_strength(features, selected_horizon))
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


def technical_debate_node(state: BerkshireState):
    """
    Deterministic policy-driven technical debate node.

    Rating/confidence decisions are made before LLM generation. The LLM only
    provides explanation and rebuttal language.
    """
    ticker = state.get("ticker", "UNKNOWN")
    selected_horizon = normalize_horizon(state.get("horizon", "swing"))
    signals = state.get("analyst_signals", {})
    debate = state.get("debate", {})
    active = debate.get("active_challenge", {}) if isinstance(debate.get("active_challenge"), dict) else {}
    awaiting = debate.get("awaiting_response_from")

    current = signals.get("technical", {}) if isinstance(signals.get("technical"), dict) else {}
    current_rating = parse_rating(current.get("rating")) or Rating.HOLD
    current_confidence = _clamp_confidence(current.get("confidence", 0.0))
    features = current.get("features", {})
    current_details = current.get("details", "")

    # Fallback explanation without LLM.
    explanation = _deterministic_explanation(features, current_rating.value)
    revised_rating = current_rating
    revised_confidence = current_confidence
    debate_response = "Maintaining stance from quantitative technical indicators."
    claims_conceded = []
    claims_disputed = []
    weighting_statement = ""
    horizon_alignment_note = f"Technical view aligned to {horizon_label(selected_horizon)}."
    dialogue_response = ""

    # Only include challenge context if this call is for a debate turn.
    challenge_context = ""
    opponent_analyst = "unknown"
    opponent_rating = Rating.HOLD
    opponent_confidence = 0.0
    contradiction_severity = 0.0
    if awaiting == "technical" and active:
        coalition = active.get("coalition", {})
        my_case = active.get("my_case", {})
        opponent_case = active.get("opponent_case", {})
        opponent_analyst = _resolve_opponent_analyst(active, opponent_case, "technical")
        opponent_rating = parse_rating(opponent_case.get("rating")) or Rating.HOLD
        opponent_confidence = _clamp_confidence(opponent_case.get("confidence", 0.0))
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

    prompt = f"""You are a technical analyst.
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
1) Explain in plain English what the technical analysis indicators imply.
2) Explicitly address the opponent claims and rebut or concede.
3) Keep your reasoning grounded in the provided technical price/volume evidence for the selected horizon.
4) Do not invent facts beyond provided features/context.
5) IMPORTANT: the opponent is {opponent_analyst}. Do not describe the opponent argument as "technical".
6) Do not say you "fully agree", "perfectly mirror", or "fully align" with the opponent. Only concede specific points if warranted.
7) You are not deciding rating/confidence. These are fixed by policy; explain and defend them.

Return ONLY valid JSON:
{{"explanation":"<3-6 concise sentences>", "claims_conceded":["<opponent claim accepted>"], "claims_disputed":["<opponent claim disputed>"], "weighting_statement":"<which factor dominated and why>", "horizon_alignment_note":"<why this technical stance fits selected horizon>", "dialogue_response":"<1-2 natural-language sentences responding directly to opponent>"}}
"""

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        is_debate_turn = awaiting == "technical" and bool(active)
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
                debate_response = dialogue_response or (
                    f"I maintain the technical stance for {horizon_label(selected_horizon)} "
                    f"because the core price and volume indicators still support it."
                )
                lowered_response = debate_response.lower()
                agreement_phrases = (
                    "fully agree",
                    "perfectly mirrors",
                    "fully align",
                    "fully aligned",
                    "entirely agree",
                    "same assessment",
                )
                if any(phrase in lowered_response for phrase in agreement_phrases):
                    debate_response = _safe_dialogue_response(
                        opponent_analyst,
                        horizon_label(selected_horizon),
                        claims_disputed,
                        claims_conceded,
                    )
    except Exception as e:
        explanation = f"{explanation} LLM refinement unavailable."
        debate_response = "Maintaining stance from quantitative technical indicators due to unavailable LLM refinement."
        weighting_statement = ""

    # Guardrail: avoid incoherent phrasing that treats non-technical opponent as technical.
    lowered = debate_response.lower()
    if opponent_analyst != "technical" and "technical case you've presented" in lowered:
        debate_response = (
            f"I acknowledge the {opponent_analyst} concerns, but for "
            f"{horizon_label(selected_horizon)} the price action still supports my stance."
        )
    if any(phrase in lowered for phrase in ("fully agree", "perfectly mirrors", "same assessment")):
        debate_response = _safe_dialogue_response(
            opponent_analyst,
            horizon_label(selected_horizon),
            claims_disputed,
            claims_conceded,
        )

    print(
        f"\n[Technical Debate] {ticker}: rating {current_rating.value} -> {revised_rating.value}"
    )
    print(f"[Technical Debate] Explanation: {explanation}")
    position_changed = revised_rating.value != current_rating.value

    return {
        "analyst_signals": {
            "technical": {
                "rating": revised_rating.value,
                "confidence": revised_confidence,
                "features": features,
                "details": f"{current_details} | Technical explanation: {explanation}".strip(),
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
