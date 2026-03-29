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
from shared.horizon import normalize_horizon, horizon_label
from shared.stance import Rating, parse_rating

load_dotenv()

_RATING_ORDER = [
    Rating.STRONG_SELL,
    Rating.SELL,
    Rating.HOLD,
    Rating.BUY,
    Rating.STRONG_BUY,
]


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
        "final_position",
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
    final_position = payload.get("final_position")
    if not isinstance(final_position, dict):
        return False
    if parse_rating(final_position.get("rating")) is None:
        return False
    try:
        float(final_position.get("confidence"))
    except (TypeError, ValueError):
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


def technical_debate_node(state: BerkshireState):
    """
    LLM-powered explainer/debater for technical analysis.

    Keeps rating anchored to existing quantitative output unless the LLM returns
    a valid revised rating from the allowed enum set.
    """
    ticker = state.get("ticker", "UNKNOWN")
    selected_horizon = normalize_horizon(state.get("horizon", "swing"))
    signals = state.get("analyst_signals", {})
    debate = state.get("debate", {})
    active = debate.get("active_challenge", {}) if isinstance(debate.get("active_challenge"), dict) else {}
    awaiting = debate.get("awaiting_response_from")

    current = signals.get("technical", {}) if isinstance(signals.get("technical"), dict) else {}
    current_rating = parse_rating(current.get("rating")) or Rating.HOLD
    current_confidence = current.get("confidence", 0.0)
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
    if awaiting == "technical" and active:
        coalition = active.get("coalition", {})
        my_case = active.get("my_case", {})
        opponent_case = active.get("opponent_case", {})
        opponent_analyst = str(opponent_case.get("analyst", "unknown"))
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

    prompt = f"""You are a technical analyst.
Ticker: {ticker}
Selected horizon: {horizon_label(selected_horizon)}
Current rating: {current_rating.value}
Current confidence: {current_confidence}
Current details: {current_details}
Features: {features}
{challenge_context}

Task:
1) Explain in plain English what the technical analysis indicators imply.
2) Explicitly address the opponent claims and rebut or concede.
3) Keep your reasoning grounded in the provided technical price/volume evidence for the selected horizon.
4) Do not invent facts beyond provided features/context.
5) IMPORTANT: the opponent is {opponent_analyst}. Do not describe the opponent argument as "technical".

Return ONLY valid JSON:
{{"explanation":"<3-6 concise sentences>", "claims_conceded":["<opponent claim accepted>"], "claims_disputed":["<opponent claim disputed>"], "weighting_statement":"<which factor dominated and why>", "horizon_alignment_note":"<why this technical stance fits selected horizon>", "dialogue_response":"<1-2 natural-language sentences responding directly to opponent>", "final_position":{{"rating":"<strong_buy|buy|hold|sell|strong_sell>", "confidence":<0.0-1.0>}}}}
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
                    '"weighting_statement":"...", "horizon_alignment_note":"...", "dialogue_response":"...", '
                    '"final_position":{"rating":"strong_buy|buy|hold|sell|strong_sell","confidence":0.0}}\n'
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
                final_position = parsed.get("final_position", {})
                llm_rating = parse_rating(final_position.get("rating"))
                if llm_rating is not None:
                    revised_rating = _bounded_rating_update(current_rating, llm_rating, max_step=1)
                parsed_conf = max(0.0, min(float(final_position.get("confidence", current_confidence)), 1.0))
                revised_confidence = round((float(current_confidence) + parsed_conf) / 2.0, 2)
                debate_response = dialogue_response or (
                    f"I maintain the technical stance for {horizon_label(selected_horizon)} "
                    f"because the core price and volume indicators still support it."
                )
    except Exception as e:
        explanation = f"{explanation} LLM refinement unavailable: {str(e)}"
        debate_response = f"LLM unavailable; maintaining quant stance. ({str(e)})"
        weighting_statement = ""

    # Guardrail: avoid incoherent phrasing that treats non-technical opponent as technical.
    lowered = debate_response.lower()
    if opponent_analyst != "technical" and "technical case you've presented" in lowered:
        debate_response = (
            f"I acknowledge the {opponent_analyst} concerns, but for "
            f"{horizon_label(selected_horizon)} the price action still supports my stance."
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
                "weighting_statement": weighting_statement,
                "horizon_alignment_note": horizon_alignment_note,
            }
        }
    }
