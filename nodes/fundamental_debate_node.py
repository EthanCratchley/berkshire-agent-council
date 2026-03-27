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
from shared.stance import Rating, parse_rating

load_dotenv()


def _deterministic_explanation(features: dict, rating: str) -> str:
    feature_bits = []
    for key in ("pe_ratio", "debt_to_equity", "profit_margin", "revenue_growth", "eps"):
        value = features.get(key)
        if value is None:
            continue
        feature_bits.append(f"{key}={value:.2f}")
    metrics = ", ".join(feature_bits) if feature_bits else "limited fundamental metrics available"
    return (
        f"The current fundamental stance is {rating}. "
        f"This is based on {metrics}. Lower valuation/leverage and higher profitability/growth "
        f"tend to support stronger ratings."
    )


def fundamental_debate_node(state: BerkshireState):
    """
    LLM-powered explainer/debater for fundamentals.

    Keeps rating anchored to existing quantitative output unless the LLM returns
    a valid revised rating from the allowed enum set.
    """
    ticker = state.get("ticker", "UNKNOWN")
    signals = state.get("analyst_signals", {})
    debate = state.get("debate", {})
    active = debate.get("active_challenge", {}) if isinstance(debate.get("active_challenge"), dict) else {}
    awaiting = debate.get("awaiting_response_from")

    current = signals.get("fundamental", {}) if isinstance(signals.get("fundamental"), dict) else {}
    current_rating = parse_rating(current.get("rating")) or Rating.HOLD
    current_confidence = current.get("confidence", 0.0)
    features = current.get("features", {})
    current_details = current.get("details", "")

    # Fallback explanation without LLM.
    explanation = _deterministic_explanation(features, current_rating.value)
    revised_rating = current_rating

    # Only include challenge context if this call is for a debate turn.
    challenge_context = ""
    if awaiting == "fundamental" and active:
        coalition = active.get("coalition", {})
        challenge_context = (
            f"\nDEBATE CONTEXT:\n"
            f"- Challenge ID: {active.get('id', '')}\n"
            f"- Action requested: {active.get('action', 'revise_or_defend')}\n"
            f"- Primary opponent: {active.get('primary_opponent', '')}\n"
            f"- Contradiction reason: {active.get('reason', '')}\n"
            f"- Opponent supporters: {len(coalition.get('supporters_of_opponent', []))}\n"
            f"- Partial-agreement analysts: {len(coalition.get('partial', []))}\n"
        )

    prompt = f"""You are a fundamental equity analyst.
Ticker: {ticker}
Current rating: {current_rating.value}
Current confidence: {current_confidence}
Current details: {current_details}
Features: {features}
{challenge_context}

Task:
1) Explain in plain English what the fundamental metrics imply.
2) If justified by the provided features and challenge context, optionally revise rating.
3) Do not invent facts beyond provided features/context.

Return ONLY valid JSON:
{{"explanation":"<3-6 concise sentences>", "revised_rating":"<strong_buy|buy|hold|sell|strong_sell>"}}
"""

    try:
        if ChatGoogleGenerativeAI is not None:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            response = llm.invoke(prompt)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                raw = raw.rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)
            llm_explanation = str(parsed.get("explanation", "")).strip()
            llm_rating = parse_rating(parsed.get("revised_rating"))
            if llm_explanation:
                explanation = llm_explanation
            if llm_rating is not None:
                revised_rating = llm_rating
    except Exception as e:
        explanation = f"{explanation} LLM refinement unavailable: {str(e)}"

    print(
        f"\n[Fundamental Debate] {ticker}: rating {current_rating.value} -> {revised_rating.value}"
    )
    print(f"[Fundamental Debate] Explanation: {explanation}")

    return {
        "analyst_signals": {
            "fundamental": {
                "rating": revised_rating.value,
                "confidence": current_confidence,
                "features": features,
                "details": f"{current_details} | Fundamental explanation: {explanation}".strip(),
                "narrative": explanation,
            }
        }
    }

