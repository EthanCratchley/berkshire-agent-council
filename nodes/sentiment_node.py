import os
import json

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
from shared.stance import Rating, parse_rating, sentiment_rating_from_score, rating_to_score, rating_to_signal

load_dotenv()


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
        "score",
        "reasoning",
        "claims_conceded",
        "claims_disputed",
        "weighting_statement",
        "horizon_alignment_note",
        "dialogue_response",
        "final_position",
    )
    if any(key not in payload for key in required):
        return False
    try:
        score = int(payload.get("score"))
    except (TypeError, ValueError):
        return False
    if score < 1 or score > 10:
        return False
    if not isinstance(payload.get("reasoning"), str):
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


def sentiment_node(state: BerkshireState):
    """
    Node for news sentiment analysis.

    Reads news articles from state["data"]["news_articles"] (populated by
    the data_fetcher via Finnhub), sends the headlines + summaries to
    Google Gemini, and returns a sentiment score from 1 to 10.

    Score guide:
        1-3  = Bearish  (negative news dominates)
        4-6  = Neutral  (mixed or no strong signal)
        7-10 = Bullish  (positive news dominates)
    """
    ticker = state.get("ticker", "UNKNOWN")
    selected_horizon = normalize_horizon(state.get("horizon", "swing"))
    data = state.get("data", {})
    news_articles = data.get("news_articles", [])
    debate = state.get("debate", {})
    prior_sentiment = state.get("analyst_signals", {}).get("sentiment", {})
    prior_rating = prior_sentiment.get("rating")

    challenge_context = ""
    is_debate_turn = False
    active_challenge = debate.get("active_challenge", {})
    awaiting = debate.get("awaiting_response_from")
    if isinstance(active_challenge, dict) and awaiting == "sentiment":
        is_debate_turn = True
        coalition = active_challenge.get("coalition", {})
        my_case = active_challenge.get("my_case", {})
        opponent_case = active_challenge.get("opponent_case", {})
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
            f"- Challenge ID: {active_challenge.get('id', '')}\n"
            f"- You are asked to: {active_challenge.get('action', 'revise_or_defend')}\n"
            f"- Primary disagreement: {active_challenge.get('reason', '')}\n"
            f"- Your current case: rating={my_case.get('rating', '')}, confidence={my_case.get('confidence', '')}, details={my_case.get('details', '')}\n"
            f"- Opponent case: rating={opponent_case.get('rating', '')}, confidence={opponent_case.get('confidence', '')}, details={opponent_case.get('details', '')}\n"
            f"- Opponent last rebuttal: {opponent_case.get('last_debate_response', '')}\n"
            f"- Supporters of opponent: {len(coalition.get('supporters_of_opponent', []))}\n"
            f"- Partial-agreement analysts: {len(coalition.get('partial', []))}\n"
            f"- Supporting analyst arguments: {supporter_lines}\n"
            f"- Partial-agreement analyst arguments: {partial_lines}\n"
            f"Keep your response grounded in the provided news evidence.\n"
        )

    # --- Edge case: no news available ---
    if not news_articles:
        print(f"[Sentiment] No news articles found for {ticker}. Defaulting to neutral.")
        position_changed = prior_rating not in (None, Rating.HOLD.value)
        return {
            "analyst_signals": {
                "sentiment": {
                    "rating": Rating.HOLD.value,
                    "confidence": 0.0,
                    "features": {
                        "sentiment_score": 0.0,
                        "news_volume": 0,
                    },
                    "details": "No news articles were available to analyze. Defaulting to neutral.",
                    "debate_response": "No news data available to revise stance; defaulting to hold.",
                    "position_changed": position_changed,
                    "counterpoints_addressed": [],
                    "claims_conceded": [],
                    "claims_disputed": [],
                    "final_position": {
                        "rating": Rating.HOLD.value,
                        "confidence": 0.0,
                    },
                    "weighting_statement": "No evidence available.",
                    "horizon_alignment_note": (
                        f"No sentiment evidence available for {horizon_label(selected_horizon)}."
                    ),
                }
            }
        }

    # --- Build the headlines summary for the LLM ---
    headlines_text = ""
    for i, article in enumerate(news_articles, start=1):
        headline = article.get("headline", "No headline")
        summary = article.get("summary", "")
        source = article.get("source", "Unknown")
        date = article.get("datetime", "")

        headlines_text += f"{i}. [{source}] {headline}\n"
        if summary:
            headlines_text += f"   Summary: {summary}\n"
        if date:
            headlines_text += f"   Date: {date}\n"
        headlines_text += "\n"

    # --- Prompt for Gemini ---
    prompt = f"""You are a stock market sentiment analyst. Analyze the following news headlines
and summaries for the stock ticker {ticker}.
Selected horizon: {horizon_label(selected_horizon)}.
Prioritize sentiment evidence relevance to this horizon.

Based on the overall tone and content of the news, provide a sentiment score
from 1 to 10 where:
- 1 to 3 means BEARISH (negative outlook, bad news dominates)
- 4 to 6 means NEUTRAL (mixed signals, no strong direction)
- 7 to 10 means BULLISH (positive outlook, good news dominates)

NEWS ARTICLES:
{headlines_text}
{challenge_context}

You MUST respond with ONLY valid JSON in this exact format, no extra text:
{{"score": <integer 1-10>, "reasoning": "<brief 1-2 sentence explanation>", "claims_conceded": ["<opponent claim accepted>"], "claims_disputed": ["<opponent claim disputed>"], "weighting_statement": "<what factor mattered more and why>", "horizon_alignment_note": "<why this sentiment view matches the selected horizon>", "dialogue_response": "<1-2 natural-language sentences responding directly to opponent>", "final_position": {{"rating":"<strong_buy|buy|hold|sell|strong_sell>", "confidence": <0.0-1.0>}}}}
"""

    claims_conceded = []
    claims_disputed = []
    weighting_statement = ""
    horizon_alignment_note = ""
    dialogue_response = ""

    # --- Call Gemini ---
    try:
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError(
                "langchain_google_genai is not installed. Unable to run LLM sentiment analysis."
            )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        response = llm.invoke(prompt)
        raw_text = _clean_json_text(response.content)

        result = json.loads(raw_text)
        if not _valid_contract(result):
            repair_prompt = (
                "Return ONLY valid JSON for this schema:\n"
                '{"score":5,"reasoning":"...","claims_conceded":["..."],"claims_disputed":["..."],'
                '"weighting_statement":"...","horizon_alignment_note":"...","dialogue_response":"...",'
                '"final_position":{"rating":"strong_buy|buy|hold|sell|strong_sell","confidence":0.0}}\n'
                f"Original response:\n{raw_text}"
            )
            repair = llm.invoke(repair_prompt)
            repaired_raw = _clean_json_text(repair.content)
            result = json.loads(repaired_raw)

        if not _valid_contract(result):
            raise ValueError("LLM response did not satisfy strict debate contract.")

        score = int(result.get("score", 5))
        reasoning = str(result.get("reasoning", "No reasoning provided.")).strip()
        claims_conceded = result.get("claims_conceded", [])
        claims_disputed = result.get("claims_disputed", [])
        weighting_statement = str(result.get("weighting_statement", "")).strip()
        horizon_alignment_note = str(result.get("horizon_alignment_note", "")).strip()
        dialogue_response = str(result.get("dialogue_response", "")).strip()

        # Clamp score to valid range
        score = max(1, min(10, score))

    except json.JSONDecodeError:
        print(f"[Sentiment] Warning: Could not parse LLM response as JSON. Raw: {raw_text}")
        score = 5
        reasoning = f"LLM response could not be parsed. Raw output: {raw_text}"
        claims_conceded = []
        claims_disputed = []
        weighting_statement = ""
        horizon_alignment_note = ""
        dialogue_response = ""

    except Exception as e:
        print(f"[Sentiment] Error calling Gemini: {e}")
        score = 5
        reasoning = f"Error during LLM call: {str(e)}"
        claims_conceded = []
        claims_disputed = []
        weighting_statement = ""
        horizon_alignment_note = ""
        dialogue_response = ""

    # --- Derive stance/rating, signal, confidence, and features ---
    rating = sentiment_rating_from_score(score)
    stance_score = rating_to_score(rating)
    signal = rating_to_signal(rating)
    position_changed = prior_rating is not None and prior_rating != rating.value

    # Confidence: how far the score is from neutral (5.5), normalized to 0.0-1.0
    confidence = round(abs(score - 5.5) / 4.5, 2)
    confidence = min(confidence, 1.0)
    debate_response = dialogue_response or (
        f"I still favor the {signal} case for {horizon_label(selected_horizon)} "
        f"because sentiment evidence is stronger than the opposing thesis."
    )

    # --- Print result for visibility ---
    label = signal.upper()
    print(
        f"\n[Sentiment] {ticker}: {label} / {rating.value.upper()} "
        f"(stance_score={stance_score:+d}, confidence: {confidence})"
    )
    print(f"[Sentiment]   sentiment_score: {score} -> {signal}")
    print(f"[Sentiment]   news_volume: {len(news_articles)}")
    print(f"[Sentiment] Reasoning: {reasoning}")

    # --- Write to state ---
    return {
        "analyst_signals": {
            "sentiment": {
                "rating": rating.value,
                "confidence": confidence,
                "features": {
                    "sentiment_score": score,
                    "news_volume": len(news_articles),
                },
                "details": (
                    f"sentiment_score={score} ({signal}); rating={rating.value}; "
                    f"articles={len(news_articles)}; horizon={selected_horizon}; {reasoning}"
                ),
                "debate_response": (
                    debate_response
                    if is_debate_turn and debate_response
                    else ("Maintaining stance based on current news weighting.")
                ),
                "position_changed": position_changed,
                "counterpoints_addressed": claims_disputed if is_debate_turn else [],
                "claims_conceded": claims_conceded if is_debate_turn else [],
                "claims_disputed": claims_disputed if is_debate_turn else [],
                "final_position": {
                    "rating": rating.value,
                    "confidence": confidence,
                },
                "weighting_statement": weighting_statement if is_debate_turn else "",
                "horizon_alignment_note": (
                    horizon_alignment_note
                    if horizon_alignment_note
                    else f"Sentiment stance prioritized for {horizon_label(selected_horizon)}."
                ),
            }
        }
    }
