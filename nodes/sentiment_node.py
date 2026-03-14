import os
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from shared.state_schema import BerkshireState

load_dotenv()


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
    data = state.get("data", {})
    news_articles = data.get("news_articles", [])

    # --- Edge case: no news available ---
    if not news_articles:
        print(f"[Sentiment] No news articles found for {ticker}. Defaulting to neutral.")
        return {
            "analyst_signals": {
                "sentiment": {
                    "signal": "neutral",
                    "confidence": 0.0,
                    "features": {
                        "sentiment_score": 0.0,
                        "news_volume": 0,
                    },
                    "details": "No news articles were available to analyze. Defaulting to neutral.",
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

Based on the overall tone and content of the news, provide a sentiment score
from 1 to 10 where:
- 1 to 3 means BEARISH (negative outlook, bad news dominates)
- 4 to 6 means NEUTRAL (mixed signals, no strong direction)
- 7 to 10 means BULLISH (positive outlook, good news dominates)

NEWS ARTICLES:
{headlines_text}

You MUST respond with ONLY valid JSON in this exact format, no extra text:
{{"score": <integer 1-10>, "reasoning": "<brief 1-2 sentence explanation>"}}
"""

    # --- Call Gemini ---
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        response = llm.invoke(prompt)
        raw_text = response.content.strip()

        # Clean potential markdown code fences from LLM response
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]  # Remove first line (```json)
            raw_text = raw_text.rsplit("```", 1)[0]  # Remove trailing ```
            raw_text = raw_text.strip()

        result = json.loads(raw_text)
        score = int(result.get("score", 5))
        reasoning = result.get("reasoning", "No reasoning provided.")

        # Clamp score to valid range
        score = max(1, min(10, score))

    except json.JSONDecodeError:
        print(f"[Sentiment] Warning: Could not parse LLM response as JSON. Raw: {raw_text}")
        score = 5
        reasoning = f"LLM response could not be parsed. Raw output: {raw_text}"

    except Exception as e:
        print(f"[Sentiment] Error calling Gemini: {e}")
        score = 5
        reasoning = f"Error during LLM call: {str(e)}"

    # --- Derive signal, confidence, and features ---
    if score <= 3:
        signal = "bearish"
    elif score >= 7:
        signal = "bullish"
    else:
        signal = "neutral"

    # Confidence: how far the score is from neutral (5), normalized to 0.0-1.0
    confidence = round(abs(score - 5.5) / 4.5, 2)
    confidence = min(confidence, 1.0)

    # Normalize score to -1.0 to 1.0 for RF/KNN features
    sentiment_score_normalized = round((score - 5.5) / 4.5, 2)

    # --- Print result for visibility ---
    label = signal.upper()
    print(f"\n[Sentiment] {ticker}: {label} (confidence: {confidence})")
    print(f"[Sentiment]   sentiment_score: {sentiment_score_normalized} -> {signal}")
    print(f"[Sentiment]   news_volume: {len(news_articles)}")
    print(f"[Sentiment] Reasoning: {reasoning}")

    # --- Write to state ---
    return {
        "analyst_signals": {
            "sentiment": {
                "signal": signal,
                "confidence": confidence,
                "features": {
                    "sentiment_score": sentiment_score_normalized,
                    "news_volume": len(news_articles),
                },
                "details": f"sentiment_score={sentiment_score_normalized} ({signal}); articles={len(news_articles)}; {reasoning}",
            }
        }
    }
