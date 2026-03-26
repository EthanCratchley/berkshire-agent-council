"""
Industry-standard stance contract.

Canonical field in node outputs: rating (enum value string).
Derived at runtime: score (+2..-2), signal (bullish/neutral/bearish).
"""

from enum import Enum
from typing import Optional


class Rating(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


_RATING_TO_SCORE = {
    Rating.STRONG_BUY: 2,
    Rating.BUY: 1,
    Rating.HOLD: 0,
    Rating.SELL: -1,
    Rating.STRONG_SELL: -2,
}

_ALIAS_TO_RATING = {
    "strong_buy": Rating.STRONG_BUY,
    "buy": Rating.BUY,
    "hold": Rating.HOLD,
    "sell": Rating.SELL,
    "strong_sell": Rating.STRONG_SELL,
    # Common external aliases that map to canonical ratings
    "outperform": Rating.BUY,
    "overweight": Rating.BUY,
    "neutral": Rating.HOLD,
    "market_weight": Rating.HOLD,
    "underperform": Rating.SELL,
    "underweight": Rating.SELL,
}

_SCORE_TO_SIGNAL = {
    1: "bullish",
    0: "neutral",
    -1: "bearish",
}


def normalize_rating(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def parse_rating(value) -> Optional[Rating]:
    key = normalize_rating(value)
    return _ALIAS_TO_RATING.get(key)


def rating_to_score(rating: Rating) -> int:
    return _RATING_TO_SCORE[rating]


def rating_to_signal(rating: Rating) -> str:
    score = rating_to_score(rating)
    sign = 1 if score > 0 else (-1 if score < 0 else 0)
    return _SCORE_TO_SIGNAL[sign]


def rating_from_aggregate_score(score: int) -> Rating:
    if score >= 3:
        return Rating.STRONG_BUY
    if score >= 2:
        return Rating.BUY
    if score <= -3:
        return Rating.STRONG_SELL
    if score <= -2:
        return Rating.SELL
    return Rating.HOLD


def sentiment_rating_from_score(score: int) -> Rating:
    if score <= 2:
        return Rating.STRONG_SELL
    if score <= 3:
        return Rating.SELL
    if score <= 6:
        return Rating.HOLD
    if score <= 8:
        return Rating.BUY
    return Rating.STRONG_BUY
