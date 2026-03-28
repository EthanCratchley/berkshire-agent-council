from shared.stance import (
    Rating,
    parse_rating,
    rating_from_aggregate_score,
    rating_to_score,
    rating_to_signal,
    sentiment_rating_from_score,
)


def test_parse_rating_accepts_canonical_and_alias():
    assert parse_rating("strong_buy") == Rating.STRONG_BUY
    assert parse_rating("underweight") == Rating.SELL
    assert parse_rating("bad_value") is None


def test_rating_score_signal_roundtrip():
    assert rating_to_score(Rating.STRONG_BUY) == 2
    assert rating_to_signal(Rating.STRONG_BUY) == "bullish"
    assert rating_from_aggregate_score(-4) == Rating.STRONG_SELL


def test_sentiment_rating_map():
    assert sentiment_rating_from_score(1) == Rating.STRONG_SELL
    assert sentiment_rating_from_score(3) == Rating.SELL
    assert sentiment_rating_from_score(5) == Rating.HOLD
    assert sentiment_rating_from_score(8) == Rating.BUY
    assert sentiment_rating_from_score(10) == Rating.STRONG_BUY
