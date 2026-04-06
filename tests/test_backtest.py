from models.backtest import (
    _two_way_vote,
    _label_midpoint_returns,
    POSITION_WEIGHTS,
)


class TestTwoWayVote:
    def test_agreement(self):
        assert _two_way_vote("BUY", "BUY") == "BUY"
        assert _two_way_vote("STRONG SELL", "STRONG SELL") == "STRONG SELL"

    def test_average_bullish(self):
        # strong_buy(2) + buy(1) = avg 1.5 → strong_buy
        assert _two_way_vote("STRONG BUY", "BUY") == "STRONG BUY"

    def test_average_neutral(self):
        # buy(1) + sell(-1) = avg 0 → hold
        assert _two_way_vote("BUY", "SELL") == "HOLD"

    def test_average_mild_bullish(self):
        # buy(1) + hold(0) = avg 0.5 → buy
        assert _two_way_vote("BUY", "HOLD") == "BUY"

    def test_average_mild_bearish(self):
        # sell(-1) + hold(0) = avg -0.5 → sell
        assert _two_way_vote("SELL", "HOLD") == "SELL"

    def test_average_bearish(self):
        # strong_sell(-2) + sell(-1) = avg -1.5 → strong_sell
        assert _two_way_vote("STRONG SELL", "SELL") == "STRONG SELL"


class TestPositionWeights:
    def test_all_labels_covered(self):
        labels = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
        for label in labels:
            assert label in POSITION_WEIGHTS

    def test_symmetry(self):
        assert POSITION_WEIGHTS["STRONG BUY"] == -POSITION_WEIGHTS["STRONG SELL"]
        assert POSITION_WEIGHTS["BUY"] == -POSITION_WEIGHTS["SELL"]
        assert POSITION_WEIGHTS["HOLD"] == 0.0


class TestLabelMidpointReturns:
    def test_hold_is_zero(self):
        midpoints = _label_midpoint_returns((0.05, 0.02, -0.02, -0.05))
        assert midpoints["HOLD"] == 0.0

    def test_bullish_positive(self):
        midpoints = _label_midpoint_returns((0.05, 0.02, -0.02, -0.05))
        assert midpoints["STRONG BUY"] > 0
        assert midpoints["BUY"] > 0

    def test_bearish_negative(self):
        midpoints = _label_midpoint_returns((0.05, 0.02, -0.02, -0.05))
        assert midpoints["STRONG SELL"] < 0
        assert midpoints["SELL"] < 0

    def test_ordering(self):
        midpoints = _label_midpoint_returns((0.05, 0.02, -0.02, -0.05))
        assert midpoints["STRONG BUY"] > midpoints["BUY"] > midpoints["HOLD"]
        assert midpoints["HOLD"] > midpoints["SELL"] > midpoints["STRONG SELL"]
