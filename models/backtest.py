"""
Backtest RF, KNN, and their 2-way vote against the held-out test set.

For each horizon, loads the trained models, predicts on the test split, then
simulates strategy returns by mapping predictions to position weights and
labels to approximate forward returns. Compares each strategy against a
buy-and-hold baseline.

Note: The LLM debate system cannot be backtested offline (requires live API
calls per ticker). This backtest covers the classical models and their vote,
which form 2 of the 3 inputs to the live 3-way comparison vote.
"""

import os
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.feature_engineering import FEATURE_ORDER
from shared.horizon import HORIZON_LABEL_CONFIG, VALID_HORIZONS
from shared.stance import parse_rating, rating_to_score

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cached_dataset.csv",
)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Position sizing: how much exposure each prediction implies.
POSITION_WEIGHTS = {
    "STRONG BUY": 1.0,
    "BUY": 0.5,
    "HOLD": 0.0,
    "SELL": -0.5,
    "STRONG SELL": -1.0,
}

# Approximate midpoint return for each label bucket, per horizon.
# Used to simulate PnL when exact forward returns aren't in the dataset.
def _label_midpoint_returns(thresholds: tuple) -> dict:
    sb, b, s, ss = thresholds
    return {
        "STRONG BUY": (sb + sb * 1.5) / 2,   # midpoint of (threshold, ~1.5x threshold)
        "BUY": (b + sb) / 2,
        "HOLD": 0.0,
        "SELL": (s + ss) / 2,
        "STRONG SELL": (ss + ss * 1.5) / 2,
    }


def _load(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _two_way_vote(rf_pred: str, knn_pred: str) -> str:
    """Simple 2-way vote between RF and KNN. Ties go to the less extreme."""
    if rf_pred == knn_pred:
        return rf_pred

    rf_score = rating_to_score(parse_rating(rf_pred))
    knn_score = rating_to_score(parse_rating(knn_pred))
    avg = (rf_score + knn_score) / 2

    if avg >= 1.5:
        return "STRONG BUY"
    if avg >= 0.5:
        return "BUY"
    if avg <= -1.5:
        return "STRONG SELL"
    if avg <= -0.5:
        return "SELL"
    return "HOLD"


def backtest_horizon(horizon: str):
    """Run backtest for a single horizon. Returns results dict."""
    cfg = HORIZON_LABEL_CONFIG[horizon]
    label_col = f"label_{horizon}"
    midpoints = _label_midpoint_returns(cfg["thresholds"])

    print(f"\n{'='*70}")
    print(f"  BACKTEST: {horizon.upper()} HORIZON  ({cfg['forward_days']}-day forward)")
    print(f"  Thresholds: {cfg['thresholds']}")
    print(f"{'='*70}")

    df = pd.read_csv(DATASET_PATH)
    df = df.sort_values("date").reset_index(drop=True)
    hdf = df.dropna(subset=[label_col]).reset_index(drop=True)

    split_idx = int(len(hdf) * 0.8)
    test_df = hdf.iloc[split_idx:].copy()
    X_test = test_df[FEATURE_ORDER].values
    y_test = test_df[label_col].values

    print(f"Test samples: {len(y_test)}")
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Tickers in test: {test_df['ticker'].nunique()}")

    imputer = _load(f"imputer_{horizon}.pkl")
    rf = _load(f"rf_{horizon}.pkl")
    knn = _load(f"knn_{horizon}.pkl")
    scaler = _load(f"scaler_{horizon}.pkl")

    if imputer is None or rf is None or knn is None or scaler is None:
        print("  Missing model artifacts. Skipping.")
        return None

    X_imp = imputer.transform(X_test)
    X_scaled = scaler.transform(X_imp)

    rf_preds = rf.predict(X_imp)
    knn_preds = knn.predict(X_scaled)
    vote_preds = np.array([_two_way_vote(r, k) for r, k in zip(rf_preds, knn_preds)])

    # --- Classification accuracy ---
    rf_acc = accuracy_score(y_test, rf_preds)
    knn_acc = accuracy_score(y_test, knn_preds)
    vote_acc = accuracy_score(y_test, vote_preds)

    print(f"\n--- Classification Accuracy ---")
    print(f"  Random Forest: {rf_acc:.4f}")
    print(f"  KNN:           {knn_acc:.4f}")
    print(f"  2-Way Vote:    {vote_acc:.4f}")

    # --- Directional accuracy (bullish/bearish/neutral) ---
    def to_direction(label):
        score = rating_to_score(parse_rating(label))
        if score > 0:
            return "bullish"
        if score < 0:
            return "bearish"
        return "neutral"

    y_dir = np.array([to_direction(l) for l in y_test])
    rf_dir = np.array([to_direction(l) for l in rf_preds])
    knn_dir = np.array([to_direction(l) for l in knn_preds])
    vote_dir = np.array([to_direction(l) for l in vote_preds])

    rf_dir_acc = (rf_dir == y_dir).mean()
    knn_dir_acc = (knn_dir == y_dir).mean()
    vote_dir_acc = (vote_dir == y_dir).mean()

    print(f"\n--- Directional Accuracy (bullish/neutral/bearish) ---")
    print(f"  Random Forest: {rf_dir_acc:.4f}")
    print(f"  KNN:           {knn_dir_acc:.4f}")
    print(f"  2-Way Vote:    {vote_dir_acc:.4f}")

    # --- Simulated strategy returns ---
    actual_returns = np.array([midpoints.get(l, 0.0) for l in y_test])

    def strategy_return(preds):
        positions = np.array([POSITION_WEIGHTS.get(p, 0.0) for p in preds])
        return positions * actual_returns

    rf_returns = strategy_return(rf_preds)
    knn_returns = strategy_return(knn_preds)
    vote_returns = strategy_return(vote_preds)
    bnh_returns = actual_returns  # buy-and-hold: always long

    rf_cumulative = rf_returns.sum()
    knn_cumulative = knn_returns.sum()
    vote_cumulative = vote_returns.sum()
    bnh_cumulative = bnh_returns.sum()

    # Per-trade average
    n = len(y_test)
    rf_avg = rf_cumulative / n
    knn_avg = knn_cumulative / n
    vote_avg = vote_cumulative / n
    bnh_avg = bnh_cumulative / n

    print(f"\n--- Simulated Strategy Returns (approx. using label midpoints) ---")
    print(f"  {'Strategy':<20} {'Cumulative':>12} {'Avg/Trade':>12} {'vs B&H':>10}")
    print(f"  {'-'*54}")
    for name, cum, avg in [
        ("Random Forest", rf_cumulative, rf_avg),
        ("KNN", knn_cumulative, knn_avg),
        ("2-Way Vote", vote_cumulative, vote_avg),
        ("Buy & Hold", bnh_cumulative, bnh_avg),
    ]:
        vs_bnh = f"{(cum / bnh_cumulative - 1) * 100:+.1f}%" if bnh_cumulative != 0 else "N/A"
        print(f"  {name:<20} {cum:>12.4f} {avg:>12.6f} {vs_bnh:>10}")

    # --- Win rate (positive return per trade) ---
    def win_rate(returns):
        trades = returns[returns != 0]
        if len(trades) == 0:
            return 0.0, 0
        return (trades > 0).mean(), len(trades)

    rf_wr, rf_trades = win_rate(rf_returns)
    knn_wr, knn_trades = win_rate(knn_returns)
    vote_wr, vote_trades = win_rate(vote_returns)

    print(f"\n--- Win Rate (% of active trades profitable) ---")
    print(f"  Random Forest: {rf_wr:.1%} ({rf_trades} trades)")
    print(f"  KNN:           {knn_wr:.1%} ({knn_trades} trades)")
    print(f"  2-Way Vote:    {vote_wr:.1%} ({vote_trades} trades)")

    # --- Vote agreement ---
    agree = (rf_preds == knn_preds).mean()
    print(f"\n--- Model Agreement ---")
    print(f"  RF/KNN exact agreement: {agree:.1%}")
    print(f"  RF/KNN directional agreement: {(rf_dir == knn_dir).mean():.1%}")

    return {
        "horizon": horizon,
        "n_test": n,
        "rf_accuracy": rf_acc,
        "knn_accuracy": knn_acc,
        "vote_accuracy": vote_acc,
        "rf_dir_accuracy": rf_dir_acc,
        "knn_dir_accuracy": knn_dir_acc,
        "vote_dir_accuracy": vote_dir_acc,
        "rf_cumulative": rf_cumulative,
        "knn_cumulative": knn_cumulative,
        "vote_cumulative": vote_cumulative,
        "bnh_cumulative": bnh_cumulative,
        "rf_win_rate": rf_wr,
        "knn_win_rate": knn_wr,
        "vote_win_rate": vote_wr,
        "rf_knn_agreement": agree,
    }


def backtest(horizons: list[str] | None = None):
    """Run backtest for specified horizons (default: all)."""
    if horizons is None:
        horizons = sorted(VALID_HORIZONS)

    results = {}
    for h in horizons:
        if h not in HORIZON_LABEL_CONFIG:
            print(f"Skipping unknown horizon: {h}")
            continue
        results[h] = backtest_horizon(h)

    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  BACKTEST SUMMARY")
        print(f"{'='*70}")
        print(f"{'Horizon':<8} {'RF Acc':>8} {'KNN Acc':>8} {'Vote Acc':>9} "
              f"{'RF Dir':>8} {'KNN Dir':>8} {'Vote Dir':>9} "
              f"{'Vote vs B&H':>12}")
        print(f"{'-'*73}")
        for h in sorted(results):
            r = results[h]
            if r is None:
                continue
            vs = f"{(r['vote_cumulative'] / r['bnh_cumulative'] - 1) * 100:+.1f}%" if r["bnh_cumulative"] != 0 else "N/A"
            print(f"{h:<8} {r['rf_accuracy']:>8.4f} {r['knn_accuracy']:>8.4f} {r['vote_accuracy']:>9.4f} "
                  f"{r['rf_dir_accuracy']:>8.4f} {r['knn_dir_accuracy']:>8.4f} {r['vote_dir_accuracy']:>9.4f} "
                  f"{vs:>12}")

    return results


if __name__ == "__main__":
    import sys as _sys
    args = _sys.argv[1:]
    selected = [a for a in args if a in VALID_HORIZONS] or None
    backtest(selected)
