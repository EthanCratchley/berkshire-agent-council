"""
Live assessment — runs the full pipeline (LLM debate + RF + KNN + 3-way vote)
on a sample of tickers and reports how the three models interact.

Measures:
- Agreement rates between LLM, RF, and KNN
- How often the 3-way vote differs from each individual model
- Debate behavior: rounds, contradictions found, resolution rate
- Per-analyst stance distribution

Run: python models/live_assessment.py [horizon]
Defaults to swing horizon. Takes ~2-3 minutes for 20 tickers.
"""

import os
import sys
import time
import logging

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.state_schema import make_initial_debate_state
from shared.horizon import normalize_horizon, horizon_label, horizon_day_range
from shared.stance import parse_rating, rating_to_score

# Diverse sample across sectors and market caps.
ASSESSMENT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",   # mega-cap tech
    "JPM", "GS",                                   # financials
    "JNJ", "UNH",                                  # healthcare
    "XOM", "CVX",                                   # energy
    "PG", "KO",                                     # consumer staples
    "CAT", "BA",                                    # industrials
    "NEE", "D",                                     # utilities
    "NFLX", "DIS",                                  # communication
    "TSLA",                                         # high-vol consumer cyclical
]


def _normalize_label(label):
    """Normalize any label format to lowercase underscore form."""
    r = parse_rating(label)
    return r.value if r is not None else None


def run_assessment(horizon: str = "swing"):
    from main import app

    horizon = normalize_horizon(horizon)
    horizon_lbl = horizon_label(horizon)
    min_days, max_days = horizon_day_range(horizon)

    print(f"\n{'='*70}")
    print(f"  LIVE ASSESSMENT — {horizon_lbl}")
    print(f"  {len(ASSESSMENT_TICKERS)} tickers")
    print(f"{'='*70}\n")

    results = []

    for i, ticker in enumerate(ASSESSMENT_TICKERS, 1):
        print(f"\n[{i}/{len(ASSESSMENT_TICKERS)}] {ticker}")
        print(f"{'-'*40}")

        initial_state = {
            "ticker": ticker,
            "horizon": horizon,
            "horizon_days": {"min": min_days, "max": max_days},
            "data": {},
            "analyst_signals": {},
            "debate": make_initial_debate_state(max_rounds=3),
            "final_report": {},
        }

        start = time.time()
        try:
            final_state = app.invoke(initial_state)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        elapsed = time.time() - start

        report = final_state.get("final_report", {})
        vote = report.get("vote", {})
        cm = report.get("classical_models", {})
        debate = final_state.get("debate", {})
        signals = final_state.get("analyst_signals", {})

        llm_rec = _normalize_label(report.get("llm_recommendation"))
        rf_pred = _normalize_label(cm.get("rf", {}).get("prediction")) if cm.get("rf") else None
        knn_pred = _normalize_label(cm.get("knn", {}).get("prediction")) if cm.get("knn") else None
        voted = _normalize_label(report.get("recommendation"))

        # Analyst stances
        analyst_stances = {}
        for analyst in ("sentiment", "fundamental", "technical", "macro"):
            sig = signals.get(analyst, {})
            if isinstance(sig, dict) and sig.get("rating"):
                analyst_stances[analyst] = {
                    "rating": _normalize_label(sig["rating"]),
                    "confidence": sig.get("confidence"),
                }

        # Debate stats
        history = debate.get("history", [])
        debate_turns = sum(1 for e in history if isinstance(e, dict) and e.get("event") == "debater_turn_result")
        unresolved = report.get("unresolved_contradictions", [])

        entry = {
            "ticker": ticker,
            "llm": llm_rec,
            "rf": rf_pred,
            "knn": knn_pred,
            "voted": voted,
            "weighted_score": report.get("weighted_score"),
            "debate_turns": debate_turns,
            "unresolved": len(unresolved),
            "analyst_stances": analyst_stances,
            "elapsed": round(elapsed, 1),
        }
        results.append(entry)

        print(f"  LLM: {llm_rec}  RF: {rf_pred}  KNN: {knn_pred}  Vote: {voted} ({vote.get('method')}, {vote.get('agreement')})")
        print(f"  Debate turns: {debate_turns}, Unresolved: {len(unresolved)}, Time: {elapsed:.1f}s")

    if not results:
        print("\nNo results collected.")
        return

    # --- Summary ---
    n = len(results)
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY ({n} tickers, {horizon_lbl})")
    print(f"{'='*70}")

    # Agreement rates
    llm_rf = sum(1 for r in results if r["llm"] and r["rf"] and r["llm"] == r["rf"])
    llm_knn = sum(1 for r in results if r["llm"] and r["knn"] and r["llm"] == r["knn"])
    rf_knn = sum(1 for r in results if r["rf"] and r["knn"] and r["rf"] == r["knn"])
    all_three = sum(1 for r in results if r["llm"] and r["rf"] and r["knn"] and r["llm"] == r["rf"] == r["knn"])

    print(f"\n--- Exact Agreement Rates ---")
    print(f"  LLM vs RF:      {llm_rf}/{n} ({llm_rf/n:.0%})")
    print(f"  LLM vs KNN:     {llm_knn}/{n} ({llm_knn/n:.0%})")
    print(f"  RF vs KNN:      {rf_knn}/{n} ({rf_knn/n:.0%})")
    print(f"  All three:      {all_three}/{n} ({all_three/n:.0%})")

    # Directional agreement
    def direction(rating):
        if rating is None:
            return None
        s = rating_to_score(parse_rating(rating))
        if s > 0:
            return "bullish"
        if s < 0:
            return "bearish"
        return "neutral"

    llm_rf_dir = sum(1 for r in results if direction(r["llm"]) and direction(r["rf"]) and direction(r["llm"]) == direction(r["rf"]))
    llm_knn_dir = sum(1 for r in results if direction(r["llm"]) and direction(r["knn"]) and direction(r["llm"]) == direction(r["knn"]))
    rf_knn_dir = sum(1 for r in results if direction(r["rf"]) and direction(r["knn"]) and direction(r["rf"]) == direction(r["knn"]))

    print(f"\n--- Directional Agreement (bullish/neutral/bearish) ---")
    print(f"  LLM vs RF:      {llm_rf_dir}/{n} ({llm_rf_dir/n:.0%})")
    print(f"  LLM vs KNN:     {llm_knn_dir}/{n} ({llm_knn_dir/n:.0%})")
    print(f"  RF vs KNN:      {rf_knn_dir}/{n} ({rf_knn_dir/n:.0%})")

    # Vote impact — how often did the vote differ from each model?
    vote_diff_llm = sum(1 for r in results if r["voted"] and r["llm"] and r["voted"] != r["llm"])
    vote_diff_rf = sum(1 for r in results if r["voted"] and r["rf"] and r["voted"] != r["rf"])
    vote_diff_knn = sum(1 for r in results if r["voted"] and r["knn"] and r["voted"] != r["knn"])

    print(f"\n--- Vote Impact (vote differs from individual model) ---")
    print(f"  Vote != LLM:    {vote_diff_llm}/{n} ({vote_diff_llm/n:.0%})")
    print(f"  Vote != RF:     {vote_diff_rf}/{n} ({vote_diff_rf/n:.0%})")
    print(f"  Vote != KNN:    {vote_diff_knn}/{n} ({vote_diff_knn/n:.0%})")

    # Distribution of recommendations
    from collections import Counter
    llm_dist = Counter(r["llm"] for r in results if r["llm"])
    rf_dist = Counter(r["rf"] for r in results if r["rf"])
    knn_dist = Counter(r["knn"] for r in results if r["knn"])
    vote_dist = Counter(r["voted"] for r in results if r["voted"])

    labels = ["strong_buy", "buy", "hold", "sell", "strong_sell"]
    print(f"\n--- Recommendation Distribution ---")
    print(f"  {'Label':<14} {'LLM':>5} {'RF':>5} {'KNN':>5} {'Vote':>5}")
    print(f"  {'-'*36}")
    for l in labels:
        print(f"  {l:<14} {llm_dist.get(l, 0):>5} {rf_dist.get(l, 0):>5} {knn_dist.get(l, 0):>5} {vote_dist.get(l, 0):>5}")

    # Debate behavior
    avg_turns = sum(r["debate_turns"] for r in results) / n
    avg_unresolved = sum(r["unresolved"] for r in results) / n
    avg_time = sum(r["elapsed"] for r in results) / n

    print(f"\n--- Debate Behavior ---")
    print(f"  Avg debate turns:        {avg_turns:.1f}")
    print(f"  Avg unresolved:          {avg_unresolved:.1f}")
    print(f"  Tickers with 0 unresolv: {sum(1 for r in results if r['unresolved'] == 0)}/{n}")
    print(f"  Avg time per ticker:     {avg_time:.1f}s")
    print(f"  Total time:              {sum(r['elapsed'] for r in results):.0f}s")

    # Per-ticker detail table
    print(f"\n--- Per-Ticker Results ---")
    print(f"  {'Ticker':<7} {'LLM':<12} {'RF':<12} {'KNN':<12} {'Vote':<12} {'Debate':>6} {'Unres':>5}")
    print(f"  {'-'*66}")
    for r in results:
        print(f"  {r['ticker']:<7} {(r['llm'] or 'N/A'):<12} {(r['rf'] or 'N/A'):<12} {(r['knn'] or 'N/A'):<12} {(r['voted'] or 'N/A'):<12} {r['debate_turns']:>6} {r['unresolved']:>5}")

    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    h = args[0] if args else "swing"
    run_assessment(h)
