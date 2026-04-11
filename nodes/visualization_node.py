import json
import os
from datetime import datetime

from shared.stance import parse_rating, rating_to_score
from shared.state_schema import BerkshireState


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value or "").strip())
    return cleaned or "unknown"


def _output_dir() -> str:
    configured = os.getenv("BERKSHIRE_VIZ_DIR", "")
    if configured:
        base = configured
    else:
        base = os.path.join(os.getcwd(), "outputs", "visualizations")
    os.makedirs(base, exist_ok=True)
    return base


def _build_breakdown(final_report: dict, analyst_signals: dict) -> list:
    breakdown = final_report.get("analyst_breakdown", [])
    if isinstance(breakdown, list) and breakdown:
        return breakdown

    rows = []
    for analyst, payload in (analyst_signals or {}).items():
        if not isinstance(payload, dict):
            continue
        rating = parse_rating(payload.get("rating"))
        if rating is None:
            continue
        confidence = payload.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))
        rows.append(
            {
                "analyst": analyst,
                "rating": rating.value,
                "confidence": confidence,
                "analyst_weight": 1.0,
                "weighted_confidence": confidence,
                "score": rating_to_score(rating),
            }
        )
    return rows


def _parse_signal_row(analyst: str, payload: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None
    rating = parse_rating(payload.get("rating"))
    if rating is None:
        return None
    confidence = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))
    return {
        "analyst": analyst,
        "rating": rating.value,
        "confidence": confidence,
        "score": rating_to_score(rating),
    }


def _build_debate_progression(analyst_signals: dict) -> list:
    rows = []
    for analyst, payload in (analyst_signals or {}).items():
        if analyst == "classical_models":
            continue
        if not isinstance(payload, dict):
            continue

        current_row = _parse_signal_row(analyst, payload)
        if current_row is None:
            continue

        revisions = payload.get("revisions", [])
        before_payload = revisions[0] if isinstance(revisions, list) and revisions else payload
        before_row = _parse_signal_row(analyst, before_payload)
        if before_row is None:
            before_row = current_row.copy()

        rows.append(
            {
                "analyst": analyst,
                "before_rating": before_row["rating"],
                "before_score": before_row["score"],
                "before_confidence": before_row["confidence"],
                "after_rating": current_row["rating"],
                "after_score": current_row["score"],
                "after_confidence": current_row["confidence"],
            }
        )

    rows.sort(key=lambda item: item["analyst"])
    return rows


def _build_model_comparison(final_report: dict) -> list:
    classical = final_report.get("classical_models", {}) if isinstance(final_report, dict) else {}
    vote = final_report.get("vote", {}) if isinstance(final_report, dict) else {}

    models = [
        ("LLM Debate", final_report.get("llm_recommendation")),
        ("Random Forest", (classical.get("rf") or {}).get("prediction") if isinstance(classical, dict) else None),
        ("KNN", (classical.get("knn") or {}).get("prediction") if isinstance(classical, dict) else None),
        ("Final Vote", final_report.get("recommendation") or vote.get("consensus")),
    ]

    rows = []
    for label, rating_value in models:
        rating = parse_rating(rating_value)
        if rating is None:
            continue
        rows.append(
            {
                "label": label,
                "rating": rating.value,
                "score": rating_to_score(rating),
            }
        )
    return rows


def _write_json_summary(output_path: str, payload: dict) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_contribution_chart(output_path: str, breakdown: list, weighted_score: float, ticker: str, horizon: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for item in breakdown:
        rows.append(
            {
                "name": str(item.get("analyst", "unknown")).title(),
                "score": float(item.get("score", 0.0)),
                "confidence": float(item.get("confidence", 0.0)),
            }
        )
    rows.sort(key=lambda x: x["score"])

    names = [row["name"] for row in rows]
    stance_scores = [row["score"] for row in rows]
    confidences = [row["confidence"] for row in rows]

    colors = []
    for score, conf in zip(stance_scores, confidences):
        alpha = 0.25 + (0.75 * max(0.0, min(conf, 1.0)))
        if score >= 0:
            colors.append((0.12, 0.47, 0.71, alpha))
        else:
            colors.append((0.84, 0.15, 0.16, alpha))

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.barh(names, stance_scores, color=colors, edgecolor="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.axvline(weighted_score, color="#2ca02c", linestyle="--", linewidth=1.2)
    ax.text(
        0.01,
        0.96,
        f"Final weighted score: {weighted_score:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#2ca02c", "boxstyle": "round,pad=0.2"},
    )
    ax.set_xlim(-2.2, 2.2)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("Stance Score (-2 to +2)")
    ax.set_title(f"{ticker} Council Diverging Contributions ({horizon})")
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_debate_progress_chart(output_path: str, progression: list, ticker: str, horizon: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [str(item.get("analyst", "unknown")).title() for item in progression]
    if not labels:
        return

    before_scores = [float(item.get("before_score", 0.0)) for item in progression]
    after_scores = [float(item.get("after_score", 0.0)) for item in progression]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    before_cycle = before_scores + before_scores[:1]
    after_cycle = after_scores + after_scores[:1]
    angles_cycle = angles + angles[:1]

    fig = plt.figure(figsize=(7.0, 6.8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_cycle, before_cycle, linewidth=2.0, color="#ff7f0e", label="Before debate")
    ax.fill(angles_cycle, before_cycle, color="#ff7f0e", alpha=0.15)
    ax.plot(angles_cycle, after_cycle, linewidth=2.0, color="#1f77b4", label="After debate")
    ax.fill(angles_cycle, after_cycle, color="#1f77b4", alpha=0.20)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_ylim(-2, 2)
    ax.set_title(f"{ticker} Council Debate Shift ({horizon})", va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.10))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_model_comparison_chart(output_path: str, final_report: dict, ticker: str, horizon: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _build_model_comparison(final_report)
    if not rows:
        return

    labels = [row["label"] for row in rows]
    scores = [float(row["score"]) for row in rows]

    colors = []
    for score in scores:
        if score > 0:
            colors.append((0.12, 0.47, 0.71, 0.85))
        elif score < 0:
            colors.append((0.84, 0.15, 0.16, 0.85))
        else:
            colors.append((0.50, 0.50, 0.50, 0.85))

    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    ax.barh(labels, scores, color=colors, edgecolor="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_xlim(-2.2, 2.2)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("Stance Score (-2 to +2)")
    ax.set_title(f"{ticker} Final Decision Stack ({horizon})")
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    for idx, row in enumerate(rows):
        ax.text(
            row["score"] + (0.08 if row["score"] >= 0 else -0.08),
            idx,
            row["rating"],
            va="center",
            ha="left" if row["score"] >= 0 else "right",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def visualization_node(state: BerkshireState):
    """
    Create report-ready visualization artifacts from final synthesis output.

    Always exports a JSON summary and attempts to export PNG charts if
    matplotlib is available in the environment.
    """
    final_report = state.get("final_report", {}) or {}
    analyst_signals = state.get("analyst_signals", {}) or {}

    ticker = str(final_report.get("ticker") or state.get("ticker") or "UNKNOWN")
    horizon_label = str(final_report.get("horizon_label") or state.get("horizon") or "swing")
    weighted_score = float(final_report.get("weighted_score", 0.0) or 0.0)
    recommendation = str(final_report.get("recommendation", "hold"))
    breakdown = _build_breakdown(final_report, analyst_signals)
    debate_progression = _build_debate_progression(analyst_signals)

    out_dir = _output_dir()
    run_tag = f"{_safe_name(ticker)}_{_safe_name(horizon_label)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    artifacts = []
    summary_payload = {
        "ticker": ticker,
        "horizon": horizon_label,
        "recommendation": recommendation,
        "weighted_score": weighted_score,
        "effective_contributors": final_report.get("effective_contributors"),
        "coverage_warning": final_report.get("coverage_warning", ""),
        "highest_unresolved_severity": final_report.get("highest_unresolved_severity", 0.0),
        "analyst_breakdown": breakdown,
        "debate_progression": debate_progression,
        "classical_models": final_report.get("classical_models", {}),
        "llm_recommendation": final_report.get("llm_recommendation", ""),
        "raw_recommendation": final_report.get("raw_recommendation", ""),
    }

    json_path = os.path.join(out_dir, f"{run_tag}_summary.json")
    _write_json_summary(json_path, summary_payload)
    artifacts.append({"type": "summary_json", "path": json_path})

    chart_error = ""
    if breakdown:
        contribution_chart_path = os.path.join(out_dir, f"{run_tag}_diverging_bar.png")
        debate_shift_chart_path = os.path.join(out_dir, f"{run_tag}_debate_shift.png")
        model_stack_chart_path = os.path.join(out_dir, f"{run_tag}_model_stack.png")
        try:
            _save_contribution_chart(
                contribution_chart_path,
                breakdown,
                weighted_score,
                ticker,
                horizon_label,
            )
            artifacts.append({"type": "diverging_bar_chart", "path": contribution_chart_path})

            if debate_progression:
                _save_debate_progress_chart(
                    debate_shift_chart_path,
                    debate_progression,
                    ticker,
                    horizon_label,
                )
                artifacts.append({"type": "debate_shift_chart", "path": debate_shift_chart_path})

            _save_model_comparison_chart(
                model_stack_chart_path,
                final_report,
                ticker,
                horizon_label,
            )
            artifacts.append({"type": "model_stack_chart", "path": model_stack_chart_path})
        except Exception as exc:
            chart_error = str(exc)

    updates = {
        "visualizations": artifacts,
        "visualization_dir": out_dir,
    }
    if chart_error:
        updates["visualization_note"] = (
            "Summary exported; chart export failed. "
            f"Install matplotlib to enable PNG chart generation. Error: {chart_error}"
        )

    return {"final_report": updates}
