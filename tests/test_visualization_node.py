import os

from nodes.visualization_node import visualization_node


def test_visualization_node_exports_summary_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("BERKSHIRE_VIZ_DIR", str(tmp_path))

    state = {
        "ticker": "MSFT",
        "horizon": "swing",
        "analyst_signals": {
            "sentiment": {
                "rating": "buy",
                "confidence": 0.8,
                "revisions": [{"rating": "hold", "confidence": 0.55}],
            },
            "technical": {
                "rating": "hold",
                "confidence": 0.5,
                "revisions": [{"rating": "sell", "confidence": 0.45}],
            },
            "fundamental": {
                "rating": "strong_buy",
                "confidence": 0.9,
                "revisions": [{"rating": "buy", "confidence": 0.65}],
            },
            "macro": {
                "rating": "sell",
                "confidence": 0.4,
                "revisions": [{"rating": "hold", "confidence": 0.35}],
            },
            "classical_models": {
                "rf": {"prediction": "buy", "probabilities": {"buy": 0.62, "hold": 0.25, "sell": 0.13}},
                "knn": {"prediction": "hold", "probabilities": {"buy": 0.33, "hold": 0.44, "sell": 0.23}},
            },
        },
        "final_report": {
            "ticker": "MSFT",
            "horizon_label": "Swing (2-8 weeks)",
            "llm_recommendation": "buy",
            "raw_recommendation": "buy",
            "recommendation": "buy",
            "weighted_score": 0.74,
            "effective_contributors": 4,
            "analyst_breakdown": [
                {"analyst": "sentiment", "score": 1, "confidence": 0.8},
                {"analyst": "technical", "score": 0, "confidence": 0.5},
                {"analyst": "fundamental", "score": 2, "confidence": 0.9},
                {"analyst": "macro", "score": -1, "confidence": 0.4},
            ],
            "classical_models": {
                "rf": {"prediction": "buy"},
                "knn": {"prediction": "hold"},
            },
            "vote": {"consensus": "buy"},
        },
    }

    result = visualization_node(state)
    report = result["final_report"]

    assert "visualizations" in report
    assert isinstance(report["visualizations"], list)
    assert len(report["visualizations"]) >= 1

    summary_items = [
        item for item in report["visualizations"] if item.get("type") == "summary_json"
    ]
    assert summary_items
    summary_path = summary_items[0].get("path")
    assert summary_path
    assert os.path.exists(summary_path)

    artifact_types = {item.get("type") for item in report["visualizations"]}
    assert "diverging_bar_chart" in artifact_types
    assert "debate_shift_chart" in artifact_types
    assert "model_stack_chart" in artifact_types
