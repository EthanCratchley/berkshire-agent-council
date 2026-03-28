from shared.state_schema import BerkshireState
from shared.horizon import normalize_horizon, horizon_label


def technical_node(state: BerkshireState):
    """
    Node for technical analysis of stock data.
    """
    selected_horizon = normalize_horizon(state.get("horizon", "swing"))
    # TODO: Replace placeholder with actual technical indicators + scoring.
    return {
        "analyst_signals": {
            "technical": {
                "rating": "hold",
                "confidence": 0.0,
                "features": {},
                "details": (
                    "Technical analysis node not implemented yet; defaulting to hold. "
                    f"Horizon: {selected_horizon}."
                ),
                "horizon_alignment_note": (
                    f"Technical signal unavailable for {horizon_label(selected_horizon)}."
                ),
            }
        }
    }
