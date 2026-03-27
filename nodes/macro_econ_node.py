from shared.state_schema import BerkshireState
from shared.horizon import normalize_horizon, horizon_label


def macro_econ_node(state: BerkshireState):
    """
    Node for macroeconomic analysis.
    """
    selected_horizon = normalize_horizon(state.get("horizon", "swing"))
    # TODO: Replace placeholder with macro indicator scoring.
    return {
        "analyst_signals": {
            "macro": {
                "rating": "hold",
                "confidence": 0.0,
                "features": {},
                "details": (
                    "Macro analysis node not implemented yet; defaulting to hold. "
                    f"Horizon: {selected_horizon}."
                ),
                "horizon_alignment_note": (
                    f"Macro signal unavailable for {horizon_label(selected_horizon)}."
                ),
            }
        }
    }
