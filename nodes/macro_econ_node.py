from shared.state_schema import BerkshireState


def macro_econ_node(state: BerkshireState):
    """
    Node for macroeconomic analysis.
    """
    # TODO: Replace placeholder with macro indicator scoring.
    return {
        "analyst_signals": {
            "macro": {
                "rating": "hold",
                "confidence": 0.0,
                "features": {},
                "details": "Macro analysis node not implemented yet; defaulting to hold.",
            }
        }
    }
