from typing import Tuple

VALID_HORIZONS = {"short", "swing", "long"}

HORIZON_LABELS = {
    "short": "Short-term (1-10 trading days)",
    "swing": "Swing (2-8 weeks)",
    "long": "Long-term (6-24 months)",
}

HORIZON_DAY_RANGES = {
    "short": (1, 10),
    "swing": (14, 56),
    "long": (180, 730),
}

HORIZON_ANALYST_WEIGHTS = {
    "short": {
        "sentiment": 1.35,
        "technical": 1.25,
        "fundamental": 0.75,
        "macro": 0.75,
    },
    "swing": {
        "sentiment": 1.0,
        "technical": 1.0,
        "fundamental": 1.0,
        "macro": 1.0,
    },
    "long": {
        "sentiment": 0.75,
        "technical": 0.75,
        "fundamental": 1.35,
        "macro": 1.25,
    },
}


def normalize_horizon(value) -> str:
    if not isinstance(value, str):
        return "swing"
    cleaned = value.strip().lower()
    aliases = {
        "short_term": "short",
        "short-term": "short",
        "shortterm": "short",
        "swing_trade": "swing",
        "swing-trade": "swing",
        "long_term": "long",
        "long-term": "long",
        "longterm": "long",
    }
    cleaned = aliases.get(cleaned, cleaned)
    return cleaned if cleaned in VALID_HORIZONS else "swing"


def horizon_label(horizon: str) -> str:
    normalized = normalize_horizon(horizon)
    return HORIZON_LABELS.get(normalized, HORIZON_LABELS["swing"])


def horizon_day_range(horizon: str) -> Tuple[int, int]:
    normalized = normalize_horizon(horizon)
    return HORIZON_DAY_RANGES.get(normalized, HORIZON_DAY_RANGES["swing"])


def analyst_weights_for_horizon(horizon: str) -> dict:
    normalized = normalize_horizon(horizon)
    return HORIZON_ANALYST_WEIGHTS.get(normalized, HORIZON_ANALYST_WEIGHTS["swing"]).copy()
