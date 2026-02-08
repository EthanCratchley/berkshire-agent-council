from typing import TypedDict, Dict, Any

class BerkshireState(TypedDict):
    """
    State schema for the Berkshire Agent Council.
    """
    ticker: str
    data: Dict[str, Any]
    analyst_signals: Dict[str, Any]
