from typing import TypedDict, Dict, Any, Annotated # Annotated is for attaching the reducer function to a dict key

def merge_signals(existing_signals: dict, new_signals: dict) -> dict:
    """
    Reducer function to merge existing analyst signals with new ones.
    This ensures that agent outputs dont overwrite each other but rather combine to provide a more comprehensive view of the stock.
    """
    merged = existing_signals.copy() if existing_signals else {}
    merged.update(new_signals)
    return merged

def read_only_data(existing_data: dict, new_data: dict) -> dict:
    """
    Only allows data to be written if its currently empty.
    Locks the raw data so agents cannot alter facts.
    """
    if existing_data:
        return existing_data
    return new_data


class BerkshireState(TypedDict):
    """
    State schema for the Berkshire Agent Council.
    """
    # Python type hinting syntax just tells python that a berkshire state object will have an object ticker that holds a string
    ticker: str
    data: Annotated[Dict[str, Any], read_only_data] # The raw data now has read only rule
    analyst_signals: Annotated[Dict[str, Any], merge_signals] # Merge rule to eliminate overwriting
