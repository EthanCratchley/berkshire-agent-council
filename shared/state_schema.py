from typing import TypedDict, Dict, Any, Annotated  # Annotated attaches reducer functions to state keys

def merge_signals(existing_signals: dict, new_signals: dict) -> dict:
    """
    Reducer function to merge existing analyst signals with new ones.
    This ensures that agent outputs dont overwrite each other but rather combine to provide a more comprehensive view of the stock.
    """
    merged = existing_signals.copy() if existing_signals else {}

    if not new_signals:
        return merged

    for analyst, incoming_payload in new_signals.items():
        prior_payload = merged.get(analyst)

        if isinstance(prior_payload, dict) and isinstance(incoming_payload, dict):
            prior_snapshot = {k: v for k, v in prior_payload.items() if k != "revisions"}
            incoming_snapshot = {k: v for k, v in incoming_payload.items() if k != "revisions"}

            prior_revisions = prior_payload.get("revisions", [])
            if not isinstance(prior_revisions, list):
                prior_revisions = []

            incoming_revisions = incoming_payload.get("revisions", [])
            if not isinstance(incoming_revisions, list):
                incoming_revisions = []

            all_revisions = list(prior_revisions)

            # Preserve the previous version before replacing, unless payload is unchanged.
            if prior_snapshot and prior_snapshot != incoming_snapshot:
                all_revisions.append(prior_snapshot)

            all_revisions.extend(incoming_revisions)

            merged_payload = incoming_payload.copy()
            if all_revisions:
                merged_payload["revisions"] = all_revisions

            merged[analyst] = merged_payload
        else:
            merged[analyst] = incoming_payload

    return merged

def read_only_data(existing_data: dict, new_data: dict) -> dict:
    """
    Only allows data to be written if its currently empty.
    Locks the raw data so agents cannot alter facts.
    """
    if existing_data:
        return existing_data
    return new_data


def make_initial_debate_state(max_rounds: int = 3) -> dict:
    """
    Build the initial debate metadata used by the orchestrator loop.
    """
    return {
        "round": 0,
        "max_rounds": max_rounds,
        "queue": [],
        "active_challenge": None,
        "awaiting_response_from": None,
        "next_node": None,
        "history": [],
        "unresolved_contradictions": [],
        "closed_pairs": [],
        "status": "not_started",
    }


def merge_debate(existing_debate: dict, new_debate: dict) -> dict:
    """
    Reducer that protects debate metadata from accidental wipeout.

    Default behavior:
    - scalar fields overwrite when provided
    - list fields append when provided

    Optional control:
    - pass "_replace_lists": ["queue", "history", "unresolved_contradictions", "closed_pairs"]
      in new_debate to replace those lists instead of appending.
    """
    existing_debate = existing_debate or {}
    new_debate = new_debate or {}

    merged = make_initial_debate_state(
        max_rounds=existing_debate.get("max_rounds", new_debate.get("max_rounds", 3))
    )

    # Seed from existing debate state first.
    for key, value in existing_debate.items():
        if key in ("queue", "history", "unresolved_contradictions", "closed_pairs"):
            merged[key] = list(value) if isinstance(value, list) else []
        else:
            merged[key] = value

    replace_lists = new_debate.get("_replace_lists", [])
    if not isinstance(replace_lists, list):
        replace_lists = []
    replace_lists = set(replace_lists)

    # Merge scalar fields.
    for key in ("round", "max_rounds", "active_challenge", "awaiting_response_from", "next_node", "status"):
        if key in new_debate:
            merged[key] = new_debate[key]

    # Merge list fields.
    for key in ("queue", "history", "unresolved_contradictions", "closed_pairs"):
        if key not in new_debate:
            continue

        incoming = new_debate.get(key, [])
        if not isinstance(incoming, list):
            continue

        if key in replace_lists:
            merged[key] = list(incoming)
        else:
            current = merged.get(key, [])
            if not isinstance(current, list):
                current = []
            merged[key] = current + incoming

    return merged


def merge_dict(existing_value: dict, new_value: dict) -> dict:
    """
    Simple dict merge reducer used for low-risk state buckets (e.g., final_report).
    """
    merged = existing_value.copy() if existing_value else {}
    if new_value:
        merged.update(new_value)
    return merged


class BerkshireState(TypedDict):
    """
    State schema for the Berkshire Agent Council.
    """
    # Python type hinting syntax just tells python that a berkshire state object will have an object ticker that holds a string
    ticker: str
    data: Annotated[Dict[str, Any], read_only_data] # The raw data now has read only rule
    analyst_signals: Annotated[Dict[str, Any], merge_signals] # Merge rule to eliminate overwriting
    debate: Annotated[Dict[str, Any], merge_debate]
    final_report: Annotated[Dict[str, Any], merge_dict]
