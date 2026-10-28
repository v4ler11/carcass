"""Build conversations from the spans written under a trace directory."""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from conversations.conversation import Conversation
from core.config import TRACES_DIR
from tracing.attributes import Attr
from tracing.exporter import read_spans


def list_conversations(directory: Path = TRACES_DIR) -> list[Conversation]:
    """Every conversation found under `directory`, most recent first, trees not loaded."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in read_spans(directory, resolve=False):
        conversation_id = span["attributes"].get(Attr.CONVERSATION_ID)
        if isinstance(conversation_id, str):
            grouped[conversation_id].append(span)

    conversations = [
        _conversation(conversation_id, spans, directory)
        for conversation_id, spans in grouped.items()
    ]
    return sorted(conversations, key=lambda item: item.started_at, reverse=True)


def get_conversation(conversation_id: str, *, directory: Path = TRACES_DIR) -> Conversation:
    """One conversation, already populated: metadata and tree are both available."""
    spans = [
        span
        for span in read_spans(directory, resolve=False)
        if span["attributes"].get(Attr.CONVERSATION_ID) == conversation_id
    ]
    if not spans:
        raise KeyError(f"no spans for conversation {conversation_id!r}")
    return _conversation(conversation_id, spans, directory).populate()


def _conversation(
    conversation_id: str, spans: list[dict[str, Any]], directory: Path
) -> Conversation:
    ordered = sorted(spans, key=lambda span: span["start_time"])
    attributes = [span["attributes"] for span in ordered]

    starts = [datetime.fromisoformat(span["start_time"]) for span in ordered]
    ends = [datetime.fromisoformat(span["end_time"]) for span in ordered]

    metadata: dict[str, Any] = {}
    for values in attributes:
        for key, value in values.items():
            if key.startswith(Attr.LANGFUSE_METADATA_PREFIX):
                metadata[key.removeprefix(Attr.LANGFUSE_METADATA_PREFIX)] = _as_json(value)

    cost: float | None = None
    for values in attributes:
        raw = values.get(Attr.LANGFUSE_COST_DETAILS)
        total = _as_json(raw).get("total") if isinstance(raw, str) else None
        if isinstance(total, int | float):
            cost = (cost or 0.0) + total

    return Conversation(
        conversation_id=conversation_id,
        user_id=next((values[Attr.USER_ID] for values in attributes if Attr.USER_ID in values), None),
        metadata=metadata,
        models=tuple(
            dict.fromkeys(
                values[Attr.REQUEST_MODEL]
                for values in attributes
                if Attr.REQUEST_MODEL in values
            )
        ),
        started_at=min(starts),
        ended_at=max(ends),
        duration_ms=(max(ends) - min(starts)).total_seconds() * 1000,
        turns=sum(1 for span in ordered if span["name"] == "invoke_agent"),
        spans=len(ordered),
        input_tokens=sum(values.get(Attr.USAGE_INPUT_TOKENS, 0) for values in attributes),
        output_tokens=sum(values.get(Attr.USAGE_OUTPUT_TOKENS, 0) for values in attributes),
        cost=cost,
        trace_ids=tuple(dict.fromkeys(span["context"]["trace_id"] for span in ordered)),
        _spans=ordered,
        _blobs=directory / "blobs_traces.db",
    )


def _as_json(value: str) -> Any:
    """Metadata values are stored as JSON strings; plain strings stay strings."""
    try:
        return json.loads(value)
    except ValueError:
        return value
