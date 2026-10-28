"""A conversation: its metadata, plus a span tree loaded on demand."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from conversations.node import Node, build_tree


@dataclass(slots=True)
class Conversation:
    """What a conversation list can show without reading a single blob.

    The tree is not there yet: call `populate()` first, then read it. That is
    what `get_conversation()` does for you.
    """

    conversation_id: str
    user_id: str | None
    metadata: Mapping[str, Any]
    models: tuple[str, ...]
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    turns: int
    spans: int
    input_tokens: int
    output_tokens: int
    cost: float | None
    trace_ids: tuple[str, ...]
    _spans: list[dict[str, Any]] = field(repr=False, compare=False)
    _blobs: Path = field(repr=False, compare=False)
    _tree: tuple[Node, ...] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def tree(self) -> tuple[Node, ...]:
        """Root spans in start order: one per turn or standalone call, branches included."""
        if self._tree is None:
            raise RuntimeError("call populate() before reading the tree")
        return self._tree

    def populate(self) -> Self:
        """Load the tree, then return this conversation. Repeating it is a no-op."""
        if self._tree is None:
            self._tree = build_tree(self._spans, self._blobs)
        return self
