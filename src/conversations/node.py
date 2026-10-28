"""A span in a conversation tree; content refs resolve only on demand."""

import json
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from tracing.db.content import CONTENT_KEYS, REF_KEY, ContentStore, resolve_content


@dataclass(slots=True)
class Node:
    """One span of a conversation tree; its content refs resolve on `populate()`."""

    name: str
    span_id: str
    trace_id: str
    parent_id: str | None
    start_time: str
    end_time: str
    attributes: Mapping[str, Any]
    _blobs: Path = field(repr=False, compare=False)
    children: tuple[Self, ...] = ()

    @property
    def duration_ms(self) -> float:
        end = datetime.fromisoformat(self.end_time)
        return (end - datetime.fromisoformat(self.start_time)).total_seconds() * 1000

    def populate(self) -> Self:
        """Resolve every content ref in this node and its descendants, in place.

        Message and tool-definition attributes become parsed JSON; the other
        attributes are copied unchanged. Returns this node, so both
        `leaf.populate()` and `leaf = leaf.populate()` work. Repeating it is a
        no-op: resolved values are no longer strings and are kept as they are.

        A turn root usually holds no messages of its own - its nested `chat`
        nodes do - so populating the root resolves the whole subtree.
        """
        nodes = list(self._walk())
        if any(_has_refs(node.attributes) for node in nodes):
            with ContentStore(self._blobs) as store:
                for node in nodes:
                    node.attributes = _decoded(node.attributes, node.name, store)
        else:
            for node in nodes:
                node.attributes = _decoded(node.attributes, node.name, None)
        return self

    def _walk(self) -> Iterator[Self]:
        yield self
        for child in self.children:
            yield from child._walk()


def build_tree(spans: list[dict[str, Any]], blobs: Path) -> tuple[Node, ...]:
    """Root spans of one conversation, children nested by `parent_id`."""
    ordered = sorted(spans, key=lambda span: span["start_time"])
    by_id = {(span["context"]["trace_id"], span["context"]["span_id"]): span for span in ordered}

    roots: list[dict[str, Any]] = []
    children: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for span in ordered:
        parent_id = span["parent_id"]
        trace_id = span["context"]["trace_id"]
        if parent_id is not None and (trace_id, parent_id) in by_id:
            children[(trace_id, parent_id)].append(span)
        else:
            roots.append(span)

    def build(span: dict[str, Any]) -> Node:
        key = (span["context"]["trace_id"], span["context"]["span_id"])
        return Node(
            name=span["name"],
            span_id=span["context"]["span_id"],
            trace_id=span["context"]["trace_id"],
            parent_id=span["parent_id"],
            start_time=span["start_time"],
            end_time=span["end_time"],
            attributes=span["attributes"],
            children=tuple(build(child) for child in children.get(key, ())),
            _blobs=blobs,
        )

    return tuple(build(span) for span in roots)


def _has_refs(attributes: Mapping[str, Any]) -> bool:
    return any(
        isinstance(value := attributes.get(key), str) and REF_KEY in value
        for key in CONTENT_KEYS
    )


def _decoded(
    attributes: Mapping[str, Any], span_name: str, store: ContentStore | None
) -> dict[str, Any]:
    decoded = dict(attributes)
    for key in CONTENT_KEYS:
        raw = decoded.get(key)
        if not isinstance(raw, str):
            continue
        if store is not None:
            raw = resolve_content(key, raw, store, span_name=span_name)
        decoded[key] = json.loads(raw)
    return decoded
