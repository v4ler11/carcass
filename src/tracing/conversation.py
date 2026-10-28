"""Conversation context: the turn graph and the processor that tags spans."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Self
from uuid import uuid4

from opentelemetry import context, trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import (
    Link,
    SpanContext,
    SpanKind,
    format_span_id,
    format_trace_id,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from tracing.attributes import Attr, attribute_value

tracer = trace.get_tracer(Attr.SERVICE_NAME)


class ConversationSpanProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        conversation_id = context.get_value(Attr.CONVERSATION_ID, parent_context)
        if isinstance(conversation_id, str):
            span.set_attribute(Attr.CONVERSATION_ID, conversation_id)
            span.set_attribute(Attr.SESSION_ID, conversation_id)

        trace_attributes = context.get_value(Attr.TRACE_ATTRIBUTES, parent_context)
        if isinstance(trace_attributes, dict):
            for key, value in trace_attributes.items():
                span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


@dataclass(frozen=True)
class Turn:
    """A serializable turn edge: its conversation and its span's traceparent.

    `conversation_turn(parent=turn)` records the same edge twice: as an OTel
    `Link` for OTel-native readers, and as `gen_ai.conversation.parent_*`
    attributes for readers scanning the JSONL. Both name the same span.
    """

    conversation_id: str
    traceparent: str

    @classmethod
    def of(cls, conversation_id: str, span: trace.Span) -> Self:
        """Name the span a call just ran in, for the next call to continue."""
        span_context = span.get_span_context()
        flags = f"{int(span_context.trace_flags):02x}"
        return cls(
            conversation_id=conversation_id,
            traceparent=(
                f"00-{format_trace_id(span_context.trace_id)}-"
                f"{format_span_id(span_context.span_id)}-{flags}"
            ),
        )


def parse_traceparent(value: str) -> SpanContext | None:
    """The span a valid `traceparent` names; None if the header is malformed."""
    extracted = TraceContextTextMapPropagator().extract({"traceparent": value})
    span_context = trace.get_current_span(extracted).get_span_context()
    return span_context if span_context.is_valid else None


@contextmanager
def conversation_turn(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[tuple[trace.Span, Turn]]:
    """Start the root span of one conversation turn: a node in the conversation graph.

    The span is tagged with `gen_ai.conversation.id` (inherited from `parent`
    or the ambient turn, generated otherwise) and links to `parent` when given,
    so turns in different traces are still connected. Two turns passing the same
    `parent` are siblings: the conversation branches there.

    The edge is recorded twice: as an OTel `Link`, which OTel-native readers
    follow, and as `gen_ai.conversation.parent_trace_id` / `parent_span_id`,
    which readers scanning the JSONL can query. Both name the same span.

    The id is also attached to the ambient context, so every span started inside
    the turn (`chat`, `execute_tool`) carries it: selection is per span, not per
    trace, so a trace that serves two conversations cannot mix them. A turn
    started inside another turn (a decision called as a tool, a sub-agent)
    inherits the enclosing conversation when the caller names none.
    """
    if parent is not None:
        if conversation_id is not None and conversation_id != parent.conversation_id:
            raise ValueError(
                f"parent belongs to conversation {parent.conversation_id!r}, "
                f"not {conversation_id!r}"
            )
        conversation_id = parent.conversation_id

    if conversation_id is None:
        ambient = context.get_value(Attr.CONVERSATION_ID)
        conversation_id = ambient if isinstance(ambient, str) else uuid4().hex

    parent_context = (
        parse_traceparent(parent.traceparent) if parent is not None else None
    )
    if parent is not None and parent_context is None:
        raise ValueError(f"invalid traceparent: {parent.traceparent!r}")

    inherited = context.get_value(Attr.TRACE_ATTRIBUTES)
    trace_attributes: dict[str, Any] = (
        dict(inherited) if isinstance(inherited, dict) else {}
    )
    if user_id is not None:
        trace_attributes[Attr.USER_ID] = user_id
    for key, value in (metadata or {}).items():
        trace_attributes[f"{Attr.LANGFUSE_METADATA_PREFIX}{key}"] = attribute_value(
            value
        )

    turn_attributes: dict[str, Any] = {
        **trace_attributes,
        **(attributes or {}),
        Attr.CONVERSATION_ID: conversation_id,
    }
    if parent_context is not None:
        turn_attributes[Attr.PARENT_TRACE_ID] = format_trace_id(parent_context.trace_id)
        turn_attributes[Attr.PARENT_SPAN_ID] = format_span_id(parent_context.span_id)

    ambient_context = context.set_value(Attr.CONVERSATION_ID, conversation_id)
    ambient_context = context.set_value(
        Attr.TRACE_ATTRIBUTES, trace_attributes, ambient_context
    )
    token = context.attach(ambient_context)
    try:
        links = [Link(parent_context)] if parent_context is not None else None
        with tracer.start_as_current_span(
            name, kind=kind, attributes=turn_attributes, links=links
        ) as span:
            yield span, Turn.of(conversation_id, span)
    finally:
        context.detach(token)
