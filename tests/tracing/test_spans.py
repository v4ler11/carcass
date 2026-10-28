import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from openai.types.chat import ChatCompletionMessageParam
from opentelemetry.sdk.trace import TracerProvider

from chat.complete import complete
from tests.stub_server import StubFactory, completion
from tracing import build_provider, configure, conversation_turn, read_spans
from tracing.attributes import Attr


def _chat(spans: list[dict[str, Any]], model: str) -> dict[str, Any]:
    return next(
        span
        for span in spans
        if span["name"] == "chat"
        and span["attributes"].get("gen_ai.request.model") == model
    )


def test_writes_parent_and_child_to_one_daily_file(
    traces_dir: Path, provider: TracerProvider
) -> None:
    name = f"parent-{uuid4().hex}"
    tracer = provider.get_tracer("test")

    with (
        tracer.start_as_current_span(name),
        tracer.start_as_current_span(f"{name}-child"),
    ):
        pass
    provider.force_flush()

    spans = [span for span in read_spans(traces_dir) if span["name"].startswith(name)]
    parent = next(span for span in spans if span["name"] == name)
    child = next(span for span in spans if span["name"] == f"{name}-child")

    assert child["context"]["trace_id"] == parent["context"]["trace_id"]
    assert child["parent_id"] == parent["context"]["span_id"]
    assert parent["parent_id"] is None

    day = datetime.fromisoformat(parent["start_time"]).astimezone().strftime("%Y%m%d")
    assert (traces_dir / f"{day}.jsonl").exists()


async def test_complete_writes_a_client_span_with_usage(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        return completion(
            model,
            {"role": "assistant", "content": "pong"},
            prompt=7,
            completion_tokens=2,
        )

    client = await stub(handler)
    message: ChatCompletionMessageParam = {"role": "user", "content": "ping"}

    await complete(client=client, model=model, messages=[message])
    provider.force_flush()

    span = _chat(read_spans(traces_dir), model)
    assert span["kind"] == "SpanKind.CLIENT"
    assert span["attributes"]["gen_ai.operation.name"] == "chat"
    assert span["attributes"]["gen_ai.request.model"] == model
    assert span["attributes"]["gen_ai.usage.input_tokens"] == 7
    assert span["attributes"]["gen_ai.usage.output_tokens"] == 2


async def test_span_messages_store_content_out_of_line(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        return completion(
            model,
            {"role": "assistant", "content": "pong"},
            prompt=7,
            completion_tokens=2,
        )

    client = await stub(handler)
    message: ChatCompletionMessageParam = {"role": "user", "content": "ping"}

    await complete(client=client, model=model, messages=[message])
    provider.force_flush()

    lines = [
        json.loads(line)
        for path in traces_dir.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    raw = _chat(lines, model)
    raw_messages = json.loads(raw["attributes"]["gen_ai.input.messages"])
    assert set(raw_messages[0]["content"]) == {"$ref"}

    span = _chat(read_spans(traces_dir), model)
    assert json.loads(span["attributes"]["gen_ai.input.messages"]) == [message]
    assert json.loads(span["attributes"]["gen_ai.output.messages"]) == [
        {"role": "assistant", "content": "pong"}
    ]


async def test_reasoning_fields_store_content_out_of_line(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"
    reply = {
        "role": "assistant",
        "content": "pong",
        "reasoning": "thinking",
        "reasoning_details": [
            {
                "type": "reasoning.text",
                "text": "thinking",
                "signature": "signature",
                "format": "unknown",
                "index": 0,
            }
        ],
    }

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        return completion(model, reply, prompt=7, completion_tokens=2)

    client = await stub(handler)
    message: ChatCompletionMessageParam = {"role": "user", "content": "ping"}

    await complete(client=client, model=model, messages=[message])
    provider.force_flush()

    lines = [
        json.loads(line)
        for path in traces_dir.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    raw = _chat(lines, model)
    stored = json.loads(raw["attributes"]["gen_ai.output.messages"])[0]
    assert set(stored["reasoning"]) == {"$ref"}
    assert set(stored["reasoning_details"][0]["text"]) == {"$ref"}
    assert set(stored["reasoning_details"][0]["signature"]) == {"$ref"}

    span = _chat(read_spans(traces_dir), model)
    resolved = json.loads(span["attributes"]["gen_ai.output.messages"])[0]
    assert resolved["reasoning"] == "thinking"
    assert resolved["reasoning_details"][0]["text"] == "thinking"
    assert resolved["reasoning_details"][0]["signature"] == "signature"


def test_turn_tags_every_span_with_the_conversation_id(
    traces_dir: Path, provider: TracerProvider
) -> None:
    conversation = f"conv-{uuid4().hex}"

    with conversation_turn("invoke_agent", conversation_id=conversation):
        pass
    provider.force_flush()

    span = next(
        span
        for span in read_spans(traces_dir)
        if span["attributes"].get("gen_ai.conversation.id") == conversation
    )
    assert span["attributes"]["gen_ai.conversation.id"] == conversation


def test_turn_carries_user_session_and_metadata(
    traces_dir: Path, provider: TracerProvider
) -> None:
    conversation = f"conv-{uuid4().hex}"

    with (
        conversation_turn(
            "invoke_agent",
            conversation_id=conversation,
            user_id="user-1",
            metadata={"tenant": "acme", "attempt": 2},
        ),
        provider.get_tracer("test").start_as_current_span("child"),
    ):
        pass
    provider.force_flush()

    spans = [
        span
        for span in read_spans(traces_dir)
        if span["attributes"].get("gen_ai.conversation.id") == conversation
    ]
    assert len(spans) == 2
    for span in spans:
        assert span["attributes"]["session.id"] == conversation
        assert span["attributes"]["user.id"] == "user-1"
        assert span["attributes"]["langfuse.trace.metadata.tenant"] == "acme"
        assert span["attributes"]["langfuse.trace.metadata.attempt"] == "2"
        assert span["resource"]["attributes"]["deployment.environment.name"] == "test"


def test_parent_turn_sets_the_conversation(
    traces_dir: Path, provider: TracerProvider
) -> None:
    conversation = f"conv-{uuid4().hex}"

    with conversation_turn("first", conversation_id=conversation) as (_, parent):
        pass
    with conversation_turn("second", parent=parent) as (_, child):
        pass

    assert child.conversation_id == conversation


def test_parent_turn_rejects_a_mismatched_conversation(
    traces_dir: Path, provider: TracerProvider
) -> None:
    conversation = f"conv-{uuid4().hex}"

    with conversation_turn("first", conversation_id=conversation) as (_, parent):
        pass

    with (
        pytest.raises(ValueError, match="belongs to conversation"),
        conversation_turn("second", conversation_id="other", parent=parent),
    ):
        pass


def test_provider_sets_the_environment(tmp_path: Path) -> None:
    built = build_provider(tmp_path, environment="prod")
    try:
        assert built.resource.attributes["deployment.environment.name"] == "prod"
    finally:
        built.shutdown()


async def test_tool_definitions_store_content_out_of_line(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Return the value stored under a key.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                },
            },
        }
    ]

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        return completion(
            model, {"role": "assistant", "content": "ok"}, prompt=1, completion_tokens=1
        )

    client = await stub(handler)
    message: ChatCompletionMessageParam = {"role": "user", "content": "ping"}

    await complete(client=client, model=model, messages=[message], tools=tools)
    provider.force_flush()

    lines = [
        json.loads(line)
        for path in traces_dir.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    raw = _chat(lines, model)
    assert set(json.loads(raw["attributes"]["gen_ai.tool.definitions"])) == {"$ref"}

    span = _chat(read_spans(traces_dir), model)
    assert json.loads(span["attributes"]["gen_ai.tool.definitions"]) == tools


def test_configure_returns_the_provider_already_installed(
    provider: TracerProvider, tmp_path: Path
) -> None:
    assert configure(tmp_path) is provider
    provider.force_flush()
    assert list(tmp_path.glob("*.jsonl")) == []


def test_tool_result_refs_text_and_image_urls(
    traces_dir: Path, provider: TracerProvider
) -> None:
    name = f"tool-{uuid4().hex}"
    result = [
        {
            "role": "tool",
            "tool_call_id": "tool-1",
            "content": [
                {"type": "text", "text": "metadata_parcel.json"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0"},
                },
            ],
        }
    ]

    with provider.get_tracer("test").start_as_current_span(name) as span:
        span.set_attribute(Attr.TOOL_CALL_RESULT, json.dumps(result))
    provider.force_flush()

    raw = next(
        json.loads(line)
        for path in traces_dir.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["name"] == name
    )
    stored = json.loads(raw["attributes"][Attr.TOOL_CALL_RESULT])
    assert set(stored[0]["content"][0]["text"]) == {"$ref"}
    assert set(stored[0]["content"][1]["image_url"]["url"]) == {"$ref"}
    assert "data:image/png" not in raw["attributes"][Attr.TOOL_CALL_RESULT]

    resolved = next(span for span in read_spans(traces_dir) if span["name"] == name)
    assert json.loads(resolved["attributes"][Attr.TOOL_CALL_RESULT]) == result
