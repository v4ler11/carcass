import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanContext
from pydantic import BaseModel

from chat.agent import run_agent, run_agent_structured
from chat.structured import complete_structured
from chat.tools.base import Tool
from chat.tools.context import ToolContext
from tests.stub_server import StubFactory, completion
from tracing import parse_traceparent, read_spans


def _conversation(
    spans: list[dict[str, Any]], conversation_id: str
) -> list[dict[str, Any]]:
    return [
        span
        for span in spans
        if span["attributes"].get("gen_ai.conversation.id") == conversation_id
    ]


def _assert_parent_edge(span: dict[str, Any], parent: SpanContext) -> None:
    """The link and the parent attributes must name the same span."""
    assert span["attributes"]["gen_ai.conversation.parent_trace_id"] == (
        f"{parent.trace_id:032x}"
    )
    assert span["attributes"]["gen_ai.conversation.parent_span_id"] == (
        f"{parent.span_id:016x}"
    )
    assert span["links"]
    link = span["links"][0]["context"]
    assert link["trace_id"] == f"0x{parent.trace_id:032x}"
    assert link["span_id"] == f"0x{parent.span_id:016x}"


class Capital(BaseModel):
    country: str
    city: str


class Lookup(Tool):
    @property
    def name(self) -> str:
        return "lookup"

    async def execute(self, ctx, tool_call, args) -> list[ChatCompletionMessageParam]:
        return [{"role": "tool", "tool_call_id": tool_call.id, "content": "42"}]

    def into_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Return the number stored under a key.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                },
            },
        }


async def test_agent_run_records_one_trace(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"
    conversation_id = uuid4().hex
    requests: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if len(requests) == 1:
            return completion(
                model,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"key": "answer"}',
                            },
                        }
                    ],
                },
                prompt=10,
                completion_tokens=2,
            )
        return completion(
            model,
            {"role": "assistant", "content": "42"},
            prompt=20,
            completion_tokens=3,
        )

    client = await stub(handler)
    message: ChatCompletionMessageParam = {
        "role": "user",
        "content": "look up the answer",
    }

    _response, usage, _transcript, _turn = await run_agent(
        ToolContext(),
        client=client,
        model=model,
        messages=[message],
        tools=[Lookup()],
        conversation_id=conversation_id,
        user_id="user-1",
        metadata={"tenant": "acme", "attempt": 1},
    )
    provider.force_flush()

    spans = read_spans(traces_dir)
    selected = _conversation(spans, conversation_id)
    root = next(span for span in selected if span["name"] == "invoke_agent")
    chats = [span for span in selected if span["name"] == "chat"]
    tools = [span for span in selected if span["name"] == "execute_tool"]

    assert len(chats) == 2
    assert len(tools) == 1
    assert root["attributes"]["gen_ai.conversation.id"] == conversation_id
    for span in selected:
        assert span["attributes"]["session.id"] == conversation_id
        assert span["attributes"]["user.id"] == "user-1"
        assert span["attributes"]["langfuse.trace.metadata.tenant"] == "acme"
        assert span["attributes"]["langfuse.trace.metadata.attempt"] == "1"
    assert tools[0]["attributes"]["gen_ai.tool.call.arguments"] == '{"key": "answer"}'
    assert (
        json.loads(tools[0]["attributes"]["gen_ai.tool.call.result"])[0]["content"]
        == "42"
    )
    assert all(chat["parent_id"] == root["context"]["span_id"] for chat in chats)
    assert tools[0]["parent_id"] == root["context"]["span_id"]
    assert {span["context"]["trace_id"] for span in selected} == {
        root["context"]["trace_id"]
    }

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        30,
        5,
        35,
    )
    assert [chat["attributes"]["gen_ai.usage.input_tokens"] for chat in chats] == [
        10,
        20,
    ]
    assert root["start_time"] is not None and root["end_time"] is not None


async def test_two_runs_with_one_id_form_one_conversation(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"
    conversation_id = uuid4().hex
    seen = {"count": 0}

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        seen["count"] += 1
        return completion(
            model,
            {"role": "assistant", "content": f"answer {seen['count']}"},
            prompt=5,
            completion_tokens=1,
        )

    client = await stub(handler)
    first: ChatCompletionMessageParam = {"role": "user", "content": "first"}
    second: ChatCompletionMessageParam = {"role": "user", "content": "second"}

    _response, _usage, transcript, turn = await run_agent(
        ToolContext(),
        client=client,
        model=model,
        messages=[first],
        conversation_id=conversation_id,
    )
    await run_agent(
        ToolContext(),
        client=client,
        model=model,
        messages=[*transcript, second],
        conversation_id=conversation_id,
        parent=turn,
    )
    provider.force_flush()

    spans = read_spans(traces_dir)
    first_context = parse_traceparent(turn.traceparent)
    assert first_context is not None
    second_root = next(
        span
        for span in spans
        if span["attributes"].get("gen_ai.conversation.parent_trace_id")
        == f"{first_context.trace_id:032x}"
    )
    assert second_root["name"] == "invoke_agent"
    assert (
        second_root["attributes"]["gen_ai.conversation.parent_span_id"]
        == f"{first_context.span_id:016x}"
    )
    assert second_root.get("links"), "the second turn must link to the first"

    assert second_root["attributes"]["gen_ai.conversation.id"] == conversation_id
    _assert_parent_edge(second_root, first_context)

    roots = [
        span
        for span in _conversation(spans, conversation_id)
        if span["name"] == "invoke_agent"
    ]
    assert len(roots) == 2
    assert len({span["context"]["trace_id"] for span in roots}) == 2


async def test_structured_answer_joins_the_agent_conversation(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"
    conversation_id = uuid4().hex
    requests: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if len(requests) == 1:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"key": "answer"}',
                        },
                    }
                ],
            }
        elif "response_format" in body:
            message = {
                "role": "assistant",
                "content": '{"country": "France", "city": "Paris"}',
            }
        else:
            message = {"role": "assistant", "content": "42"}
        return completion(model, message, prompt=10, completion_tokens=2)

    client = await stub(handler)
    message: ChatCompletionMessageParam = {
        "role": "user",
        "content": "What is the capital of France?",
    }

    answer, usage, transcript, turn = await run_agent(
        ToolContext(),
        client=client,
        model=model,
        messages=[message],
        tools=[Lookup()],
        conversation_id=conversation_id,
    )
    (
        parsed,
        structured_response,
        structured_usage,
        _structured_turn,
    ) = await complete_structured(
        client=client,
        model=model,
        messages=transcript,
        response_model=Capital,
        conversation_id=conversation_id,
        parent=turn,
    )
    structured_answer: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": structured_response.choices[0].message.content or "",
    }
    transcript.append(structured_answer)
    provider.force_flush()

    assert answer.choices[0].message.content == "42"
    assert parsed == Capital(country="France", city="Paris")
    assert transcript[-1].get("content") == '{"country": "France", "city": "Paris"}'
    assert usage.prompt_tokens == 20
    assert structured_usage.prompt_tokens == 10

    spans = _conversation(read_spans(traces_dir), conversation_id)
    first_context = parse_traceparent(turn.traceparent)
    assert first_context is not None
    structured_span = next(
        span for span in spans if span["name"] == "complete_structured"
    )
    _assert_parent_edge(structured_span, first_context)
    assert len({span["context"]["trace_id"] for span in spans}) == 2


async def test_run_agent_structured_returns_a_validated_answer(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"
    conversation_id = uuid4().hex
    requests: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if len(requests) == 1:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"key": "answer"}',
                        },
                    }
                ],
            }
        elif "response_format" in body:
            message = {
                "role": "assistant",
                "content": '{"country": "France", "city": "Paris"}',
            }
        else:
            message = {"role": "assistant", "content": "42"}
        return completion(model, message, prompt=10, completion_tokens=2)

    client = await stub(handler)
    message: ChatCompletionMessageParam = {
        "role": "user",
        "content": "What is the capital of France?",
    }

    parsed, response, usage, transcript, turn = await run_agent_structured(
        ToolContext(),
        client=client,
        model=model,
        messages=[message],
        response_model=Capital,
        tools=[Lookup()],
        conversation_id=conversation_id,
    )
    provider.force_flush()

    assert parsed == Capital(country="France", city="Paris")
    assert (
        response.choices[0].message.content == '{"country": "France", "city": "Paris"}'
    )
    assert transcript[-1].get("content") == '{"country": "France", "city": "Paris"}'
    assert len(transcript) == 6
    assert usage.prompt_tokens == 30
    assert [request.get("response_format", {}).get("type") for request in requests] == [
        None,
        None,
        "json_schema",
    ]

    spans = _conversation(read_spans(traces_dir), conversation_id)
    root = next(span for span in spans if span["name"] == "invoke_agent_structured")
    chats = [span for span in spans if span["name"] == "chat"]
    structured = [span for span in spans if span["name"] == "complete_structured"]
    tools = [span for span in spans if span["name"] == "execute_tool"]

    assert len(chats) == 3
    assert len(structured) == 1
    assert len(tools) == 1
    assert structured[0]["parent_id"] == root["context"]["span_id"]
    assert all(chat["parent_id"] == root["context"]["span_id"] for chat in chats[:2])
    assert chats[2]["parent_id"] == structured[0]["context"]["span_id"]
    assert len({span["context"]["trace_id"] for span in spans}) == 1
    assert parse_traceparent(turn.traceparent) is not None


async def test_structured_attempts_nest_under_one_span(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"
    attempts: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        attempts.append(body)
        content = (
            "not json"
            if len(attempts) == 1
            else '{"country": "France", "city": "Paris"}'
        )
        return completion(
            model,
            {"role": "assistant", "content": content},
            prompt=10,
            completion_tokens=2,
        )

    client = await stub(handler)
    message: ChatCompletionMessageParam = {
        "role": "user",
        "content": "What is the capital of France?",
    }

    result, _raw, _usage, _turn = await complete_structured(
        client=client,
        model=model,
        messages=[message],
        response_model=Capital,
    )
    provider.force_flush()

    assert result.city == "Paris"
    assert len(attempts) == 2

    spans = read_spans(traces_dir)
    structured = next(
        span
        for span in spans
        if span["name"] == "complete_structured"
        and span["attributes"].get("gen_ai.request.model") == model
    )
    chats = [
        span
        for span in spans
        if span["context"]["trace_id"] == structured["context"]["trace_id"]
        and span["name"] == "chat"
    ]

    assert len(chats) == 2
    assert all(chat["parent_id"] == structured["context"]["span_id"] for chat in chats)
    assert structured["attributes"]["gen_ai.response.format"] == "json_schema"
    assert structured["attributes"]["gen_ai.usage.input_tokens"] == 20
