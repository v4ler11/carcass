import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.decision import (
    AnswerAnswerResourceChoice,
    AnswerAnswerResourcePredicate,
    AnswerAnswerResourceRefusal,
    AnswerAnswerResourceScore,
    Decision,
)
from openai.types.decision_create_params import Question
from opentelemetry.sdk.trace import TracerProvider

from chat.agent import run_agent
from chat.clients import Client
from chat.decisions import answers_by_name, decide
from chat.tools import Tool, ToolContext
from chat.usage import ChatUsage
from tests.stub_server import DECISIONS_PATH, StubFactory, completion
from tracing import Turn, read_spans

QUESTIONS: list[Question] = [
    {"type": "predicate", "name": "spam", "instructions": "Is this spam?"},
    {
        "type": "choice",
        "name": "team",
        "instructions": "Which team should handle this?",
        "choices": [
            {"value": "billing", "description": "Payments and refunds."},
            {"value": "other", "description": "Everything else."},
        ],
    },
    {
        "type": "score",
        "name": "severity",
        "instructions": "How severe is this?",
        "levels": [
            {"label": "Low", "description": "Cosmetic."},
            {"label": "High", "description": "Fully blocked."},
        ],
    },
]

ANSWERS: list[dict[str, Any]] = [
    {"type": "predicate", "name": "spam", "probability": 0.91},
    {
        "type": "choice",
        "name": "team",
        "choice": "billing",
        "confidence": 0.8,
        "probabilities": [
            {"value": "billing", "probability": 0.9},
            {"value": "other", "probability": 0.1},
        ],
    },
    {
        "type": "score",
        "name": "severity",
        "score": 1.1,
        "confidence": 0.55,
        "probabilities": [
            {"value": 0, "label": "Low", "probability": 0.1},
            {"value": 1, "label": "High", "probability": 0.9},
        ],
    },
    {"type": "refusal", "name": "opinion"},
]

USAGE: dict[str, Any] = {
    "input_tokens": 100,
    "output_tokens": 5,
    "total_tokens": 105,
    "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 2},
    "output_tokens_details": {"reasoning_tokens": 3},
}


async def test_decide_returns_typed_answers_and_usage(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"
    requests: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        return {"model": model, "answers": ANSWERS, "usage": USAGE}

    client = await stub(handler, path=DECISIONS_PATH)

    decision, usage, turn = await decide(
        client=client,
        model=model,
        input="I was charged twice for my order.",
        questions=QUESTIONS,
        prompt_cache_key="tickets",
    )
    provider.force_flush()

    assert len(requests) == 1
    assert requests[0]["input"] == "I was charged twice for my order."
    assert requests[0]["questions"] == QUESTIONS
    assert requests[0]["prompt_cache_key"] == "tickets"

    by_name = answers_by_name(decision)
    assert list(by_name) == ["spam", "team", "severity", "opinion"]

    spam = by_name["spam"]
    assert isinstance(spam, AnswerAnswerResourcePredicate)
    assert spam.probability == 0.91

    team = by_name["team"]
    assert isinstance(team, AnswerAnswerResourceChoice)
    assert team.choice == "billing"
    assert team.confidence == 0.8
    assert [probability.value for probability in team.probabilities] == ["billing", "other"]

    severity = by_name["severity"]
    assert isinstance(severity, AnswerAnswerResourceScore)
    assert severity.score == 1.1
    assert [probability.value for probability in severity.probabilities] == [0, 1]

    assert isinstance(by_name["opinion"], AnswerAnswerResourceRefusal)

    assert usage == ChatUsage(
        model=model,
        prompt_tokens=100,
        completion_tokens=5,
        total_tokens=105,
        cached_tokens=40,
        cache_write_tokens=2,
        reasoning_tokens=3,
    )

    span = next(
        span
        for span in read_spans(traces_dir)
        if span["name"] == "decisions"
        and span["attributes"].get("gen_ai.request.model") == model
    )
    assert span["attributes"]["gen_ai.conversation.id"] == turn.conversation_id
    assert span["kind"] == "SpanKind.CLIENT"
    assert span["attributes"]["gen_ai.operation.name"] == "decisions"
    assert span["attributes"]["gen_ai.request.model"] == model
    assert span["attributes"]["gen_ai.usage.input_tokens"] == 100
    assert span["attributes"]["gen_ai.usage.output_tokens"] == 5
    assert span["attributes"]["gen_ai.usage.cache_read.input_tokens"] == 40
    recorded = json.loads(span["attributes"]["decision.answers"])
    assert [answer["name"] for answer in recorded] == ["spam", "team", "severity", "opinion"]


def _decision(answers: list[dict[str, Any]]) -> Decision:
    return Decision.model_validate({"model": "m", "answers": answers, "usage": USAGE})


def test_answers_by_name_skips_unnamed_answers() -> None:
    decision = _decision(
        [
            {"type": "predicate", "probability": 0.5},
            {"type": "predicate", "name": "spam", "probability": 0.9},
        ]
    )

    assert list(answers_by_name(decision)) == ["spam"]


def test_answers_by_name_rejects_duplicates() -> None:
    decision = _decision(
        [
            {"type": "predicate", "name": "spam", "probability": 0.5},
            {"type": "predicate", "name": "spam", "probability": 0.5},
        ]
    )

    with pytest.raises(ValueError, match="duplicate answer name: 'spam'"):
        answers_by_name(decision)


class Classify(Tool):
    """Calls `decide` where an agent's tool would: inside the agent's turn."""

    def __init__(self, client: Client, model: str) -> None:
        self.client = client
        self.model = model
        self.turn: Turn | None = None

    @property
    def name(self) -> str:
        return "classify"

    async def execute(self, ctx, tool_call, args) -> list[ChatCompletionMessageParam]:
        _decision, _usage, self.turn = await decide(
            client=self.client,
            model=self.model,
            input="I was charged twice for my order.",
            questions=QUESTIONS,
        )
        return [{"role": "tool", "tool_call_id": tool_call.id, "content": '{"classified": true}'}]

    def into_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Classify the ticket.",
                "parameters": {"type": "object", "properties": {}},
            },
        }


async def test_decision_called_as_a_tool_joins_the_agent_conversation(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    model = f"model-{uuid4().hex}"
    conversation_id = uuid4().hex
    steps = {"count": 0}

    def chat_handler(body: dict[str, Any]) -> dict[str, Any]:
        steps["count"] += 1
        if steps["count"] == 1:
            return completion(
                model,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "classify", "arguments": "{}"},
                        }
                    ],
                },
                prompt=10,
                completion_tokens=2,
            )
        return completion(model, {"role": "assistant", "content": "triaged"}, prompt=20, completion_tokens=3)

    def decisions_handler(body: dict[str, Any]) -> dict[str, Any]:
        return {"model": model, "answers": ANSWERS, "usage": USAGE}

    client = await stub(chat_handler)
    decisions_client = await stub(decisions_handler, path=DECISIONS_PATH)
    tool = Classify(decisions_client, model)

    _response, _usage, _transcript, _turn = await run_agent(
        ToolContext(),
        client=client,
        model=model,
        messages=[{"role": "user", "content": "triage this"}],
        tools=[tool],
        conversation_id=conversation_id,
    )
    provider.force_flush()

    assert tool.turn is not None
    assert tool.turn.conversation_id == conversation_id

    spans = read_spans(traces_dir)
    root = next(
        span
        for span in spans
        if span["name"] == "invoke_agent"
        and span["attributes"].get("gen_ai.conversation.id") == conversation_id
    )
    decisions_span = next(
        span
        for span in spans
        if span["name"] == "decisions"
        and span["attributes"].get("gen_ai.conversation.id") == conversation_id
    )
    tool_span = next(
        span
        for span in spans
        if span["name"] == "execute_tool"
        and span["attributes"].get("gen_ai.conversation.id") == conversation_id
    )

    assert decisions_span["attributes"]["gen_ai.conversation.id"] == conversation_id
    assert decisions_span["context"]["trace_id"] == root["context"]["trace_id"]
    assert decisions_span["parent_id"] == tool_span["context"]["span_id"]
    assert "gen_ai.conversation.parent_trace_id" not in decisions_span["attributes"]
