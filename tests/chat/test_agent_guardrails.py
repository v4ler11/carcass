from typing import Any

import pytest
from pydantic import BaseModel

from chat.agent import run_agent, run_agent_structured
from chat.tools.context import ToolContext
from tests.stub_server import StubFactory, completion


class Capital(BaseModel):
    country: str
    city: str


def _text_guardrail(text: str) -> list[str]:
    return [] if text == "good" else ["answer is not 'good'"]


async def test_run_agent_repairs_a_text_guardrail_failure(stub: StubFactory) -> None:
    requests: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        content = "bad" if len(requests) == 1 else "good"
        return completion(
            "m",
            {"role": "assistant", "content": content},
            prompt=5,
            completion_tokens=1,
        )

    client = await stub(handler)

    response, usage, transcript, _turn = await run_agent(
        ToolContext(),
        client=client,
        model="m",
        messages=[{"role": "user", "content": "answer"}],
        guardrails=[_text_guardrail],
        max_attempts=2,
    )

    assert response.choices[0].message.content == "good"
    assert usage.prompt_tokens == 10
    assert transcript[-1].get("content") == "good"
    assert len(requests) == 2
    repair = requests[1]["messages"][-1]
    assert repair["role"] == "user"
    assert "Structured-output guardrails failed" in repair["content"]
    assert "answer is not 'good'" in repair["content"]


async def test_run_agent_raises_when_text_guardrails_never_pass(
    stub: StubFactory,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        return completion(
            "m", {"role": "assistant", "content": "bad"}, prompt=1, completion_tokens=1
        )

    client = await stub(handler)

    with pytest.raises(ValueError, match="agent guardrail verification failed"):
        await run_agent(
            ToolContext(),
            client=client,
            model="m",
            messages=[{"role": "user", "content": "answer"}],
            guardrails=[_text_guardrail],
            max_attempts=2,
        )

    assert len(requests) == 2


def _capital_guardrail(answer: Capital) -> list[str]:
    return [] if answer.city == "Paris" else [f"wrong city: {answer.city}"]


async def test_run_agent_structured_repairs_a_guardrail_failure(
    stub: StubFactory,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if "response_format" not in body:
            content = "draft"
        elif (
            len([request for request in requests if "response_format" in request]) == 1
        ):
            content = '{"country": "France", "city": "Lyon"}'
        else:
            content = '{"country": "France", "city": "Paris"}'
        return completion(
            "m",
            {"role": "assistant", "content": content},
            prompt=10,
            completion_tokens=2,
        )

    client = await stub(handler)

    parsed, response, usage, transcript, _turn = await run_agent_structured(
        ToolContext(),
        client=client,
        model="m",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_model=Capital,
        guardrails=[_capital_guardrail],
        max_attempts=2,
    )

    assert parsed == Capital(country="France", city="Paris")
    assert (
        response.choices[0].message.content == '{"country": "France", "city": "Paris"}'
    )
    assert usage.prompt_tokens == 30
    assert transcript[-1].get("content") == '{"country": "France", "city": "Paris"}'
    assert [request.get("response_format", {}).get("type") for request in requests] == [
        None,
        "json_schema",
        "json_schema",
    ]
    repair = requests[2]["messages"][-1]
    assert repair["role"] == "user"
    assert "wrong city: Lyon" in repair["content"]


async def test_run_agent_structured_prompts_schema_then_retracts_it(
    stub: StubFactory,
) -> None:
    requests: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if "response_format" in body:
            content = '{"country": "France", "city": "Paris"}'
        else:
            content = "draft"
        return completion(
            "m",
            {"role": "assistant", "content": content},
            prompt=10,
            completion_tokens=2,
        )

    client = await stub(handler)

    _parsed, _response, _usage, transcript, _turn = await run_agent_structured(
        ToolContext(),
        client=client,
        model="m",
        messages=[
            {"role": "system", "content": "You are a geographer."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        response_model=Capital,
    )

    first_messages = requests[0]["messages"]
    assert [message["role"] for message in first_messages] == [
        "system",
        "user",
        "user",
    ]
    schema_message = first_messages[1]["content"]
    assert schema_message.startswith("response_schema: ")
    assert '"city"' in schema_message

    final_messages = requests[1]["messages"]
    assert final_messages[-1]["role"] == "user"
    assert "final answer" in final_messages[-1]["content"].lower()
    assert not any(
        message.get("content", "").startswith("response_schema: ")
        for message in final_messages
    )
    for message in transcript:
        content = message.get("content")
        assert not (
            isinstance(content, str) and content.startswith("response_schema: ")
        )
