from typing import Any

import pytest
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from chat.structured import complete_structured, parse_model_output_json
from tests.stub_server import StubFactory, completion


class Inner(BaseModel):
    n: int


class Answer(BaseModel):
    name: str
    inner: Inner


EXPECTED = Answer(name="a", inner=Inner(n=1))
VALID = '{"name": "a", "inner": {"n": 1}}'


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        pytest.param(VALID, EXPECTED, id="bare-object"),
        pytest.param(f"```json\n{VALID}\n```", EXPECTED, id="fenced-code-block"),
        pytest.param(f"Here you go:\n{VALID}\nHope this helps.", EXPECTED, id="prose-around-object"),
        pytest.param(f"Use {{this}} format:\n{VALID}", EXPECTED, id="brace-before-object"),
        pytest.param(
            '{"name": "a {b}", "inner": {"n": 1}}',
            Answer(name="a {b}", inner=Inner(n=1)),
            id="braces-inside-string",
        ),
        pytest.param(
            '{"name": "a\nb", "inner": {"n": 1}}',
            Answer(name="a\nb", inner=Inner(n=1)),
            id="raw-newline-inside-string",
        ),
        pytest.param(f"{VALID} {VALID}", EXPECTED, id="first-of-two-objects"),
    ],
)
def test_parses_object_embedded_in_output(output: str, expected: Answer) -> None:
    assert parse_model_output_json(output, Answer) == expected


def test_skips_candidate_that_fails_validation() -> None:
    output = '{"name": 1, "inner": {"n": 1}} ' + VALID
    assert parse_model_output_json(output, Answer) == EXPECTED


def test_truncated_object_is_rejected() -> None:
    with pytest.raises(ValueError, match="No complete JSON object found"):
        parse_model_output_json('{"name": "a", "inner": {', Answer)


def test_plain_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="No complete JSON object found"):
        parse_model_output_json("I cannot help with that.", Answer)


def test_schema_mismatch_reports_the_model_name() -> None:
    with pytest.raises(ValueError, match="^JSON does not match Answer"):
        parse_model_output_json('{"name": "a"}', Answer)


class Capital(BaseModel):
    country: str
    city: str


JSON_ANSWER = '{"country": "France", "city": "Paris"}'


async def test_complete_structured_sends_the_strict_schema(stub: StubFactory) -> None:
    requests: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        return completion("m", {"role": "assistant", "content": JSON_ANSWER}, prompt=10, completion_tokens=2)

    client = await stub(handler)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "What is the capital of France?"},
    ]

    parsed, response, usage, _turn = await complete_structured(
        client=client, model="m", messages=messages, response_model=Capital
    )

    assert len(requests) == 1
    assert requests[0]["messages"] == messages
    assert requests[0]["response_format"]["json_schema"]["name"] == "Capital"
    assert parsed == Capital(country="France", city="Paris")
    assert response.choices[0].message.content == JSON_ANSWER
    assert usage.prompt_tokens == 10
    assert len(messages) == 1


async def test_complete_structured_raises_after_the_last_attempt(stub: StubFactory) -> None:
    attempts = {"count": 0}

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        attempts["count"] += 1
        return completion("m", {"role": "assistant", "content": "not json"}, prompt=3, completion_tokens=1)

    client = await stub(handler)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "What is the capital of France?"},
    ]

    with pytest.raises(ValueError):
        await complete_structured(client=client, model="m", messages=messages, response_model=Capital)

    assert attempts["count"] == 2
    assert len(messages) == 1
