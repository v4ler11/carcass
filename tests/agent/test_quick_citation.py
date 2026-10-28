import pytest
from pydantic import BaseModel, ValidationError

from agent.citations import (
    QuickCitation,
    citation_problems,
    cited_text,
    validate_quick_citations,
)
from agent.guardrails import (
    complete_structured_with_guardrails,
    guardrail_repair_message,
)
from tests.stub_server import StubFactory, completion


def citation(c_id: str) -> QuickCitation:
    return QuickCitation(
        claim=f"Claim for {c_id}",
        c_id=c_id,
        verification_instructions=(
            "VISUAL | zoom-1.png | bbox=[0,0,10,10] | compare the zone fill"
        ),
    )


def test_claim_is_first_and_fields_are_described() -> None:
    properties = QuickCitation.model_json_schema(by_alias=True)["properties"]

    assert list(properties) == ["claim", "c_id", "verification_instructions"]
    assert "factual claim" in properties["claim"]["description"]
    assert "TEXT |" in properties["verification_instructions"]["description"]


def test_accepts_both_inline_tag_forms() -> None:
    text = "Zone UMG (c-01)[beige fill] and ER (c-02)[blue line]."
    assert citation_problems(text, [citation("c-01"), citation("c-02")]) == []

    markdown = "Zone UMG [c-01](zoom-1.png: beige fill)."
    assert citation_problems(markdown, [citation("c-01")]) == []


def test_reports_duplicate_missing_and_unused_ids() -> None:
    text = "Zone (c-01)[a] and again (c-01)[b], then (c-02)[c]."
    problems = citation_problems(
        text, [citation("c-01"), citation("c-01"), citation("c-03")]
    )

    assert problems == [
        "citation ids used more than once in the text: c-01",
        "citation ids present more than once in citations[]: c-01",
        "inline tags without a citations[] entry: c-02",
        "citations[] entries without an inline tag: c-03",
    ]


def test_citation_requires_an_allowed_verification_prefix() -> None:
    with pytest.raises(ValidationError, match="must start with one of"):
        QuickCitation(
            claim="A claim",
            c_id="c-01",
            verification_instructions="look at the image",
        )


def test_repair_message_names_every_problem() -> None:
    message = guardrail_repair_message(
        ["inline tags without a citations[] entry: c-02"]
    )

    assert message["role"] == "user"
    assert isinstance(message["content"], str)
    assert "Structured-output guardrails failed" in message["content"]
    assert "c-02" in message["content"]


def test_cited_text_ignores_the_citations_array() -> None:
    class Answer(BaseModel):
        claim: str
        citations: list[QuickCitation]

    answer = Answer(claim="Zone (c-01)[a]", citations=[citation("c-01")])
    assert cited_text(answer) == "Zone (c-01)[a]"


class TinyAnswer(BaseModel):
    answer: str
    citations: list[QuickCitation]


async def test_repairs_a_citation_failure_then_returns(stub: StubFactory) -> None:
    requests: list[dict[str, object]] = []
    invalid = TinyAnswer(
        answer="Zone (c-01)[a] and again (c-01)[b]",
        citations=[citation("c-01")],
    )
    valid = TinyAnswer(answer="Zone (c-01)[a]", citations=[citation("c-01")])
    answers = [invalid, valid]

    def handler(body: dict[str, object]) -> dict[str, object]:
        requests.append(body)
        answer = answers.pop(0)
        return completion(
            "m",
            {"role": "assistant", "content": answer.model_dump_json(by_alias=True)},
            prompt=5,
            completion_tokens=1,
        )

    client = await stub(handler)
    parsed, _response, _usage, _turn = await complete_structured_with_guardrails(
        client=client,
        model="m",
        messages=[{"role": "user", "content": "resolve"}],
        response_model=TinyAnswer,
        guardrails=[validate_quick_citations],
        max_attempts=2,
    )

    assert parsed == valid
    assert len(requests) == 2
    second_messages = requests[1]["messages"]
    assert isinstance(second_messages, list)
    assert second_messages[-1]["role"] == "user"
    assert isinstance(second_messages[-1]["content"], str)
    assert "Structured-output guardrails failed" in second_messages[-1]["content"]
