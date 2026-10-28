import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

from agent import pipeline
from agent.citations import QuickCitation
from agent.s2_fan_out_questions import (
    AtomicQuestion,
    ExpectedAnswerType,
    QuestionPlan,
)
from agent.s3_qna import (
    AnswerConfidence,
    AnswerStatus,
    QuestionAnswer,
    VerifiedQuestionAnswer,
    s3_qna_node,
)
from chat.clients import Client
from tests.agent.test_s2_fan_out_questions import verified_context
from tests.stub_server import StubFactory, completion


def plan() -> QuestionPlan:
    return QuestionPlan(
        questions=[
            AtomicQuestion(
                id="Q01",
                category="destination_admissibility",
                question_text="Is habitation permitted in UMG?",
                expected_answer_type=ExpectedAnswerType.boolean,
                likely_source_documents=["plu_reglement: UMG Article 1"],
                rationale="The planner could not confirm admissibility.",
            ),
            AtomicQuestion(
                id="Q02",
                category="height_limit",
                question_text="What is the maximum height in UMG?",
                expected_answer_type=ExpectedAnswerType.numeric,
                likely_source_documents=["plu_reglement: UMG Article 10"],
                depends_on=["Q01"],
                rationale="Height depends on the destination classification.",
            ),
        ],
        carry_forward_limitations=[],
    )


def answer_json(question_id: str) -> str:
    return QuestionAnswer(
        question_id=question_id,
        status=AnswerStatus.confirmed,
        answer_text=(
            f"Answer for {question_id} "
            "(c-01)[TEXT | plu_reglement | page=10 | quote='maximum height']."
        ),
        answer_value=None,
        confidence=AnswerConfidence.high,
        contradicts_planner_rationale=False,
        citations=[
            QuickCitation(
                claim=f"Claim for {question_id}.",
                c_id="c-01",
                verification_instructions=(
                    "TEXT | plu_reglement | page=10 | quote='maximum height'"
                ),
            )
        ],
    ).model_dump_json(by_alias=True)


def question_id_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            match = re.search(r'"question_id":\s*"(Q\d+)"', content)
            if match and match.group(1) in {"Q01", "Q02"}:
                return match.group(1)
    raise AssertionError("question id not found in request")


async def test_s3_node_answers_dependencies_in_order(stub: StubFactory) -> None:
    requests: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if "response_format" in body:
            schema_name = body["response_format"]["json_schema"]["name"]
            if schema_name == "QuestionAnswer":
                question_id = question_id_from_messages(body["messages"])
                inputs[question_id] = next(
                    message["content"]
                    for message in body["messages"]
                    if isinstance(message.get("content"), str)
                    and message["content"].startswith(
                        "Answer this single atomic question."
                    )
                )
                return completion(
                    "m",
                    {"role": "assistant", "content": answer_json(question_id)},
                    prompt=100,
                    completion_tokens=50,
                )
            if schema_name == "CitationVerifications":
                c_ids = sorted(
                    set(re.findall(r"c-\d{2,}", body["messages"][-1]["content"]))
                )
                return completion(
                    "m",
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "verifications": [
                                    {
                                        "c_id": c_id,
                                        "confirmed": True,
                                        "reasoning": "Checked the cited page.",
                                    }
                                    for c_id in c_ids
                                ]
                            }
                        ),
                    },
                    prompt=100,
                    completion_tokens=50,
                )
            raise AssertionError(f"unexpected schema: {schema_name}")
        return completion(
            "m",
            {"role": "assistant", "content": "draft"},
            prompt=100,
            completion_tokens=50,
        )

    client = await stub(handler)
    answers = await s3_qna_node(
        client=client,
        model="m",
        parcel_context=verified_context(),
        plan=plan(),
        max_concurrency=2,
    )

    assert set(answers) == {"Q01", "Q02"}
    assert all(answer.status == AnswerStatus.confirmed for answer in answers.values())
    assert all(
        citation.confirmed
        for answer in answers.values()
        for citation in answer.citations
    )
    assert '"question_id": "Q01"' in inputs["Q02"]
    assert "Answer for Q01" in inputs["Q02"]
    assert any(
        "You are a worker agent in a French PLU" in message.get("content", "")
        for request in requests
        for message in request["messages"]
        if message.get("role") == "system"
    )


async def test_run_s3_saves_one_file_per_question(
    tmp_path: Path, monkeypatch: Any
) -> None:
    s1_path = tmp_path / "out" / "s1" / "results.json"
    s2_path = tmp_path / "out" / "s2" / "results.json"
    results_dir = tmp_path / "out" / "s3" / "results"
    s1_path.parent.mkdir(parents=True)
    s2_path.parent.mkdir(parents=True)
    s1_path.write_text(verified_context().model_dump_json(by_alias=True))
    s2_path.write_text(plan().model_dump_json(by_alias=True))

    answer = VerifiedQuestionAnswer(
        question_id="Q01",
        status=AnswerStatus.not_found,
        answer_text="No support was found in the scoped documents.",
        answer_value=None,
        confidence=AnswerConfidence.low,
        contradicts_planner_rationale=True,
        discrepancy_note="The planner rationale was not supported.",
        citations=[],
    )

    async def fake_node(**kwargs: Any) -> dict[str, VerifiedQuestionAnswer]:
        return {"Q01": answer}

    monkeypatch.setattr(pipeline, "S1_RESULTS_PATH", s1_path)
    monkeypatch.setattr(pipeline, "S2_RESULTS_PATH", s2_path)
    monkeypatch.setattr(pipeline, "S3_RESULTS_DIR", results_dir)
    monkeypatch.setattr(pipeline, "s3_qna_node", fake_node)

    results = await pipeline.run_pipeline(
        client=cast(Client, object()), model="m", stage="s3"
    )

    assert results == {"s3": {"Q01": answer}}
    path = results_dir / "Q01.json"
    assert json.loads(path.read_text()) == answer.model_dump(by_alias=True)


async def test_s3_rejects_dependency_cycles() -> None:
    cyclic = plan().model_copy(
        update={
            "questions": [
                plan().questions[0].model_copy(update={"depends_on": ["Q02"]}),
                plan().questions[1].model_copy(update={"depends_on": ["Q01"]}),
            ]
        }
    )

    with pytest.raises(ValueError, match="cyclic depends_on"):
        await s3_qna_node(
            client=cast(Client, object()),
            model="m",
            parcel_context=verified_context(),
            plan=cyclic,
        )
