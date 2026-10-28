import json
from pathlib import Path
from typing import Any, cast

import pytest

from agent import pipeline
from agent.citations import VerifiedQuickCitation
from agent.prompts.s2_fan_out_questions import TARGET_PROGRAM
from agent.s2_fan_out_questions import (
    AtomicQuestion,
    ExpectedAnswerType,
    QuestionPlan,
)
from agent.s3_qna import (
    AnswerConfidence,
    AnswerStatus,
    VerifiedQuestionAnswer,
)
from agent.s4_synthesis import (
    AnswerStore,
    MergedQuestionAnswer,
    SynthesisReport,
    render_markdown,
    s4_synthesis_node,
)
from agent.s4_synthesis.node import structure_problems
from chat.clients import Client
from tests.agent.test_s2_fan_out_questions import verified_context
from tests.stub_server import StubFactory, completion


def _citation(c_id: str, claim: str, instructions: str) -> VerifiedQuickCitation:
    return VerifiedQuickCitation(
        claim=claim,
        c_id=c_id,
        verification_instructions=instructions,
        confirmed=True,
        reasoning="Verified against the cited source.",
    )


def _question(question_id: str, category: str) -> AtomicQuestion:
    return AtomicQuestion(
        id=question_id,
        category=category,
        question_text=f"Question {question_id}",
        expected_answer_type=ExpectedAnswerType.text,
        likely_source_documents=["plu_reglement: UMG"],
        rationale="Material to the final study.",
    )


def _answer(
    question_id: str,
    *,
    status: AnswerStatus,
    contradiction: bool = False,
    category: str = "general",
) -> VerifiedQuestionAnswer:
    return VerifiedQuestionAnswer(
        question_id=question_id,
        status=status,
        answer_text=f"Answer for {question_id} (c-01)[quoted support].",
        answer_value=None,
        confidence=AnswerConfidence.high,
        contradicts_planner_rationale=contradiction,
        discrepancy_note="The planner was not supported." if contradiction else None,
        citations=[
            _citation(
                "c-01",
                f"Claim for {question_id}.",
                "TEXT | plu_reglement | page=319 | quote='quoted support'",
            )
        ],
    )


def plan() -> QuestionPlan:
    return QuestionPlan(
        questions=[
            _question("Q01", "destination_admissibility"),
            _question("Q02", "height_limit"),
            _question("Q03", "oap_binding_vs_orientation"),
        ],
        carry_forward_limitations=[],
    )


def answer_store() -> AnswerStore:
    return AnswerStore(
        answers=[
            MergedQuestionAnswer(
                question=plan().questions[0],
                answer=_answer(
                    "Q01", status=AnswerStatus.confirmed, contradiction=True
                ),
            ),
            MergedQuestionAnswer(
                question=plan().questions[1],
                answer=_answer("Q02", status=AnswerStatus.partial),
            ),
            MergedQuestionAnswer(
                question=plan().questions[2],
                answer=_answer("Q03", status=AnswerStatus.confirmed),
            ),
        ]
    )


def synthesis_text() -> str:
    return """# Study
## buildability_conclusion
Both buildings are admitted subject to an open classification (see c-03).
## parcel_identification
The parcel is 91272000CR0247 (c-01)[METADATA parcel.idu = 91272000CR0247]. [source: confirmed_parcel_context.idu] [source_type: geometric]
## zoning_context
The parcel is in UMG (c-02)[TEXT plu_reglement page 319]. [source: confirmed_parcel_context.zone.confirmed_zone] [source_type: regulatory]
## admissibility
Résidence Sociale Étudiante — admissibility_status: permitted (c-03)[TEXT plu_reglement page 319].
Résidence Jeunes Actifs — admissibility_status: conditional (c-04)[TEXT oap page 34].
The OAP alignment guidance is [ORIENTATION].
## envelope_rules
The calculated envelope is reported by compute_buildable_envelope. calculation_trace: uniform inward buffer.
## non_envelope_feasibility
Parking is a non-envelope condition (c-05)[TEXT plu_reglement page 326].
## assumptions_and_unresolved
Q02 is partial and is therefore not used as a clean conclusion.
## self_correction_log
Q01 contradicted the planner rationale; the independent Stage 3 answer won because it cites the source text.
## markdown-ready citations
c-01 through c-05 are listed in the citations array.
"""


def synthesis_report() -> SynthesisReport:
    return SynthesisReport(
        text=synthesis_text(),
        citations=[
            _citation(
                "c-01",
                "Parcel id is 91272000CR0247.",
                "METADATA | parcel.idu | 91272000CR0247",
            ),
            _citation(
                "c-02",
                "Zone UMG is confirmed.",
                "TEXT | plu_reglement | page=319 | quote='La zone UMG'",
            ),
            _citation(
                "c-03",
                "Habitation is permitted.",
                "TEXT | plu_reglement | page=319 | quote='habitation'",
            ),
            _citation(
                "c-04",
                "OAP alignment is conditional.",
                "TEXT | oap_91272_20210706 | page=34 | quote='implantation'",
            ),
            _citation(
                "c-05",
                "Parking is a non-envelope condition.",
                "TEXT | plu_reglement | page=326 | quote='stationnement'",
            ),
        ],
    )


async def test_s4_node_produces_report_without_citation_verification(
    stub: StubFactory,
) -> None:
    requests: list[dict[str, Any]] = []
    schemas: list[str] = []
    report_json = synthesis_report().model_dump_json(by_alias=True)

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if "response_format" in body:
            schema_name = body["response_format"]["json_schema"]["name"]
            schemas.append(schema_name)
            if schema_name == "SynthesisReport":
                return completion(
                    "m",
                    {"role": "assistant", "content": report_json},
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
    result, _response, _usage, _turn = await s4_synthesis_node(
        client=client,
        model="m",
        parcel_context=verified_context(),
        answer_store=answer_store(),
    )

    assert result.text == synthesis_text()
    assert schemas == ["SynthesisReport"]
    assert [citation.id for citation in result.citations] == [
        "c-01",
        "c-02",
        "c-03",
        "c-04",
        "c-05",
    ]
    assert "confirmed" not in result.citations[0].model_dump()
    system_prompt = requests[0]["messages"][0]["content"]
    assert "ANSWER_STORE" in system_prompt
    assert "91272000CR0247" in system_prompt
    assert "Résidence Sociale Étudiante" in system_prompt
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == [
        "compute_buildable_envelope",
        "read_parcel_info",
        "read_plan_de_zonage_central_meta",
        "documents_list",
        "pages_read",
        "pages_search",
    ]


async def test_run_s4_merges_answers_and_writes_markdown(
    tmp_path: Path, monkeypatch: Any
) -> None:
    s1_path = tmp_path / "out" / "s1" / "results.json"
    s2_path = tmp_path / "out" / "s2" / "results.json"
    s3_dir = tmp_path / "out" / "s3" / "results"
    store_path = tmp_path / "out" / "s4" / "answer_store.json"
    results_path = tmp_path / "out" / "s4" / "results.json"
    markdown_path = tmp_path / "out" / "s4" / "result.md"
    s1_path.parent.mkdir(parents=True)
    s2_path.parent.mkdir(parents=True)
    s3_dir.mkdir(parents=True)
    s1_path.write_text(verified_context().model_dump_json(by_alias=True))
    s2_path.write_text(plan().model_dump_json(by_alias=True))
    for merged in answer_store().answers:
        (s3_dir / f"{merged.question.id}.json").write_text(
            merged.answer.model_dump_json(by_alias=True)
        )

    captured: dict[str, Any] = {}

    async def fake_node(**kwargs: Any) -> tuple[Any, ...]:
        captured.update(kwargs)
        return synthesis_report(), None, None, None

    async def fake_markdown(**kwargs: Any) -> tuple[str, None, None]:
        captured["markdown_report"] = kwargs["report"]
        return "# Result\n", None, None

    monkeypatch.setattr(pipeline, "S1_RESULTS_PATH", s1_path)
    monkeypatch.setattr(pipeline, "S2_RESULTS_PATH", s2_path)
    monkeypatch.setattr(pipeline, "S3_RESULTS_DIR", s3_dir)
    monkeypatch.setattr(pipeline, "S4_ANSWER_STORE_PATH", store_path)
    monkeypatch.setattr(pipeline, "S4_RESULTS_PATH", results_path)
    monkeypatch.setattr(pipeline, "S4_MARKDOWN_PATH", markdown_path)
    monkeypatch.setattr(pipeline, "s4_synthesis_node", fake_node)
    monkeypatch.setattr(pipeline, "render_markdown", fake_markdown)

    result = await pipeline.run_pipeline(
        client=cast(Client, object()), model="m", stage="s4"
    )

    assert result == {"s4": synthesis_report()}
    merged_store = AnswerStore.model_validate_json(store_path.read_text())
    assert [entry.question.id for entry in merged_store.answers] == [
        "Q01",
        "Q02",
        "Q03",
    ]
    assert merged_store.carry_forward_limitations == []
    assert captured["answer_store"] == merged_store
    assert json.loads(results_path.read_text()) == synthesis_report().model_dump(
        mode="json", by_alias=True
    )
    assert markdown_path.read_text() == "# Result\n"


def markdown_for(ids: list[str]) -> str:
    body = " ".join(f"[{c_id}]" for c_id in ids)
    entries = "\n".join(f"- **{c_id}** — claim" for c_id in ids)
    return f"# Etude\n\n{body} support.\n\n## Citations\n\n{entries}"


ALL_IDS = ["c-01", "c-02", "c-03", "c-04", "c-05"]


async def test_render_markdown_passes_the_full_report(stub: StubFactory) -> None:
    requests: list[dict[str, Any]] = []

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        return completion(
            "m",
            {"role": "assistant", "content": markdown_for(ALL_IDS)},
            prompt=100,
            completion_tokens=50,
        )

    client = await stub(handler)
    text, _response, _usage = await render_markdown(
        client=client,
        model="m",
        report=synthesis_report(),
        plan=plan(),
    )

    assert text == markdown_for(ALL_IDS) + "\n"
    assert requests[0]["messages"][0]["content"].startswith("You rewrite a completed")
    user_prompt = requests[0]["messages"][1]["content"]
    assert "STUDY_JSON" in user_prompt
    assert "c-01" in user_prompt
    assert "Résidence Sociale Étudiante" in user_prompt


async def test_render_markdown_retries_once_when_citations_are_dropped(
    stub: StubFactory,
) -> None:
    requests: list[dict[str, Any]] = []
    replies = [markdown_for(ALL_IDS[:3]), markdown_for(ALL_IDS)]

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        return completion(
            "m",
            {"role": "assistant", "content": replies[len(requests) - 1]},
            prompt=100,
            completion_tokens=50,
        )

    client = await stub(handler)
    text, _response, usage = await render_markdown(
        client=client, model="m", report=synthesis_report(), plan=plan()
    )

    assert text == markdown_for(ALL_IDS) + "\n"
    assert len(requests) == 2
    assert "c-04, c-05" in requests[1]["messages"][-1]["content"]
    assert usage.prompt_tokens == 200


async def test_render_markdown_raises_when_citations_stay_wrong(
    stub: StubFactory,
) -> None:
    bad = markdown_for(ALL_IDS[:4] + ["c-99"])

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        return completion(
            "m",
            {"role": "assistant", "content": bad},
            prompt=100,
            completion_tokens=50,
        )

    client = await stub(handler)
    with pytest.raises(ValueError, match="c-05.*c-99|c-99.*c-05"):
        await render_markdown(
            client=client, model="m", report=synthesis_report(), plan=plan()
        )


def test_structure_problems_accepts_the_fixture() -> None:
    assert structure_problems(synthesis_text(), TARGET_PROGRAM["use_types"]) == []  # type: ignore[arg-type]


def test_structure_problems_flags_missing_conclusion_and_status() -> None:
    text = synthesis_text().replace("## buildability_conclusion", "## summary")
    text = text.replace(
        "Résidence Jeunes Actifs — admissibility_status: conditional",
        "Résidence Jeunes Actifs is conditional",
    )

    problems = structure_problems(text, TARGET_PROGRAM["use_types"])  # type: ignore[arg-type]

    assert any("buildability_conclusion" in problem for problem in problems)
    assert any("Résidence Jeunes Actifs" in problem for problem in problems)
    assert len(problems) == 2


def test_structure_problems_requires_the_documented_order() -> None:
    text = (
        synthesis_text()
        .replace("parcel_identification", "TMP")
        .replace("zoning_context", "parcel_identification")
        .replace("TMP", "zoning_context")
    )

    problems = structure_problems(text, TARGET_PROGRAM["use_types"])  # type: ignore[arg-type]

    assert any("out of order" in problem for problem in problems)
