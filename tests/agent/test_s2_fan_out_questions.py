import json
from pathlib import Path
from typing import Any, cast

from agent import pipeline
from agent.citations import VerifiedQuickCitation
from agent.prompts.s2_fan_out_questions import SEED_CHECKLIST, render_system_prompt
from agent.s1_overlay_resolution import VerifiedParcelContext
from agent.s2_fan_out_questions import (
    AtomicQuestion,
    ExpectedAnswerType,
    QuestionPlan,
    s2_fan_out_questions_node,
)
from chat.clients import Client
from tests.agent.test_s1_overlay_resolution import parcel_context
from tests.stub_server import StubFactory, completion


def plan() -> QuestionPlan:
    return QuestionPlan(
        questions=[
            AtomicQuestion(
                id=f"Q{index:02}",
                category=item["category"],
                question_text=f"What does the PLU require for {item['category']}?",
                expected_answer_type=ExpectedAnswerType.text,
                likely_source_documents=["plu_reglement_section1_UMG"],
                rationale=f"{item['category']} is load-bearing for buildability.",
            )
            for index, item in enumerate(SEED_CHECKLIST, start=1)
        ],
        carry_forward_limitations=[],
    )


def verified_context() -> VerifiedParcelContext:
    base = parcel_context()
    return VerifiedParcelContext(
        **base.model_dump(exclude={"citations"}),
        citations=[
            VerifiedQuickCitation(
                **citation.model_dump(by_alias=True),
                confirmed=True,
                reasoning="Verified.",
            )
            for citation in base.citations
        ],
    )


def test_render_system_prompt_includes_context_and_fixed_inputs() -> None:
    context = verified_context()
    prompt = render_system_prompt(
        confirmed_parcel_context=context.model_dump_json(by_alias=True)
    )

    assert "91272000CR0247" in prompt
    assert "Résidence Sociale Étudiante" in prompt
    assert "destination_admissibility" in prompt
    assert "Produce the full question list now" in prompt
    assert "carry_forward_limitations" in prompt


async def test_s2_node_uses_all_tools_and_renders_context(
    stub: StubFactory,
) -> None:
    requests: list[dict[str, Any]] = []
    expected = plan()

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if "response_format" in body:
            return completion(
                "m",
                {
                    "role": "assistant",
                    "content": expected.model_dump_json(by_alias=True),
                },
                prompt=100,
                completion_tokens=50,
            )
        return completion(
            "m",
            {"role": "assistant", "content": "draft"},
            prompt=100,
            completion_tokens=50,
        )

    client = await stub(handler)
    result, _response, _usage, _turn = await s2_fan_out_questions_node(
        client=client,
        model="m",
        parcel_context=verified_context(),
    )

    assert result == expected
    assert requests[0]["messages"][0]["content"].startswith(
        "You are the question-planning agent"
    )
    assert "91272000CR0247" in requests[0]["messages"][0]["content"]
    assert "Résidence Jeunes Actifs" in requests[0]["messages"][0]["content"]
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == [
        "compute_buildable_envelope",
        "read_parcel_info",
        "read_plan_de_zonage_central_meta",
        "documents_list",
        "pages_read",
        "pages_search",
    ]
    assert requests[1]["response_format"]["json_schema"]["name"] == "QuestionPlan"


async def test_run_s2_loads_s1_from_disk_and_saves_result(
    tmp_path: Path, monkeypatch: Any
) -> None:
    s1_path = tmp_path / "out" / "s1" / "results.json"
    s2_path = tmp_path / "out" / "s2" / "results.json"
    s1_path.parent.mkdir(parents=True)
    s1_path.write_text(verified_context().model_dump_json(indent=2, by_alias=True))
    monkeypatch.setattr(pipeline, "S1_RESULTS_PATH", s1_path)
    monkeypatch.setattr(pipeline, "S2_RESULTS_PATH", s2_path)

    expected = plan()
    seen: list[str] = []

    async def fake_node(**kwargs: Any) -> tuple[Any, ...]:
        seen.append(kwargs["parcel_context"].idu)
        return expected, None, None, None

    monkeypatch.setattr(pipeline, "s2_fan_out_questions_node", fake_node)

    results = await pipeline.run_pipeline(
        client=cast(Client, object()), model="m", stage="s2"
    )

    assert seen == ["91272000CR0247"]
    assert results == {"s2": expected}
    assert json.loads(s2_path.read_text()) == expected.model_dump(by_alias=True)
