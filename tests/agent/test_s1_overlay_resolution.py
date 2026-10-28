import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from opentelemetry.sdk.trace import TracerProvider

from agent.citations import CitationVerification, QuickCitation
from agent.s1_overlay_resolution import (
    Confidence,
    EmplacementReserve,
    ERCategory,
    GeometryType,
    GeoreferenceQualityFlag,
    OAPSector,
    ParcelContext,
    PrescriptionCheck,
    PresenceStatus,
    ProximityToBoundary,
    VerifiedParcelContext,
    ZoneDetermination,
    s1_overlay_resolution_node,
)
from tests.stub_server import StubFactory, completion
from tracing import read_spans


def citation(c_id: str, claim: str) -> QuickCitation:
    return QuickCitation(
        claim=claim,
        c_id=c_id,
        verification_instructions=("VISUAL | zoom-1.png | bbox=[0,0,10,10] | checked"),
    )


def parcel_context() -> ParcelContext:
    return ParcelContext(
        idu="91272000CR0247",
        zone=ZoneDetermination(
            metadata_hypothesis=(
                "Metadata zone code UMG (c-01)[metadata_parcel.json zone.code=UMG]"
            ),
            confirmed_zone="UMG",
            hypothesis_confirmed=True,
            fully_within_zone=True,
            confidence=Confidence.high,
        ),
        emplacements_reserves=[
            EmplacementReserve(
                category=ERCategory.elargissements_voirie,
                overlaps_parcel=False,
                confidence=Confidence.medium,
            ),
            EmplacementReserve(
                category=ERCategory.voirie_creation_amelioration,
                overlaps_parcel=True,
                geometry_type=GeometryType.line,
                reference_id="a",
                nature=(
                    "Cycle path (c-02)[05_er_creation_amelioration_voirie.md row a]"
                ),
                affected_area_m2_estimate=120.0,
                affected_boundary_description="Along the northern frontage.",
                confidence=Confidence.medium,
            ),
            EmplacementReserve(
                category=ERCategory.equipements_espaces_publics,
                overlaps_parcel=False,
                confidence=Confidence.low,
            ),
        ],
        oap_sector=OAPSector(
            sector_name="Quartier Joliot Curie",
            metadata_flag_count=2,
            confirmed_within_sector=False,
            proximity_to_boundary=ProximityToBoundary.near_edge,
            confidence=Confidence.low,
        ),
        other_prescriptions_checklist=[
            PrescriptionCheck(
                type="espace_boise_classe",
                presence=PresenceStatus.absent,
                confidence=Confidence.high,
                comments=(
                    "The general legend has no classed-woodland symbol over "
                    "the parcel (c-05)[01_legende.png: no espace boisé "
                    "classé symbol over parcel]."
                ),
            )
        ],
        georeference_quality_flag=GeoreferenceQualityFlag(
            residual_mean_m=1.87,
            affects_boundary_calls=False,
            note=(
                "Residual is below the 3 m threshold "
                "(c-03)[metadata_parcel.json "
                "georeference.parcel_edge_residual_mean_m]"
            ),
        ),
        unresolved_flags=[
            (
                "OAP boundary is near the parcel edge "
                "(c-04)[zoom-2.png: OAP hatch adjacent to parcel]"
            )
        ],
        citations=[
            citation("c-01", "Metadata zone is UMG."),
            citation("c-02", "Table row a reserves a cycle path."),
            citation("c-03", "Georeference residual is below the 3 m threshold."),
            citation("c-04", "The OAP boundary is near the parcel edge."),
            citation("c-05", "No classed-woodland symbol overlaps the parcel."),
        ],
    )


def corrected_context() -> ParcelContext:
    base = parcel_context()
    return base.model_copy(
        update={
            "other_prescriptions_checklist": [
                base.other_prescriptions_checklist[0].model_copy(
                    update={"comments": None}
                )
            ],
            "citations": [
                citation for citation in base.citations if citation.id != "c-05"
            ],
        }
    )


async def test_node_runs_agent_with_graph_and_doc_tools(
    traces_dir: Path, provider: TracerProvider, stub: StubFactory
) -> None:
    requests: list[dict[str, Any]] = []
    expected = parcel_context()
    expected_final = corrected_context()
    parcel_responses = 0
    verdicts = [
        CitationVerification(
            c_id=f"c-0{index}",
            confirmed=index != 5,
            reasoning=f"checked c-0{index}",
        )
        for index in range(1, 6)
    ]

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        nonlocal parcel_responses
        requests.append(body)
        if "response_format" in body:
            schema_name = body["response_format"]["json_schema"]["name"]
            if schema_name == "ParcelContext":
                content = (
                    expected if parcel_responses == 0 else expected_final
                ).model_dump_json(by_alias=True)
                parcel_responses += 1
            elif schema_name == "CitationVerifications":
                content = json.dumps(
                    {"verifications": [verdict.model_dump() for verdict in verdicts]}
                )
            else:
                raise AssertionError(f"unexpected schema: {schema_name}")
            return completion(
                "m",
                {"role": "assistant", "content": content},
                prompt=100,
                completion_tokens=50,
            )
        if not any(message.get("role") == "tool" for message in body["messages"]):
            return completion(
                "m",
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_parcel",
                            "type": "function",
                            "function": {
                                "name": "read_parcel_info",
                                "arguments": "{}",
                            },
                        },
                        {
                            "id": "call_docs",
                            "type": "function",
                            "function": {
                                "name": "documents_list",
                                "arguments": "{}",
                            },
                        },
                    ],
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
    run_id = uuid4().hex
    parsed, _response, _usage, _turn = await s1_overlay_resolution_node(
        client=client, model="m", metadata={"run_id": run_id}
    )
    provider.force_flush()

    assert isinstance(parsed, VerifiedParcelContext)
    assert parsed.model_dump(exclude={"citations"}) == expected_final.model_dump(
        exclude={"citations"}
    )
    assert [citation.c_id for citation in parsed.citations] == [
        "c-01",
        "c-02",
        "c-03",
        "c-04",
    ]
    assert all(citation.confirmed for citation in parsed.citations)
    assert len(requests) == 7
    assert requests[2]["response_format"]["json_schema"]["name"] == "ParcelContext"
    assert (
        requests[5]["response_format"]["json_schema"]["name"] == "CitationVerifications"
    )
    assert requests[6]["response_format"]["json_schema"]["name"] == "ParcelContext"

    first_messages = requests[0]["messages"]
    assert [message["role"] for message in first_messages] == [
        "system",
        "user",
        "user",
    ]
    assert "regulatory-geometry verification agent" in first_messages[0]["content"]
    assert "CITATION RULES" in first_messages[0]["content"]
    assert first_messages[1]["content"].startswith("response_schema: ")

    tool_names = [tool["function"]["name"] for tool in requests[0]["tools"]]
    assert tool_names == [
        "compute_buildable_envelope",
        "read_parcel_info",
        "read_plan_de_zonage_central_meta",
        "documents_list",
        "pages_read",
        "pages_search",
    ]

    tool_messages = requests[1]["messages"]
    called = [call["function"]["name"] for call in tool_messages[3]["tool_calls"]]
    assert called == ["read_parcel_info", "documents_list"]

    by_id = {
        message["tool_call_id"]: message["content"]
        for message in tool_messages
        if message.get("role") == "tool"
    }
    parcel_parts = by_id["call_parcel"]
    assert any(part["type"] == "image_url" for part in parcel_parts)
    parcel_text = "".join(
        part["text"] for part in parcel_parts if part["type"] == "text"
    )
    assert "metadata_parcel.json" in parcel_text
    assert "zoom-1.png" in parcel_text
    assert "### zoom-0.png" not in parcel_text

    docs_text = by_id["call_docs"]
    assert "oap_91272_20210706" in docs_text

    verification_messages = requests[3]["messages"]
    assert "Verify each citation" in verification_messages[-1]["content"]

    correction_messages = requests[6]["messages"]
    correction_prompt = correction_messages[-1]["content"]
    assert "c-05" in correction_prompt
    assert "checked c-05" in correction_prompt
    assert "Remove each rejected c_id" in correction_prompt
    assert "regulatory-geometry verification agent" in correction_messages[0]["content"]
    assert any(message.get("role") == "tool" for message in correction_messages)
    assert correction_messages[-2]["role"] == "assistant"

    spans = read_spans(traces_dir, resolve=False)
    runs = [
        span
        for span in spans
        if span["name"] == "invoke_agent_structured"
        and span["attributes"].get("langfuse.trace.metadata.run_id") == run_id
    ]
    assert len(runs) == 2
    by_step = {
        span["attributes"]["langfuse.trace.metadata.step"]: span for span in runs
    }
    assert set(by_step) == {"run_agent_structured", "verify_citations"}
    assert {span["attributes"]["langfuse.trace.metadata.step_n"] for span in runs} == {
        "1",
        "2",
    }
    assert len({span["attributes"]["gen_ai.conversation.id"] for span in runs}) == 1
    assert by_step["verify_citations"]["attributes"][
        "gen_ai.conversation.parent_span_id"
    ] == by_step["run_agent_structured"]["context"]["span_id"].removeprefix("0x")

    corrections = [
        span
        for span in spans
        if span["name"] == "complete_structured"
        and span["attributes"].get("langfuse.trace.metadata.step")
        == "correct_citations"
        and span["attributes"].get("langfuse.trace.metadata.run_id") == run_id
    ]
    assert len(corrections) == 1
    correction = corrections[0]
    assert correction["attributes"]["langfuse.trace.metadata.step_n"] == "3"
    assert (
        correction["attributes"]["gen_ai.conversation.id"]
        == by_step["run_agent_structured"]["attributes"]["gen_ai.conversation.id"]
    )
    assert correction["attributes"]["gen_ai.conversation.parent_span_id"] == by_step[
        "run_agent_structured"
    ]["context"]["span_id"].removeprefix("0x")

    final_messages = requests[2]["messages"]
    assert not any(
        isinstance(message.get("content"), str)
        and message["content"].startswith("response_schema: ")
        for message in final_messages
    )


def test_parcel_context_ends_with_citations() -> None:
    assert list(ParcelContext.model_fields)[-1] == "citations"
    assert ParcelContext.model_fields["citations"].is_required()
