import json
from typing import Any

import pytest
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel

from agent.citations import (
    CitationVerification,
    QuickCitation,
    VerifiedQuickCitation,
    apply_verifications,
    verify_citations,
)
from chat.tools import Tool, ToolContext
from tests.stub_server import StubFactory, completion


class CitedOutput(BaseModel):
    citations: list[QuickCitation]


class Lookup(Tool):
    @property
    def name(self) -> str:
        return "lookup"

    async def execute(
        self, ctx: ToolContext, tool_call: Any, args: dict[str, Any]
    ) -> list[ChatCompletionMessageParam]:
        return [
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "The zoning sheet shows UMG.",
            }
        ]

    def into_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Check a cited source.",
                "parameters": {"type": "object", "properties": {}},
            },
        }


def output() -> CitedOutput:
    return CitedOutput(
        citations=[
            QuickCitation(
                c_id="c-01",
                claim="The parcel is in zone UMG.",
                verification_instructions=(
                    "VISUAL | zoom-1.png | bbox=[0,0,10,10] | check the zone color"
                ),
            ),
            QuickCitation(
                c_id="c-02",
                claim="The residual is below 3 m.",
                verification_instructions=(
                    "METADATA | georeference.residual_mean_m | 1.87"
                ),
            ),
        ]
    )


async def test_verify_citations_returns_one_verdict_per_c_id(
    stub: StubFactory,
) -> None:
    requests: list[dict[str, Any]] = []
    verdicts = [
        CitationVerification(
            c_id="c-01",
            confirmed=True,
            reasoning="The zoning sheet shows the UMG color over the parcel.",
        ),
        CitationVerification(
            c_id="c-02",
            confirmed=False,
            reasoning="The metadata residual is 1.87 m, but the cited field is absent.",
        ),
    ]

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if "response_format" in body:
            return completion(
                "m",
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "verifications": [
                                verdict.model_dump() for verdict in verdicts
                            ]
                        }
                    ),
                },
                prompt=10,
                completion_tokens=2,
            )
        if not any(message.get("role") == "tool" for message in body["messages"]):
            return completion(
                "m",
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                prompt=10,
                completion_tokens=2,
            )
        return completion(
            "m",
            {"role": "assistant", "content": "draft"},
            prompt=10,
            completion_tokens=2,
        )

    client = await stub(handler)
    result = await verify_citations(
        client=client,
        model="m",
        structured_output=output(),
        citation_ids=["c-01", "c-02"],
        tools=[Lookup()],
    )

    assert result == verdicts
    assert len(requests) == 3
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["lookup"]
    prompt = requests[0]["messages"][-1]["content"]
    assert "c-01" in prompt and "c-02" in prompt
    assert "The parcel is in zone UMG." in prompt
    assert "Verify each citation" in prompt
    assert requests[2]["response_format"]["json_schema"]["name"] == (
        "CitationVerifications"
    )


def test_apply_verifications_enriches_each_quick_citation() -> None:
    source = output()
    verdicts = [
        CitationVerification(
            c_id="c-01",
            confirmed=True,
            reasoning="The zoning sheet shows UMG.",
        ),
        CitationVerification(
            c_id="c-02",
            confirmed=False,
            reasoning="The residual field does not support the claim.",
        ),
    ]

    verified = apply_verifications(source, verdicts)

    assert all(isinstance(citation, VerifiedQuickCitation) for citation in verified)
    assert [citation.c_id for citation in verified] == ["c-01", "c-02"]
    assert verified[0].claim == "The parcel is in zone UMG."
    assert verified[0].confirmed is True
    assert verified[1].confirmed is False
    assert verified[1].reasoning == "The residual field does not support the claim."


async def test_verify_citations_returns_empty_without_calling_the_model(
    stub: StubFactory,
) -> None:
    def handler(body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("model must not be called without citations")

    client = await stub(handler)
    result = await verify_citations(
        client=client,
        model="m",
        structured_output=CitedOutput(citations=[]),
        citation_ids=[],
        tools=[],
    )

    assert result == []


async def test_verify_citations_repairs_missing_c_ids(stub: StubFactory) -> None:
    requests: list[dict[str, Any]] = []
    verdicts = [
        CitationVerification(
            c_id="c-01", confirmed=True, reasoning="Supported by the zoning sheet."
        ),
        CitationVerification(
            c_id="c-02", confirmed=True, reasoning="Supported by the metadata field."
        ),
    ]

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if "response_format" in body:
            submitted = verdicts if len(requests) > 2 else verdicts[:1]
            return completion(
                "m",
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "verifications": [
                                verdict.model_dump() for verdict in submitted
                            ]
                        }
                    ),
                },
                prompt=10,
                completion_tokens=2,
            )
        return completion(
            "m",
            {"role": "assistant", "content": "draft"},
            prompt=10,
            completion_tokens=2,
        )

    client = await stub(handler)
    result = await verify_citations(
        client=client,
        model="m",
        structured_output=output(),
        citation_ids=["c-01", "c-02"],
        tools=[],
        max_attempts=2,
    )

    assert result == verdicts
    assert len(requests) == 3
    assert "missing c_ids: c-02" in requests[2]["messages"][-1]["content"]


async def test_verify_citations_only_sends_requested_ids(stub: StubFactory) -> None:
    requests: list[dict[str, Any]] = []
    verdicts = [
        CitationVerification(
            c_id="c-02",
            confirmed=True,
            reasoning="The metadata field shows 1.87 m.",
        )
    ]

    def handler(body: dict[str, Any]) -> dict[str, Any]:
        requests.append(body)
        if "response_format" in body:
            return completion(
                "m",
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "verifications": [
                                verdict.model_dump() for verdict in verdicts
                            ]
                        }
                    ),
                },
                prompt=10,
                completion_tokens=2,
            )
        return completion(
            "m",
            {"role": "assistant", "content": "draft"},
            prompt=10,
            completion_tokens=2,
        )

    client = await stub(handler)
    result = await verify_citations(
        client=client,
        model="m",
        structured_output=output(),
        citation_ids=["c-02"],
        tools=[],
    )

    assert result == verdicts
    prompt = requests[0]["messages"][-1]["content"]
    assert "c-02" in prompt
    assert "c-01" not in prompt
    assert "The parcel is in zone UMG." not in prompt


async def test_verify_citations_rejects_unknown_requested_ids(
    stub: StubFactory,
) -> None:
    def handler(body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("model must not be called for unknown citation ids")

    client = await stub(handler)
    with pytest.raises(ValueError, match="not in structured_output: c-99"):
        await verify_citations(
            client=client,
            model="m",
            structured_output=output(),
            citation_ids=["c-99"],
            tools=[],
        )
