"""Verify the citations of a structured output with a tool-using agent."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from agent.citations.quick_citation import QuickCitation
from chat.agent import run_agent_structured
from chat.clients import Client
from chat.tools import Tool, ToolContext
from tracing import Turn

VERIFICATION_INSTRUCTIONS = """You verify citations in a structured planning analysis.

For every citation below, use the tools to inspect the cited source or image
independently. Do not accept a citation merely because another citation or a
prior answer asserts it. Set confirmed=true only when the evidence directly
supports the claim as stated. Set confirmed=false when the evidence contradicts
it or is insufficient.

For each c_id, write reasoning that names the evidence checked and explains
the confirmation or contradiction. Do not restate the claim without evidence.

Return exactly one verification for every c_id, with no duplicates or
omissions. Return only the structured output."""


class CitationVerification(BaseModel):
    """One independent verdict on a citation."""

    c_id: str = Field(
        ...,
        pattern=r"^c-\d{2,}$",
        description=(
            "Identifier of the citation being verified. It must match exactly "
            "one citation in the input, with no duplicate or missing c_id."
        ),
    )
    confirmed: bool = Field(
        ...,
        description=(
            "True only if the cited evidence directly supports the claim as "
            "stated. False if it contradicts the claim or does not provide "
            "enough evidence to confirm it."
        ),
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description=(
            "Evidence-based explanation for the verdict. Name the source, "
            "image, table row, or metadata field inspected and state what it "
            "shows. For a false verdict, state what the evidence contradicts "
            "or why it is insufficient."
        ),
    )


class CitationVerifications(BaseModel):
    """One verdict for every citation in the input."""

    verifications: list[CitationVerification] = Field(
        ...,
        description=(
            "Exactly one entry per input c_id, in input order. No duplicates, "
            "no omissions, and no c_id that was not in the input."
        ),
    )


class VerifiedQuickCitation(QuickCitation):
    """A citation enriched with its verification result."""

    confirmed: bool = Field(
        ...,
        description="Whether the cited evidence supports the claim.",
    )
    reasoning: str = Field(
        ...,
        min_length=1,
        description="Evidence-based explanation for the verdict.",
    )

    @property
    def c_id(self) -> str:
        return self.id


def _citation_value(citation: Any, key: str) -> Any:
    if isinstance(citation, Mapping):
        return citation.get(key)
    return getattr(citation, key, None)


def _citation_id(citation: Any) -> str:
    c_id = _citation_value(citation, "c_id")
    if isinstance(c_id, str):
        return c_id
    identifier = _citation_value(citation, "id")
    if isinstance(identifier, str):
        return identifier
    raise TypeError("each citation must expose a c_id")


def _citations(value: BaseModel) -> list[Any]:
    citations = getattr(value, "citations", None)
    if not isinstance(citations, list):
        raise TypeError(f"{type(value).__name__} must expose a citations list")
    return citations


def _verification_prompt(citations: Sequence[Any]) -> str:
    blocks: list[str] = []
    for index, citation in enumerate(citations, start=1):
        blocks.append(
            "\n".join(
                [
                    f"{index}. c_id: {_citation_id(citation)}",
                    f"   claim: {_citation_value(citation, 'claim')}",
                    "   verification_instructions: "
                    + str(_citation_value(citation, "verification_instructions")),
                ]
            )
        )
    return (
        "Verify each citation below explicitly. Use the tools to inspect the "
        "named source, image, table, or metadata field.\n\n" + "\n".join(blocks)
    )


def _exact_c_ids(
    expected: Sequence[str],
) -> Callable[[CitationVerifications], list[str]]:
    def guardrail(answer: CitationVerifications) -> list[str]:
        got = [verification.c_id for verification in answer.verifications]
        problems: list[str] = []
        duplicated = sorted({c_id for c_id in got if got.count(c_id) > 1})
        if duplicated:
            problems.append("duplicate c_ids: " + ", ".join(duplicated))
        missing = sorted(set(expected) - set(got))
        if missing:
            problems.append("missing c_ids: " + ", ".join(missing))
        unexpected = sorted(set(got) - set(expected))
        if unexpected:
            problems.append("unexpected c_ids: " + ", ".join(unexpected))
        return problems

    return guardrail


async def verify_citations(
    *,
    client: Client,
    model: str,
    structured_output: BaseModel,
    citation_ids: Sequence[str],
    tools: Sequence[Tool],
    max_steps: int = 10,
    max_attempts: int = 3,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> list[CitationVerification]:
    """Verify only the requested citation IDs with a tool-using agent.

    `citation_ids` selects the citations to verify, in the returned verdict
    order. Other citations in `structured_output` are neither sent to the
    verifier nor verified. Pass `parent` to connect this run to the previous
    pipeline turn; without it, the run starts a new conversation.
    """
    citations = _citations(structured_output)
    by_id: dict[str, Any] = {}
    for citation in citations:
        c_id = _citation_id(citation)
        if c_id in by_id:
            raise ValueError("duplicate citation ids in structured_output")
        by_id[c_id] = citation

    requested = list(citation_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate citation ids requested")
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(
            "requested citation ids are not in structured_output: " + ", ".join(unknown)
        )
    if not requested:
        return []

    selected = [by_id[c_id] for c_id in requested]
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": VERIFICATION_INSTRUCTIONS
            + "\n\n"
            + _verification_prompt(selected),
        },
    ]
    parsed, _response, _usage, _transcript, _turn = await run_agent_structured(
        ToolContext(),
        client=client,
        model=model,
        messages=messages,
        response_model=CitationVerifications,
        tools=tools,
        guardrails=[_exact_c_ids(requested)],
        max_steps=max_steps,
        max_attempts=max_attempts,
        conversation_id=conversation_id,
        parent=parent,
        user_id=user_id,
        metadata=metadata,
        extra_body=extra_body,
        **params,
    )
    verdicts = {
        verification.c_id: verification for verification in parsed.verifications
    }
    return [verdicts[c_id] for c_id in requested]


def apply_verifications(
    structured_output: BaseModel,
    verifications: Sequence[CitationVerification],
) -> list[VerifiedQuickCitation]:
    """Replace each citation with a QuickCitation plus its confirmation."""
    citations = _citations(structured_output)
    by_id = {verification.c_id: verification for verification in verifications}
    if len(by_id) != len(verifications):
        raise ValueError("duplicate verification c_ids")

    expected = {_citation_id(citation) for citation in citations}
    missing = sorted(expected - set(by_id))
    unexpected = sorted(set(by_id) - expected)
    if missing or unexpected:
        raise ValueError(
            "verification c_ids do not match citations: "
            f"missing={missing}, unexpected={unexpected}"
        )

    verified: list[VerifiedQuickCitation] = []
    for citation in citations:
        c_id = _citation_id(citation)
        verification = by_id[c_id]
        data = (
            citation.model_dump(by_alias=True)
            if isinstance(citation, BaseModel)
            else dict(citation)
        )
        verified.append(
            VerifiedQuickCitation(
                **data,
                confirmed=verification.confirmed,
                reasoning=verification.reasoning,
            )
        )
    return verified
