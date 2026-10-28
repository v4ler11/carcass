"""Run a structured agent and verify or correct its citations in a loop."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from agent.citations.quick_citation import validate_quick_citations
from agent.citations.verify_citations import (
    CitationVerification,
    VerifiedQuickCitation,
    apply_verifications,
    verify_citations,
)
from chat.agent import run_agent_structured
from chat.clients import Client
from chat.structured import complete_structured
from chat.tools import Tool, ToolContext
from chat.usage import ChatUsage
from chat.utils import first_choice
from tracing import Turn


def _citation_key(citation: Any) -> tuple[str, str]:
    return citation.claim, citation.verification_instructions


def _correction_message(
    failed: Sequence[tuple[Any, CitationVerification]],
) -> ChatCompletionMessageParam:
    details = "\n".join(
        f"- {verification.c_id}: {citation.claim}\n  verifier: {verification.reasoning}"
        for citation, verification in failed
    )
    return {
        "role": "user",
        "content": (
            "Independent citation verification rejected these citations:\n\n"
            f"{details}\n\n"
            "Remove each rejected c_id from citations[] and remove every inline "
            "(c-XX)[...] tag with that c_id from all text fields. Remove any "
            "claim that depended on a removed citation. Do not renumber or "
            "reuse the removed c_ids, and do not add new citations or claims. "
            "Keep all other content unchanged and return the complete corrected "
            "structured output."
        ),
    }


async def run_agent_with_verified_citations[T: BaseModel](
    ctx: ToolContext,
    *,
    client: Client,
    model: str,
    messages: list[ChatCompletionMessageParam],
    response_model: type[T],
    tools: Sequence[Tool] = (),
    guardrails: Sequence[Callable[[T], list[str]]] = (),
    max_steps: int = 10,
    max_attempts: int = 2,
    max_citation_attempts: int = 3,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[T, list[VerifiedQuickCitation], ChatCompletion, ChatUsage, Turn]:
    """Run the agent, then verify and correct citations until all pass.

    The initial call uses `metadata` as given. Verification and correction
    calls use the same metadata with `step`/`step_n` advanced. Corrections
    inherit the full agent transcript and remove rejected citation IDs rather
    than guessing replacements.
    """
    if max_citation_attempts < 1:
        raise ValueError("max_citation_attempts must be positive")

    parsed, response, usage, transcript, turn = await run_agent_structured(
        ctx,
        client=client,
        model=model,
        messages=messages,
        response_model=response_model,
        tools=tools,
        guardrails=guardrails,
        max_steps=max_steps,
        max_attempts=max_attempts,
        conversation_id=conversation_id,
        parent=parent,
        user_id=user_id,
        metadata=metadata,
        extra_body=extra_body,
        **params,
    )

    stage_metadata = dict(metadata or {})
    step_n = int(stage_metadata.get("step_n", 1)) + 1
    conversation = list(transcript)
    verified: dict[str, tuple[CitationVerification, tuple[str, str]]] = {}
    last_turn = turn
    corrections = 0

    while True:
        citations = getattr(parsed, "citations", None)
        if not isinstance(citations, list):
            raise TypeError(f"{type(parsed).__name__} must expose a citations list")
        current = {citation.id: citation for citation in citations}
        pending = [
            c_id
            for c_id, citation in current.items()
            if c_id not in verified
            or not verified[c_id][0].confirmed
            or verified[c_id][1] != _citation_key(citation)
        ]
        if not pending:
            break

        verdicts = await verify_citations(
            client=client,
            model=model,
            structured_output=parsed,
            citation_ids=pending,
            tools=tools,
            max_steps=max_steps,
            parent=last_turn,
            user_id=user_id,
            metadata={
                **stage_metadata,
                "step": "verify_citations",
                "step_n": step_n,
            },
            extra_body=extra_body,
            **params,
        )
        step_n += 1

        for verification in verdicts:
            citation = current[verification.c_id]
            verified[verification.c_id] = (
                verification,
                _citation_key(citation),
            )
        failed = [
            (current[verification.c_id], verification)
            for verification in verdicts
            if not verification.confirmed
        ]
        if not failed:
            continue
        if corrections >= max_citation_attempts:
            details = "; ".join(
                f"{verification.c_id}: {verification.reasoning}"
                for _, verification in failed
            )
            raise ValueError(
                "citation verification failed after "
                f"{corrections} correction round(s): {details}"
            )

        correction = _correction_message(failed)
        conversation.append(correction)
        (
            parsed,
            response,
            correction_usage,
            correction_turn,
        ) = await complete_structured(
            client=client,
            model=model,
            messages=conversation,
            response_model=response_model,
            parent=last_turn,
            user_id=user_id,
            metadata={
                **stage_metadata,
                "step": "correct_citations",
                "step_n": step_n,
                "round": corrections + 1,
            },
            extra_body=extra_body,
            **params,
        )
        usage.consume(correction_usage)
        step_n += 1
        corrections += 1

        problems = [
            problem
            for guardrail in (guardrails or [validate_quick_citations])
            for problem in guardrail(parsed)
        ]
        if problems:
            raise ValueError(
                "corrected structured output failed guardrails: " + "; ".join(problems)
            )
        conversation.append(
            {
                "role": "assistant",
                "content": first_choice(response).content
                or parsed.model_dump_json(by_alias=True),
            }
        )
        last_turn = correction_turn

    citations = getattr(parsed, "citations", [])
    final_verdicts: list[CitationVerification] = []
    for citation in citations:
        entry = verified.get(citation.id)
        if (
            entry is None
            or not entry[0].confirmed
            or entry[1] != _citation_key(citation)
        ):
            raise ValueError(
                f"citation {citation.id!r} was not confirmed by verification"
            )
        final_verdicts.append(entry[0])

    return (
        parsed,
        apply_verifications(parsed, final_verdicts),
        response,
        usage,
        last_turn,
    )
