"""Stage 1: resolve the zone and graphic prescriptions for a parcel.

The model calls the graph and document tools itself, then returns the
structured parcel context under the citation guardrail.
"""

from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletion

from agent.citations import (
    run_agent_with_verified_citations,
    validate_quick_citations,
)
from agent.prompts.quick_citation import CITATION_PROMPT
from agent.prompts.s1_overlay_resolution import SYSTEM_PROMPT, USER_PROMPT
from agent.s1_overlay_resolution.models import ParcelContext, VerifiedParcelContext
from agent.tools import ALL_TOOLS
from chat.clients import Client
from chat.tools import ToolContext
from chat.usage import ChatUsage
from tracing import Turn

TOOLS = ALL_TOOLS


async def s1_overlay_resolution_node(
    *,
    client: Client,
    model: str,
    max_steps: int = 10,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    max_citation_attempts: int = 3,
    **params: Any,
) -> tuple[VerifiedParcelContext, ChatCompletion, ChatUsage, Turn]:
    """Resolve the parcel context, correcting unconfirmed citations in a loop."""
    if max_citation_attempts < 1:
        raise ValueError("max_citation_attempts must be positive")

    stage_metadata: dict[str, Any] = {
        "stage": "s1_overlay_resolution",
        "step": "run_agent_structured",
        "step_n": 1,
        **(metadata or {}),
    }

    (
        parsed,
        verified_citations,
        response,
        usage,
        last_turn,
    ) = await run_agent_with_verified_citations(
        ToolContext(),
        client=client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\n\n{CITATION_PROMPT}",
            },
            {"role": "user", "content": USER_PROMPT},
        ],
        response_model=ParcelContext,
        tools=TOOLS,
        guardrails=[validate_quick_citations],
        max_steps=max_steps,
        max_attempts=max_citation_attempts,
        max_citation_attempts=max_citation_attempts,
        conversation_id=conversation_id,
        parent=parent,
        user_id=user_id,
        metadata=stage_metadata,
        extra_body=extra_body,
        **params,
    )

    resolved = VerifiedParcelContext(
        **parsed.model_dump(exclude={"citations"}),
        citations=verified_citations,
    )

    return resolved, response, usage, last_turn
