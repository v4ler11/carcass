"""Stage 4: synthesize the final study and export a Markdown copy."""

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from agent.citations import validate_quick_citations
from agent.prompts.quick_citation import CITATION_PROMPT
from agent.prompts.s2_fan_out_questions import TARGET_PROGRAM
from agent.prompts.s4_synthesis import (
    MARKDOWN_SYSTEM_PROMPT,
    USER_PROMPT,
    render_markdown_prompt,
    render_system_prompt,
)
from agent.s1_overlay_resolution.models import VerifiedParcelContext
from agent.s2_fan_out_questions.models import QuestionPlan
from agent.s4_synthesis.models import AnswerStore, SynthesisReport
from agent.tools import ALL_TOOLS
from chat.agent import run_agent_structured
from chat.clients import Client
from chat.complete import complete
from chat.tools import ToolContext
from chat.usage import ChatUsage
from chat.utils import first_choice
from tracing import Turn

SECTION_KEYS = (
    "buildability_conclusion",
    "parcel_identification",
    "zoning_context",
    "admissibility",
    "envelope_rules",
    "non_envelope_feasibility",
    "assumptions_and_unresolved",
    "self_correction_log",
)
STATUSES = ("permitted", "restricted", "conditional", "incompatible")
_STATUS_LINE = re.compile(
    r"admissibility_status:\s*(permitted|restricted|conditional|incompatible)\b"
)
_C_ID = re.compile(r"\bc-\d{2,}\b")
_C_ENTRY = re.compile(r"\*\*(c-\d{2,})\*\*")
MARKDOWN_ATTEMPTS = 2


def structure_problems(text: str, use_types: Sequence[str]) -> list[str]:
    """Missing or misordered sections and per-building status lines."""
    problems: list[str] = []
    cursor = 0
    for key in SECTION_KEYS:
        position = text.find(key, cursor)
        if position == -1:
            problems.append(
                f"section key {key!r} is missing or out of order; the required "
                f"order is {', '.join(SECTION_KEYS)}"
            )
        else:
            cursor = position + len(key)
    for building in use_types:
        pattern = re.compile(
            re.escape(building) + r"[^\n]*?" + _STATUS_LINE.pattern,
        )
        if not pattern.search(text):
            problems.append(
                f"no line '{building} — admissibility_status: <one of "
                f"{'|'.join(STATUSES)}>' for the building {building!r}"
            )
    return problems


def structure_guardrail(use_types: Sequence[str]) -> Callable[[BaseModel], list[str]]:
    def guardrail(answer: BaseModel) -> list[str]:
        text = getattr(answer, "text", None)
        if not isinstance(text, str):
            return [f"{type(answer).__name__} must expose a text field"]
        return structure_problems(text, use_types)

    return guardrail


def markdown_problems(markdown: str, citation_ids: Sequence[str]) -> list[str]:
    """Citation ids that the rewrite dropped, invented, or left unlisted."""
    expected = set(citation_ids)
    found = set(_C_ID.findall(markdown))
    listed = set(_C_ENTRY.findall(markdown))
    problems: list[str] = []
    if dropped := sorted(expected - found):
        problems.append("citation ids missing from the Markdown: " + ", ".join(dropped))
    if invented := sorted(found - expected):
        problems.append("citation ids not in the study: " + ", ".join(invented))
    if unlisted := sorted(expected - listed):
        problems.append(
            "citation ids without a '**c-XX**' entry in the Citations section: "
            + ", ".join(unlisted)
        )
    return problems


async def s4_synthesis_node(
    *,
    client: Client,
    model: str,
    parcel_context: VerifiedParcelContext,
    answer_store: AnswerStore,
    target_program: Mapping[str, Any] = TARGET_PROGRAM,
    max_steps: int = 16,
    max_attempts: int = 2,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[SynthesisReport, ChatCompletion, ChatUsage, Turn]:
    """Produce the full cited synthesis without citation verification."""
    stage_metadata: dict[str, Any] = {
        "stage": "s4_synthesis",
        "step": "synthesize",
        "step_n": 1,
        **(metadata or {}),
    }
    system_prompt = (
        render_system_prompt(
            confirmed_parcel_context=parcel_context.model_dump_json(
                indent=2, by_alias=True
            ),
            answer_store=answer_store,
            target_program=dict(target_program),
        )
        + "\n\n"
        + CITATION_PROMPT
    )

    parsed, response, usage, _transcript, turn = await run_agent_structured(
        ToolContext(),
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT},
        ],
        response_model=SynthesisReport,
        tools=ALL_TOOLS,
        guardrails=[
            validate_quick_citations,
            structure_guardrail(cast("Sequence[str]", target_program["use_types"])),
        ],
        max_steps=max_steps,
        max_attempts=max_attempts,
        user_id=user_id,
        metadata=stage_metadata,
        extra_body=extra_body,
        **params,
    )

    return parsed, response, usage, turn


async def render_markdown(
    *,
    client: Client,
    model: str,
    report: SynthesisReport,
    plan: QuestionPlan | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[str, ChatCompletion, ChatUsage]:
    """Rewrite the structured study as Markdown, preserving every citation id.

    The rewrite is a plain completion, so its citation ids are checked here: a
    dropped or invented id is sent back once for correction and then raised.
    """
    citation_ids = [citation.id for citation in report.citations]
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": MARKDOWN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": render_markdown_prompt(report=report, plan=plan),
        },
    ]
    total: ChatUsage | None = None
    for attempt in range(1, MARKDOWN_ATTEMPTS + 1):
        response, usage = await complete(
            client=client,
            model=model,
            messages=messages,
            extra_body=extra_body,
            **params,
        )
        if total is None:
            total = usage
        else:
            total.consume(usage)
        content = first_choice(response).content
        if not content or not content.strip():
            raise ValueError("Markdown rewrite returned empty content")
        markdown = content.strip() + "\n"
        problems = markdown_problems(markdown, citation_ids)
        if not problems:
            return markdown, response, total
        if attempt == MARKDOWN_ATTEMPTS:
            raise ValueError(
                "Markdown rewrite changed the citations: " + "; ".join(problems)
            )
        messages = [
            *messages,
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "The rewrite is not acceptable:\n- "
                    + "\n- ".join(problems)
                    + "\nReturn the complete corrected Markdown."
                ),
            },
        ]
    raise AssertionError("unreachable")
