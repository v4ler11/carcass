"""Stage 2: turn the verified parcel context into an atomic question plan."""

from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletion

from agent.prompts.s2_fan_out_questions import (
    SEED_CHECKLIST,
    USER_PROMPT,
    render_system_prompt,
)
from agent.s1_overlay_resolution.models import VerifiedParcelContext
from agent.s2_fan_out_questions.models import QuestionPlan
from agent.tools import ALL_TOOLS
from chat.agent import run_agent_structured
from chat.clients import Client
from chat.tools import ToolContext
from chat.usage import ChatUsage
from tracing import Turn


def _validate_question_plan(
    plan: QuestionPlan, *, has_unresolved_flags: bool
) -> list[str]:
    problems: list[str] = []
    categories = {question.category for question in plan.questions}
    required = [
        item["category"]
        for item in SEED_CHECKLIST
        if item["category"] != "unresolved_flag_followups" or has_unresolved_flags
    ]
    missing = [category for category in required if category not in categories]
    if missing:
        problems.append("missing seed categories: " + ", ".join(missing))

    ids = [question.id for question in plan.questions]
    duplicated = sorted(
        {question_id for question_id in ids if ids.count(question_id) > 1}
    )
    if duplicated:
        problems.append("duplicate question ids: " + ", ".join(duplicated))
    unknown = sorted(
        {
            dependency
            for question in plan.questions
            for dependency in question.depends_on
            if dependency not in ids
        }
    )
    if unknown:
        problems.append("unknown depends_on ids: " + ", ".join(unknown))
    self_dependencies = sorted(
        question.id for question in plan.questions if question.id in question.depends_on
    )
    if self_dependencies:
        problems.append(
            "self-referential depends_on ids: " + ", ".join(self_dependencies)
        )
    return problems


async def s2_fan_out_questions_node(
    *,
    client: Client,
    model: str,
    parcel_context: VerifiedParcelContext,
    max_steps: int = 10,
    max_attempts: int = 2,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[QuestionPlan, ChatCompletion, ChatUsage, Turn]:
    """Plan atomic downstream questions from the verified S1 parcel context."""
    stage_metadata: dict[str, Any] = {
        "stage": "s2_fan_out_questions",
        "step": "fan_out_questions",
        "step_n": 1,
        **(metadata or {}),
    }

    parsed, response, usage, _transcript, turn = await run_agent_structured(
        ToolContext(),
        client=client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": render_system_prompt(
                    confirmed_parcel_context=parcel_context.model_dump_json(
                        indent=2, by_alias=True
                    )
                ),
            },
            {"role": "user", "content": USER_PROMPT},
        ],
        response_model=QuestionPlan,
        tools=ALL_TOOLS,
        guardrails=[
            lambda plan: _validate_question_plan(
                plan,
                has_unresolved_flags=bool(parcel_context.unresolved_flags),
            )
        ],
        max_steps=max_steps,
        max_attempts=max_attempts,
        conversation_id=conversation_id,
        parent=parent,
        user_id=user_id,
        metadata=stage_metadata,
        extra_body=extra_body,
        **params,
    )
    return parsed, response, usage, turn
