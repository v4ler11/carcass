import json
from collections.abc import Mapping, Sequence
from typing import Any

from openai.types.decision import Answer, Decision
from openai.types.decision_create_params import InputUnionMember1, Question
from opentelemetry.trace import SpanKind

from chat.clients import Client
from chat.usage import ChatUsage
from tracing import Turn, conversation_turn
from tracing.attributes import Attr


async def decide(
    *,
    client: Client,
    model: str,
    input: str | Sequence[InputUnionMember1],
    questions: Sequence[Question],
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[Decision, ChatUsage, Turn]:
    """Ask typed questions about shared evidence and get probabilities back.

    Unlike `complete`, this returns no text: a `predicate` answer is the
    probability that the condition holds, `choice` and `score` answers carry a
    distribution over your options plus a `confidence`, and any question may
    come back as a `refusal`. Answers arrive in question order; use
    `answers_by_name` to index them.

    `params` are passed to `decisions.create` verbatim (`prompt_cache_key`,
    `prompt_cache_options`, `safety_identifier`, ...).

    The call is a conversation turn: it tags its span with
    `gen_ai.conversation.id`, links to `parent` when given, and returns the
    `Turn` the next call continues. Called inside another turn - a decision
    used as a tool - it joins that turn's conversation and nests in its trace.
    """
    body = {**client.extra_body, **(extra_body or {})}
    with conversation_turn(
        "decisions",
        kind=SpanKind.CLIENT,
        conversation_id=conversation_id,
        parent=parent,
        user_id=user_id,
        metadata=metadata,
        attributes={
            Attr.OPERATION_NAME: "decisions",
            Attr.REQUEST_MODEL: model,
        },
    ) as (span, turn):
        decision = await client.sdk.decisions.create(
            model=model,
            input=input,
            questions=questions,
            extra_body=body or None,
            **params,
        )

        usage = ChatUsage.from_decision(decision.usage, model)
        span.set_attributes(usage.to_attributes())
        answers = json.dumps([answer.model_dump(exclude_none=True, mode="json") for answer in decision.answers])
        span.set_attribute(Attr.DECISION_ANSWERS, answers)
        return decision, usage, turn


def answers_by_name(decision: Decision) -> dict[str, Answer]:
    """Answers keyed by question name, refusals included.

    Unnamed questions are skipped: they can only be matched by their position
    in `decision.answers`.
    """
    by_name: dict[str, Answer] = {}
    for answer in decision.answers:
        if answer.name is None:
            continue
        if answer.name in by_name:
            raise ValueError(f"duplicate answer name: {answer.name!r}")
        by_name[answer.name] = answer
    return by_name
