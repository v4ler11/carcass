"""Generate/verify/repair guardrails for structured and plain agent outputs."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from chat.agent import run_agent, run_agent_structured
from chat.clients import Client
from chat.structured import complete_structured, guardrail_repair_message
from chat.tools import Tool, ToolContext
from chat.usage import ChatUsage
from chat.utils import first_choice
from tracing import Turn

Guardrail = Callable[[BaseModel], list[str]]
TextGuardrail = Callable[[str], list[str]]


async def complete_structured_with_guardrails[T: BaseModel](
    *,
    client: Client,
    model: str,
    messages: list[ChatCompletionMessageParam],
    response_model: type[T],
    guardrails: Sequence[Guardrail],
    max_attempts: int = 5,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[T, ChatCompletion, ChatUsage, Turn]:
    """Generate, verify with every guardrail, and repair until all pass.

    A guardrail returns the list of violations in the parsed answer; an empty
    list means it passed. All violations are fed back together in one user
    repair turn, so several guardrails never cause several serial retries.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    transcript = list(messages)
    usage = ChatUsage(model=model)
    turn = parent
    problems: list[str] = []

    for _ in range(max_attempts):
        parsed, response, attempt_usage, turn = await complete_structured(
            client=client,
            model=model,
            messages=transcript,
            response_model=response_model,
            conversation_id=conversation_id,
            parent=turn,
            user_id=user_id,
            metadata=metadata,
            extra_body=extra_body,
            **params,
        )
        usage.consume(attempt_usage)

        problems = [
            problem for guardrail in guardrails for problem in guardrail(parsed)
        ]
        if not problems:
            return parsed, response, usage, turn

        transcript.append(
            {
                "role": "assistant",
                "content": first_choice(response).content or "",
            }
        )
        transcript.append(guardrail_repair_message(problems))

    raise ValueError(
        "guardrail verification failed after "
        f"{max_attempts} attempt(s): " + "; ".join(problems)
    )


async def run_agent_with_guardrails(
    ctx: ToolContext,
    *,
    client: Client,
    model: str,
    messages: list[ChatCompletionMessageParam],
    guardrails: Sequence[TextGuardrail],
    tools: Sequence[Tool] = (),
    max_steps: int = 10,
    max_attempts: int = 5,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[ChatCompletion, ChatUsage, list[ChatCompletionMessageParam], Turn]:
    """`run_agent` with generate/verify/repair over the final plain text."""
    return await run_agent(
        ctx,
        client=client,
        model=model,
        messages=messages,
        tools=tools,
        max_steps=max_steps,
        guardrails=guardrails,
        max_attempts=max_attempts,
        conversation_id=conversation_id,
        parent=parent,
        user_id=user_id,
        metadata=metadata,
        extra_body=extra_body,
        **params,
    )


async def run_agent_structured_with_guardrails[T: BaseModel](
    ctx: ToolContext,
    *,
    client: Client,
    model: str,
    messages: list[ChatCompletionMessageParam],
    response_model: type[T],
    guardrails: Sequence[Guardrail],
    tools: Sequence[Tool] = (),
    max_steps: int = 10,
    max_attempts: int = 5,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[T, ChatCompletion, ChatUsage, list[ChatCompletionMessageParam], Turn]:
    """`run_agent_structured` with generate/verify/repair over the parsed answer."""
    return await run_agent_structured(
        ctx,
        client=client,
        model=model,
        messages=messages,
        response_model=response_model,
        guardrails=guardrails,
        tools=tools,
        max_steps=max_steps,
        max_attempts=max_attempts,
        conversation_id=conversation_id,
        parent=parent,
        user_id=user_id,
        metadata=metadata,
        extra_body=extra_body,
        **params,
    )
