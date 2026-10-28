import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
)
from pydantic import BaseModel

from chat.clients import Client
from chat.complete import complete
from chat.structured import (
    complete_structured,
    get_schema,
    guardrail_repair_message,
)
from chat.tools.base import Tool
from chat.tools.context import ToolContext
from chat.tools.execute import execute_tools
from chat.usage import ChatUsage
from chat.utils import first_choice
from tracing import Turn, conversation_turn
from tracing.attributes import Attr


def _function_tool_calls(
    message: ChatCompletionMessage,
) -> list[ChatCompletionMessageToolCall]:
    return [
        tool_call
        for tool_call in message.tool_calls or []
        if isinstance(tool_call, ChatCompletionMessageToolCall)
    ]


def _to_assistant_param(
    message: ChatCompletionMessage,
) -> ChatCompletionAssistantMessageParam:
    result: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "content": message.content,
    }
    tool_calls: list[ChatCompletionMessageToolCallParam] = []
    for tool_call in _function_tool_calls(message):
        tool_calls.append(
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
        )
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


async def _run_tool_steps(
    ctx: ToolContext,
    *,
    client: Client,
    model: str,
    transcript: list[ChatCompletionMessageParam],
    usage: ChatUsage,
    tools: Sequence[Tool],
    max_steps: int,
    extra_body: Mapping[str, Any] | None,
    params: Mapping[str, Any],
) -> ChatCompletion:
    """Run plain tool rounds until the model answers, or the budget runs out."""
    tool_defs = [tool.into_chat_tool() for tool in tools]

    for _ in range(max_steps):
        step_params: dict[str, Any] = dict(params)
        if tool_defs:
            step_params["tools"] = tool_defs
            step_params.setdefault("tool_choice", "auto")

        response, step_usage = await complete(
            client=client,
            model=model,
            messages=transcript,
            extra_body=extra_body,
            **step_params,
        )
        usage.consume(step_usage)

        message = first_choice(response)
        transcript.append(_to_assistant_param(message))

        if not _function_tool_calls(message):
            return response

        transcript.extend(await execute_tools(ctx, tools, transcript))

    # Tool budget exhausted: force one final answer, without tools.
    response, step_usage = await complete(
        client=client, model=model, messages=transcript, extra_body=extra_body, **params
    )
    usage.consume(step_usage)
    transcript.append(_to_assistant_param(first_choice(response)))

    return response


async def run_agent(
    ctx: ToolContext,
    *,
    client: Client,
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: Sequence[Tool] = (),
    max_steps: int = 10,
    guardrails: Sequence[Callable[[str], list[str]]] = (),
    max_attempts: int = 5,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[ChatCompletion, ChatUsage, list[ChatCompletionMessageParam], Turn]:
    """Run the tool loop until the model answers without calling a tool.

    The loop returns the model's own answer and never reshapes it. For a
    validated final answer, use `run_agent_structured` instead of composing
    this with `complete_structured` yourself.

    `guardrails` inspect the final text and return violations. All violations
    are fed back in one repair turn; `max_attempts` bounds the answer attempts,
    including the first. Tool rounds never carry `response_format`: strict JSON
    and tool calling conflict on most OpenAI-compatible servers. The input list
    is copied, never mutated.

    The run is one trace under `invoke_agent`, and every span inside it carries
    `gen_ai.conversation.id`. The returned `Turn` names the run's span: pass
    `turn` as `parent` to the next run to continue the same conversation. Two
    runs that pass the same parent are branches of one conversation.
    """
    with conversation_turn(
        "invoke_agent",
        conversation_id=conversation_id,
        parent=parent,
        user_id=user_id,
        metadata=metadata,
        attributes={
            Attr.OPERATION_NAME: "invoke_agent",
            Attr.REQUEST_MODEL: model,
        },
    ) as (_span, turn):
        transcript = list(messages)
        usage = ChatUsage(model=model)
        response = await _run_tool_steps(
            ctx,
            client=client,
            model=model,
            transcript=transcript,
            usage=usage,
            tools=tools,
            max_steps=max_steps,
            extra_body=extra_body,
            params=params,
        )

        if guardrails:
            if max_attempts < 1:
                raise ValueError("max_attempts must be positive")
            problems: list[str] = []
            for attempt in range(max_attempts):
                content = first_choice(response).content or ""
                problems = [
                    problem
                    for guardrail in guardrails
                    for problem in guardrail(content)
                ]
                if not problems:
                    return response, usage, transcript, turn
                if attempt + 1 == max_attempts:
                    break

                transcript.append(guardrail_repair_message(problems))
                response, step_usage = await complete(
                    client=client,
                    model=model,
                    messages=transcript,
                    extra_body=extra_body,
                    **params,
                )
                usage.consume(step_usage)
                transcript.append(_to_assistant_param(first_choice(response)))

            raise ValueError(
                "agent guardrail verification failed after "
                f"{max_attempts} attempt(s): " + "; ".join(problems)
            )

        return response, usage, transcript, turn


def _with_response_schema(
    messages: list[ChatCompletionMessageParam],
    response_model: type[BaseModel],
) -> tuple[list[ChatCompletionMessageParam], ChatCompletionMessageParam]:
    """Put the target schema after the system messages, before the request."""
    schema = get_schema(response_model)["json_schema"]["schema"]
    content = "response_schema: " + json.dumps(
        schema, ensure_ascii=False, separators=(",", ":")
    )
    schema_message: ChatCompletionMessageParam = {"role": "user", "content": content}

    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        index += 1

    return [*messages[:index], schema_message, *messages[index:]], schema_message


STRUCTURED_ANSWER_PROMPT = (
    "Return the final answer now as JSON matching the required response schema."
)


async def run_agent_structured[T: BaseModel](
    ctx: ToolContext,
    *,
    client: Client,
    model: str,
    messages: list[ChatCompletionMessageParam],
    response_model: type[T],
    tools: Sequence[Tool] = (),
    max_steps: int = 10,
    guardrails: Sequence[Callable[[T], list[str]]] = (),
    max_attempts: int = 2,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[T, ChatCompletion, ChatUsage, list[ChatCompletionMessageParam], Turn]:
    """Run the tool loop, then return a validated structured answer.

    A user message `response_schema: <json>` is inserted after the leading
    system messages so the model knows the target shape while choosing tools.
    Tool rounds are plain completions. Once the model stops calling tools, its
    plain answer is appended, the schema message is removed, and a user turn
    asks for the final answer so the request does not end on an assistant
    message (required by some providers). `complete_structured` requests the
    same information as `response_model`. The structured reply is appended too,
    so the returned transcript ends with the validated answer.

    `max_steps` bounds the plain tool rounds. Without guardrails,
    `max_attempts` bounds the structured call's JSON validation/repair
    attempts. With guardrails, it bounds the generate/verify/repair attempts
    instead, and each structured call still gets its normal JSON repair.
    The final structured calls nest under the run's span and inherit its
    conversation, metadata, and user.
    """
    with conversation_turn(
        "invoke_agent_structured",
        conversation_id=conversation_id,
        parent=parent,
        user_id=user_id,
        metadata=metadata,
        attributes={
            Attr.OPERATION_NAME: "invoke_agent_structured",
            Attr.REQUEST_MODEL: model,
        },
    ) as (_span, turn):
        transcript, schema_message = _with_response_schema(messages, response_model)
        usage = ChatUsage(model=model)
        await _run_tool_steps(
            ctx,
            client=client,
            model=model,
            transcript=transcript,
            usage=usage,
            tools=tools,
            max_steps=max_steps,
            extra_body=extra_body,
            params=params,
        )
        final_messages = [
            message for message in transcript if message is not schema_message
        ]
        final_messages.append({"role": "user", "content": STRUCTURED_ANSWER_PROMPT})

        if guardrails:
            if max_attempts < 1:
                raise ValueError("max_attempts must be positive")
            problems: list[str] = []
            for attempt in range(max_attempts):
                (
                    parsed,
                    response,
                    structured_usage,
                    _structured_turn,
                ) = await complete_structured(
                    client=client,
                    model=model,
                    messages=final_messages,
                    response_model=response_model,
                    extra_body=extra_body,
                    **params,
                )
                usage.consume(structured_usage)
                problems = [
                    problem for guardrail in guardrails for problem in guardrail(parsed)
                ]
                if not problems:
                    final_messages.append(_to_assistant_param(first_choice(response)))
                    return parsed, response, usage, final_messages, turn
                if attempt + 1 == max_attempts:
                    break

                final_messages.append(
                    {
                        "role": "assistant",
                        "content": first_choice(response).content or "",
                    }
                )
                final_messages.append(guardrail_repair_message(problems))

            raise ValueError(
                "structured agent guardrail verification failed after "
                f"{max_attempts} attempt(s): " + "; ".join(problems)
            )

        (
            parsed,
            response,
            structured_usage,
            _structured_turn,
        ) = await complete_structured(
            client=client,
            model=model,
            messages=final_messages,
            response_model=response_model,
            max_attempts=max_attempts,
            extra_body=extra_body,
            **params,
        )
        usage.consume(structured_usage)
        final_messages.append(_to_assistant_param(first_choice(response)))

        return parsed, response, usage, final_messages, turn
