import asyncio
import json
from collections.abc import Sequence

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from chat.tools.base import Tool
from chat.tools.context import ToolContext
from chat.utils import (
    get_unanswered_tool_calls,
    messages_since_last_user_message,
    tool_message,
)
from tracing.attributes import Attr

tracer = trace.get_tracer(__name__)


def _clip(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


async def _answer(
    ctx: ToolContext,
    tool: Tool | None,
    tool_call: ChatCompletionMessageToolCall,
) -> list[ChatCompletionMessageParam]:
    """Answer one call: never raises for tool misbehaviour, always returns at least one message."""
    name = tool_call.function.name
    arguments = tool_call.function.arguments or "{}"
    with tracer.start_as_current_span(
        "execute_tool",
        kind=SpanKind.INTERNAL,
        attributes={
            Attr.OPERATION_NAME: "execute_tool",
            Attr.TOOL_NAME: name,
            Attr.TOOL_CALL_ID: tool_call.id,
            Attr.TOOL_CALL_ARGUMENTS: arguments,
        },
    ) as span:
        replies: list[ChatCompletionMessageParam]
        if tool is None:
            replies = [tool_message(
                tool_call.id,
                f"Unknown tool '{name}': no such tool is available in this session.",
            )]
        else:
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                replies = [tool_message(
                    tool_call.id,
                    f"Failed to execute '{name}': invalid JSON in arguments: {_clip(arguments)}",
                )]
            else:
                if not isinstance(args, dict):
                    replies = [tool_message(
                        tool_call.id,
                        f"Failed to execute '{name}': arguments must be a JSON object, got {type(args).__name__}.",
                    )]
                else:
                    try:
                        executed = await tool.execute(ctx, tool_call, args)
                    except Exception as error:  # noqa: BLE001 — tool code is foreign, every failure must become a reply
                        replies = [tool_message(
                            tool_call.id,
                            f"Failed to execute '{name}': {type(error).__name__}: {_clip(str(error))}",
                        )]
                    else:
                        replies = executed or [tool_message(tool_call.id, f"'{name}' completed with no output.")]

        span.set_attribute(Attr.TOOL_CALL_RESULT, json.dumps(replies, default=str))
        return replies


async def execute_tools(
    ctx: ToolContext,
    tools: Sequence[Tool],
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Answer every unanswered tool call since the last user message.

    Every call gets exactly one reply: a tool that is missing, misparsed,
    raising or returning nothing still yields a tool message. A transcript
    ending on an assistant tool call is rejected by several OpenAI-compatible
    servers, and the model cannot recover without seeing the error.

    Consecutive tool calls run concurrently under one `asyncio.gather`; replies
    stay in model order, and one tool's failure or exception never affects its
    siblings.
    """
    names = [tool.name for tool in tools]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"duplicate tool names: {duplicated}")

    by_name = {tool.name: tool for tool in tools}
    calls = list(get_unanswered_tool_calls(messages_since_last_user_message(messages)))

    answers = await asyncio.gather(
        *(_answer(ctx, by_name.get(call.function.name), call) for call in calls)
    )
    return [message for answer in answers for message in answer]
