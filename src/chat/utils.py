from collections.abc import Iterator
from typing import Any

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolMessageParam,
)


def first_choice(response: ChatCompletion) -> ChatCompletionMessage:
    """The only choice of a non-streaming completion, or a clear failure."""
    if not response.choices:
        raise ValueError("the model returned no choices")
    return response.choices[0].message


def messages_since_last_user_message(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Messages after the most recent user turn - the exchange still in flight."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return messages[index + 1:]
    return list(messages)


def get_unanswered_tool_calls(
    messages: list[ChatCompletionMessageParam],
) -> Iterator[ChatCompletionMessageToolCall]:
    answered: set[str] = set()
    for message in messages:
        if message["role"] == "tool":
            answered.add(message["tool_call_id"])

    for message in messages:
        if message["role"] != "assistant":
            continue
        for raw in message.get("tool_calls") or []:
            tool_call = ChatCompletionMessageToolCall.model_validate(raw)
            if tool_call.id not in answered:
                yield tool_call


def tool_message(tool_call_id: str, content: str) -> ChatCompletionToolMessageParam:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def str_to_streaming(json_str: str) -> str:
    return f"data: {json_str}\n\n"


def extra_body(**kwargs: Any) -> dict[str, Any]:
    """Provider-specific request fields that the OpenAI SDK has no parameter for."""
    return {k: v for k, v in kwargs.items() if v is not None}
