import json
from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from chat.clients import Client
from chat.usage import ChatUsage
from chat.utils import first_choice
from tracing.attributes import Attr

tracer = trace.get_tracer(__name__)


async def complete(
    *,
    client: Client,
    model: str,
    messages: list[ChatCompletionMessageParam],
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[ChatCompletion, ChatUsage]:
    """One non-streaming chat completion.

    `params` are passed to `chat.completions.create` verbatim (temperature,
    tools, tool_choice, response_format, max_tokens, stop, seed, ...).
    Provider-specific knobs go in `extra_body`; omit a parameter instead of
    passing `None`, which would serialize as JSON null.
    """
    body = {**client.extra_body, **(extra_body or {})}
    with tracer.start_as_current_span(
        "chat",
        kind=SpanKind.CLIENT,
        attributes={
            Attr.OPERATION_NAME: "chat",
            Attr.REQUEST_MODEL: model,
        },
    ) as span:
        input_messages = json.dumps(messages)
        span.set_attribute(Attr.INPUT_MESSAGES, input_messages)
        tools = params.get("tools")
        if tools:
            span.set_attribute(Attr.TOOL_DEFINITIONS, json.dumps(tools, ensure_ascii=False))

        response = await client.sdk.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            extra_body=body or None,
            **params,
        )

        if not isinstance(response, ChatCompletion):
            raise TypeError(f"expected a non-streaming completion, got {type(response).__name__}")

        if response.choices:
            output_messages = json.dumps(
                [first_choice(response).model_dump(exclude_none=True, mode="json")]
            )
            span.set_attribute(Attr.OUTPUT_MESSAGES, output_messages)

        usage = ChatUsage.from_openai(response.usage, model)
        span.set_attributes(usage.to_attributes())

    return response, usage
