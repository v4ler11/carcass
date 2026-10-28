import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from openai.lib._pydantic import to_strict_json_schema
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ValidationError

from chat.clients import Client
from chat.complete import complete
from chat.usage import ChatUsage
from chat.utils import first_choice
from tracing import Turn, conversation_turn
from tracing.attributes import Attr


def guardrail_repair_message(
    problems: Sequence[str],
) -> ChatCompletionUserMessageParam:
    """The critic's correction request, appended to the conversation."""
    details = "\n".join(f"- {problem}" for problem in problems)
    return {
        "role": "user",
        "content": (
            "Structured-output guardrails failed:\n"
            f"{details}\n\n"
            "Return the complete corrected structured output. Fix only the "
            "violations above and keep the same schema."
        ),
    }


def get_schema(model_class: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_class.__name__,
            "schema": to_strict_json_schema(model_class),
            "strict": True,
        },
    }


def parse_model_output_json[T: BaseModel](output: str, parse_into: type[T]) -> T:
    decoder = json.JSONDecoder(strict=False)
    invalid: ValidationError | None = None

    for candidate in re.finditer(r"\{", output):
        try:
            value, _ = decoder.raw_decode(output, candidate.start())
        except json.JSONDecodeError:
            continue
        try:
            return parse_into.model_validate(value)
        except ValidationError as e:
            invalid = invalid or e

    if invalid is not None:
        raise ValueError(
            f"JSON does not match {parse_into.__name__}: {invalid}"
        ) from invalid
    raise ValueError("No complete JSON object found in output.")


async def complete_structured[T: BaseModel](
    *,
    client: Client,
    model: str,
    messages: list[ChatCompletionMessageParam],
    response_model: type[T],
    max_attempts: int = 2,
    conversation_id: str | None = None,
    parent: Turn | None = None,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> tuple[T, ChatCompletion, ChatUsage, Turn]:
    """Completion constrained to `response_model`.

    Transport retries live on the client (`max_retries`); the second attempt
    here is a repair round that feeds the validation error back to the model
    instead of repeating an identical request.

    The endpoint must accept OpenAI's `response_format` with a `json_schema`
    payload; a server that rejects it fails the call instead of silently
    degrading to unconstrained JSON.

    One `complete_structured` span wraps every attempt, so a `chat` span with
    this parent is a repair round rather than a fresh request.

    The returned `Turn` names this call's span: pass `turn` as `parent` to
    chain it to the step whose answer it validates.
    """
    response_format = get_schema(response_model)

    with conversation_turn(
        "complete_structured",
        conversation_id=conversation_id,
        parent=parent,
        user_id=user_id,
        metadata=metadata,
        attributes={
            Attr.OPERATION_NAME: "chat",
            Attr.REQUEST_MODEL: model,
            Attr.RESPONSE_FORMAT: response_format["type"],
            # The wrapper is the retry/repair chain; the `chat` spans nested in
            # it are the generations.
        },
    ) as (span, turn):
        messages = list(messages)
        usage = ChatUsage(model=model)
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            response, request_usage = await complete(
                client=client,
                model=model,
                messages=messages,
                extra_body=extra_body,
                response_format=response_format,
                **params,
            )
            usage.consume(request_usage)

            content = first_choice(response).content or ""
            try:
                parsed = parse_model_output_json(content, response_model)
            except ValueError as e:
                last_error = e
                if attempt + 1 == max_attempts:
                    break

                assistant_message: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                    "content": content,
                }
                repair_message: ChatCompletionUserMessageParam = {
                    "role": "user",
                    "content": f"That output failed validation:\n{e}\nReturn only corrected JSON matching the schema.",
                }
                messages.append(assistant_message)
                messages.append(repair_message)
                continue

            span.set_attributes(usage.to_attributes())
            return parsed, response, usage, turn

        assert last_error is not None
        raise last_error
