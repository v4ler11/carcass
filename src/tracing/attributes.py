import json
from typing import Any

from core.config import SERVICE_NAME as DEFAULT_SERVICE_NAME


class Attr:
    SERVICE_NAME = DEFAULT_SERVICE_NAME

    CONVERSATION_ID = "gen_ai.conversation.id"
    TRACE_ATTRIBUTES = "carcass.trace.attributes"
    SESSION_ID = "session.id"
    USER_ID = "user.id"
    HTTP_ROUTE = "http.route"

    OPERATION_NAME = "gen_ai.operation.name"
    REQUEST_MODEL = "gen_ai.request.model"
    RESPONSE_FORMAT = "gen_ai.response.format"
    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"
    TOOL_DEFINITIONS = "gen_ai.tool.definitions"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
    TOOL_CALL_RESULT = "gen_ai.tool.call.result"
    PARENT_TRACE_ID = "gen_ai.conversation.parent_trace_id"
    PARENT_SPAN_ID = "gen_ai.conversation.parent_span_id"

    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
    USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens"
    USAGE_TEXT_INPUT_TOKENS = "gen_ai.usage.text.input_tokens"
    USAGE_IMAGE_INPUT_TOKENS = "gen_ai.usage.image.input_tokens"
    USAGE_AUDIO_INPUT_TOKENS = "gen_ai.usage.audio.input_tokens"
    USAGE_TEXT_OUTPUT_TOKENS = "gen_ai.usage.text.output_tokens"
    USAGE_AUDIO_OUTPUT_TOKENS = "gen_ai.usage.audio.output_tokens"
    USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"

    LANGFUSE_METADATA_PREFIX = "langfuse.trace.metadata."
    LANGFUSE_COST_DETAILS = "langfuse.observation.cost_details"
    DECISION_ANSWERS = "decision.answers"


def attribute_value(value: Any) -> str:
    """OTel attributes hold scalars; metadata values travel as strings."""
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
