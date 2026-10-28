from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from chat.utils import get_unanswered_tool_calls, messages_since_last_user_message


def user(content: str = "go") -> ChatCompletionUserMessageParam:
    return {"role": "user", "content": content}


def assistant(*call_ids: str) -> ChatCompletionAssistantMessageParam:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            for call_id in call_ids
        ],
    }


def tool(call_id: str, content: str = "ok") -> ChatCompletionToolMessageParam:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def segment(input_messages: list[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
    return messages_since_last_user_message(input_messages)


def unanswered(input_messages: list[ChatCompletionMessageParam]) -> list[str]:
    return [call.id for call in get_unanswered_tool_calls(input_messages)]


def test_returns_only_the_exchange_after_the_last_user_message() -> None:
    transcript: list[ChatCompletionMessageParam] = [
        user("first"),
        assistant("a"),
        tool("a"),
        user("second"),
        assistant("b"),
    ]

    assert segment(transcript) == transcript[4:]


def test_keeps_forward_order() -> None:
    transcript: list[ChatCompletionMessageParam] = [user(), assistant("a"), tool("a"), assistant("b"), tool("b")]

    assert segment(transcript) == transcript[1:]


def test_last_message_being_the_user_leaves_nothing() -> None:
    assert segment([assistant("a"), user()]) == []


def test_returns_everything_when_no_user_message_exists() -> None:
    transcript: list[ChatCompletionMessageParam] = [assistant("a"), tool("a")]

    assert segment(transcript) == transcript


def test_empty_transcript() -> None:
    assert segment([]) == []


def test_result_is_detached_from_the_input() -> None:
    transcript: list[ChatCompletionMessageParam] = [assistant("a"), tool("a")]

    result = segment(transcript)
    result.append(user("injected"))

    assert len(transcript) == 2
    assert transcript[-1]["role"] == "tool"


def test_yields_unanswered_calls_in_model_order() -> None:
    transcript: list[ChatCompletionMessageParam] = [user(), assistant("a", "b"), assistant("c")]

    assert unanswered(transcript) == ["a", "b", "c"]


def test_skips_answered_calls() -> None:
    transcript: list[ChatCompletionMessageParam] = [
        user(),
        assistant("a", "b"),
        tool("a"),
        assistant("c"),
        tool("c"),
    ]

    assert unanswered(transcript) == ["b"]


def test_returns_nothing_when_every_call_is_answered() -> None:
    transcript: list[ChatCompletionMessageParam] = [user(), assistant("a", "b"), tool("a"), tool("b")]

    assert unanswered(transcript) == []


def test_ignores_messages_without_tool_calls() -> None:
    plain: ChatCompletionAssistantMessageParam = {"role": "assistant", "content": "no calls here"}

    assert unanswered([user(), plain]) == []
