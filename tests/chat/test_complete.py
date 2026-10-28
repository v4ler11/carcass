import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from pydantic import BaseModel

from chat.agent import run_agent
from chat.clients import Client
from chat.complete import complete
from chat.structured import complete_structured
from chat.tools.base import Tool
from chat.tools.context import ToolContext

pytestmark = pytest.mark.expensive

REQUIRED_ENV = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")


class Capital(BaseModel):
    country: str
    city: str


class Lookup(Tool):
    """Deterministic tool: proves the loop actually dispatched a call."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "lookup"

    async def execute(
        self,
        ctx: ToolContext,
        tool_call: ChatCompletionMessageToolCall,
        args: dict[str, Any],
    ) -> list[ChatCompletionMessageParam]:
        self.calls += 1
        return [{"role": "tool", "tool_call_id": tool_call.id, "content": "42"}]

    def into_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Return the number stored under a key.",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": False,
                },
            },
        }


def model_name() -> str:
    return os.environ["OPENAI_MODEL"]


@pytest.fixture
async def client() -> AsyncIterator[Client]:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        pytest.skip(f"set {', '.join(missing)} to run expensive tests")

    client = Client(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    yield client
    await client.aclose()


async def test_complete_returns_content_and_usage(client: Client) -> None:
    message: ChatCompletionMessageParam = {"role": "user", "content": "Reply with the single word: pong"}

    response, usage = await complete(client=client, model=model_name(), messages=[message])

    content = response.choices[0].message.content
    assert content and content.strip(), "model returned no content"
    assert usage.prompt_tokens > 0, "provider reported no prompt usage"
    assert usage.completion_tokens > 0, "provider reported no completion usage"


async def test_complete_structured_returns_the_model(client: Client) -> None:
    message: ChatCompletionMessageParam = {"role": "user", "content": "What is the capital of France?"}

    result, _raw, usage, _turn = await complete_structured(
        client=client,
        model=model_name(),
        messages=[message],
        response_model=Capital,
    )

    assert isinstance(result, Capital)
    assert result.country and result.city
    assert usage.prompt_tokens > 0


async def test_run_agent_calls_the_tool(client: Client) -> None:
    message: ChatCompletionMessageParam = {
        "role": "user",
        "content": "Use the lookup tool with key 'answer', then reply with the number it returned.",
    }
    tool = Lookup()

    response, usage, transcript, _turn = await run_agent(
        ToolContext(),
        client=client,
        model=model_name(),
        messages=[message],
        tools=[tool],
    )

    assert isinstance(response, ChatCompletion)
    assert tool.calls >= 1, "the model never called the tool"
    assert response.choices[0].message.content
    assert len(transcript) > 1, "transcript was not extended"
    assert usage.prompt_tokens > 0
