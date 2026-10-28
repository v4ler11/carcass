from abc import ABC, abstractmethod
from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

from chat.tools.context import ToolContext


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def execute(
        self,
        ctx: ToolContext,
        tool_call: ChatCompletionMessageToolCall,
        args: dict[str, Any],
    ) -> list[ChatCompletionMessageParam]:
        """Answer one tool call.

        Calls from the same model response run concurrently, so `execute` must
        tolerate overlapping invocations: do not mutate `ctx`, and do not share
        mutable state with sibling calls.

        Return at least one message, or the transcript ends on an assistant
        tool call and several OpenAI-compatible servers reject the next
        request. Rejecting a call (bad args, permissions) is a return, not a
        separate hook: return the reason and do no work.
        """

    @abstractmethod
    def into_chat_tool(self) -> ChatCompletionToolParam: ...
