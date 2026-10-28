import json
from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

from agent.tools.doc_tools.documents import documents
from chat.tools import Tool, ToolContext
from chat.utils import tool_message


class DocumentsList(Tool):
    @property
    def name(self) -> str:
        return "documents_list"

    async def execute(
        self,
        ctx: ToolContext,
        tool_call: ChatCompletionMessageToolCall,
        args: dict[str, Any],
    ) -> list[ChatCompletionMessageParam]:
        listing = [
            {"name": document.name, "n_pages": document.n_pages}
            for document in documents().values()
        ]
        return [tool_message(tool_call.id, json.dumps(listing, ensure_ascii=False))]

    def into_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "List the documents available for reading, with their page counts.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
