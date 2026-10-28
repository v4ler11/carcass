import json
import re
from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

from agent.tools.doc_tools.documents import documents
from chat.tools import Tool, ToolContext
from chat.utils import tool_message

CONTEXT_WORDS = 10


class PagesSearch(Tool):
    @property
    def name(self) -> str:
        return "pages_search"

    async def execute(
        self,
        ctx: ToolContext,
        tool_call: ChatCompletionMessageToolCall,
        args: dict[str, Any],
    ) -> list[ChatCompletionMessageParam]:
        document = documents().get(args["doc_name"])
        if document is None:
            raise ValueError(
                f"unknown document {args['doc_name']!r}; call documents_list first"
            )

        expression = args["expression"]
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        n_occurrences = args.get("n_occurrences", 20)
        if not isinstance(n_occurrences, int) or isinstance(n_occurrences, bool):
            raise TypeError("n_occurrences must be an integer")
        if n_occurrences < 1:
            raise ValueError("n_occurrences must be positive")

        try:
            pattern = re.compile(expression)
        except re.error as error:
            raise ValueError(f"invalid regular expression: {error}") from error
        occurrences: list[dict[str, Any]] = []
        pages: list[int] = []
        total = 0

        for page in document.pages:
            page_n = page["page"]
            text = page["markdown"]
            page_matched = False
            for match in pattern.finditer(text):
                total += 1
                page_matched = True
                if len(occurrences) < n_occurrences:
                    occurrences.append(
                        {"page_n": page_n, "text": _context(text, match)}
                    )
            if page_matched:
                pages.append(page_n)

        answer = {
            "n_occurrences": total,
            "pages": pages,
            "occurrences": occurrences,
            "has_more": total > len(occurrences),
        }
        return [tool_message(tool_call.id, json.dumps(answer, ensure_ascii=False))]

    def into_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search a document with a regular expression. Returns the "
                    "total match count, the pages with matches, and short "
                    "contexts around the first matches."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_name": {
                            "type": "string",
                            "description": "Document name from documents_list.",
                        },
                        "expression": {
                            "type": "string",
                            "description": "Python regular expression.",
                        },
                        "n_occurrences": {
                            "type": "integer",
                            "description": "Maximum number of contexts to return.",
                            "default": 20,
                        },
                    },
                    "required": ["doc_name", "expression"],
                },
            },
        }


def _context(text: str, match: re.Match[str]) -> str:
    before = text[: match.start()].split()
    after = text[match.end() :].split()
    parts = [
        "..." if len(before) > CONTEXT_WORDS else "",
        " ".join(before[-CONTEXT_WORDS:]),
        text[match.start() : match.end()].upper(),
        " ".join(after[:CONTEXT_WORDS]),
        "..." if len(after) > CONTEXT_WORDS else "",
    ]
    return " ".join(part for part in parts if part)
