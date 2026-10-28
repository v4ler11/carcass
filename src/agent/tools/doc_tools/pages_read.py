import json
from typing import Any, cast

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)

from agent.tools.doc_tools.documents import Document, documents
from chat.tools import Tool, ToolContext
from chat.utils import tool_message


class PagesRead(Tool):
    @property
    def name(self) -> str:
        return "pages_read"

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

        numbers = _parse_pages(args["page_n"], document.n_pages)
        load_images = args.get("load_images", False)
        if not isinstance(load_images, bool):
            raise TypeError("load_images must be a boolean")

        pages = [_page(document, number, load_images=load_images) for number in numbers]
        answer = json.dumps(
            {"document": document.name, "pages": pages}, ensure_ascii=False
        )
        if not load_images:
            return [tool_message(tool_call.id, answer)]

        content: list[dict[str, Any]] = [{"type": "text", "text": answer}]
        for number in numbers:
            for image in document.page(number)["images"]:
                content.append(
                    {"type": "text", "text": f"\nImage {image['path']} (page {number})"}
                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": document.image_url(image["path"])},
                    }
                )
        # The SDK types tool content as text-only; vision-capable
        # OpenAI-compatible servers accept image parts in a tool reply.
        return [
            cast(
                ChatCompletionMessageParam,
                {"role": "tool", "tool_call_id": tool_call.id, "content": content},
            )
        ]

    def into_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Read pages of a document, with image captions by default "
                    "or the images themselves when load_images is true."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_name": {
                            "type": "string",
                            "description": "Document name from documents_list.",
                        },
                        "page_n": {
                            "type": "string",
                            "description": (
                                "One page ('5'), an inclusive range ('10:20' or "
                                "'10-20'), the first pages (':10'), or the last "
                                "pages ('20:')."
                            ),
                        },
                        "load_images": {
                            "type": "boolean",
                            "description": "Attach the page images instead of their captions.",
                            "default": False,
                        },
                    },
                    "required": ["doc_name", "page_n"],
                },
            },
        }


def _parse_pages(page_n: str, n_pages: int) -> range:
    text = page_n.strip()
    if ":" not in text and "-" in text:
        text = text.replace("-", ":")
    if ":" not in text:
        start = end = _page_number(text)
    else:
        start_text, end_text = text.split(":", 1)
        start = _page_number(start_text) if start_text.strip() else 1
        end = _page_number(end_text) if end_text.strip() else n_pages

    if not 1 <= start <= end <= n_pages:
        raise ValueError(
            f"page selection {page_n!r} must be within 1:{n_pages} and not reversed"
        )
    return range(start, end + 1)


def _page_number(text: str) -> int:
    try:
        return int(text)
    except ValueError as error:
        raise ValueError(f"invalid page number {text!r}") from error


def _page(document: Document, number: int, *, load_images: bool) -> dict[str, Any]:
    page = document.page(number)
    if load_images:
        images: list[Any] = [image["path"] for image in page["images"]]
    else:
        images = [
            {"path": image["path"], "caption": document.captions.get(image["path"])}
            for image in page["images"]
        ]
    return {"page": number, "text": page["markdown"], "images": images}
