"""Files under `ingest/plu_graph` shared by the graph tools."""

import base64
import mimetypes
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from openai.types.chat import ChatCompletionMessageParam

from chat.utils import tool_message
from core import BASE_DIR

PLU_GRAPH_DIR = BASE_DIR / "ingest" / "plu_graph"
IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".webp"}


def read_files(
    tool_call_id: str,
    directory: Path,
    *,
    skip: Iterable[str] = (),
) -> list[ChatCompletionMessageParam]:
    skipped = set(skip)
    content: list[dict[str, Any]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name in skipped:
            continue

        if path.suffix.lower() in IMAGE_SUFFIXES:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            media_type = (
                mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            content.append({"type": "text", "text": f"\n\n### {path.name}\n"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                }
            )
        else:
            text = path.read_text(encoding="utf-8")
            content.append({"type": "text", "text": f"\n\n### {path.name}\n{text}"})

    if not content:
        return [tool_message(tool_call_id, f"No files in {directory}.")]

    # The SDK types tool content as text-only; vision-capable
    # OpenAI-compatible servers accept image parts in a tool reply.
    return [
        cast(
            ChatCompletionMessageParam,
            {"role": "tool", "tool_call_id": tool_call_id, "content": content},
        )
    ]
