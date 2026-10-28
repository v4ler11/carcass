"""Caption the images extracted by the OCR stage.

Reads every `ingest/<document>/s0_ingest/*.ocr.json`, sends each page image to
a vision model together with the text of the page it belongs to, and writes
`ingest/<document>/s1_caption/captions.json`: image path (`p0005-img-3.jpeg`)
to description.
"""

import asyncio
import base64
import json
import mimetypes
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from openai.types.chat import ChatCompletionUserMessageParam
from pydantic import BaseModel

import core.logger  # noqa: F401 - importing configures loguru
from chat.clients import Client
from chat.structured import complete_structured
from core import BASE_DIR
from core.config import (
    DEFAULT_CAPTION_MODEL,
    OPENROUTER_EXTRA_BODY,
    OPENROUTER_URL,
    TRACES_DIR,
    require_openrouter_api_key,
)
from tracing import configure, shutdown

INGEST_DIR = BASE_DIR / "ingest"
DEFAULT_DOCUMENT = "oap"


def document_paths(document: str) -> tuple[Path, Path]:
    root = INGEST_DIR / document
    return root / "s0_ingest", root / "s1_caption" / "captions.json"


PROMPT = """Describe the image extracted from page {page} of a document.

Page context:
<page>
{context}
</page>

In one to three sentences: what kind of visual it is (map, aerial photograph,
diagram, chart, plan, logo, ...), what it shows, and any labels, legends, or
places visible in it. Use the page context to interpret what the image
illustrates; describe only what can be seen. Write in the language of the page
context."""


class ImageCaption(BaseModel):
    description: str


@dataclass(frozen=True, slots=True)
class PageImage:
    path: str
    page: int
    context: str
    file: Path


def page_images(ocr_path: Path) -> Iterator[PageImage]:
    document = json.loads(ocr_path.read_text(encoding="utf-8"))
    assets = ocr_path.with_name(ocr_path.name.removesuffix(".ocr.json") + ".assets")
    for page in document["pages"]:
        for image in page["images"]:
            yield PageImage(
                path=image["path"],
                page=page["page"],
                context=page["markdown"],
                file=assets / image["path"],
            )


def user_message(image: PageImage) -> ChatCompletionUserMessageParam:
    encoded = base64.b64encode(image.file.read_bytes()).decode("ascii")
    media_type = mimetypes.guess_type(image.path)[0] or "application/octet-stream"
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": PROMPT.format(page=image.page, context=image.context),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
            },
        ],
    }


async def caption(*, client: Client, model: str, image: PageImage) -> tuple[str, str]:
    parsed, _response, _usage, _turn = await complete_structured(
        client=client,
        model=model,
        messages=[user_message(image)],
        response_model=ImageCaption,
        metadata={"stage": "s1_caption", "page": image.page, "image": image.path},
    )
    return image.path, parsed.description


async def caption_images(
    *,
    client: Client,
    ocr_dir: Path,
    captions_path: Path,
    model: str = DEFAULT_CAPTION_MODEL,
) -> dict[str, str]:
    images = [
        image
        for ocr_path in sorted(ocr_dir.glob("*.ocr.json"))
        for image in page_images(ocr_path)
    ]
    captions = dict(
        await asyncio.gather(
            *(caption(client=client, model=model, image=image) for image in images)
        )
    )

    captions_path.parent.mkdir(parents=True, exist_ok=True)
    captions_path.write_text(
        json.dumps(captions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("captioned {} images into {}", len(captions), captions_path)
    return captions


async def main(document: str = DEFAULT_DOCUMENT) -> None:
    ocr_dir, captions_path = document_paths(document)
    configure(TRACES_DIR)
    try:
        async with Client(
            base_url=OPENROUTER_URL,
            api_key=require_openrouter_api_key(),
            extra_body=OPENROUTER_EXTRA_BODY,
        ) as client:
            await caption_images(
                client=client, ocr_dir=ocr_dir, captions_path=captions_path
            )
    finally:
        shutdown()


def run() -> None:
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOCUMENT))


if __name__ == "__main__":
    run()
