"""OCR documents and image captions shared by the doc tools."""

import base64
import json
import mimetypes
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from core import BASE_DIR

INGEST_DIR = BASE_DIR / "ingest"


@dataclass(frozen=True, slots=True)
class Document:
    name: str
    path: Path
    pages: list[dict[str, Any]]
    captions: dict[str, str]

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def assets_dir(self) -> Path:
        return self.path.with_name(f"{self.name}.assets")

    def page(self, number: int) -> dict[str, Any]:
        for page in self.pages:
            if page["page"] == number:
                return page
        raise ValueError(f"document {self.name!r} has no page {number}")

    def image_url(self, image_path: str) -> str:
        data = base64.b64encode((self.assets_dir / image_path).read_bytes()).decode(
            "ascii"
        )
        media_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
        return f"data:{media_type};base64,{data}"


@cache
def documents() -> dict[str, Document]:
    found: dict[str, Document] = {}
    for path in sorted(INGEST_DIR.glob("*/s0_ingest/*.ocr.json")):
        name = path.name.removesuffix(".ocr.json")
        pages = json.loads(path.read_text(encoding="utf-8"))["pages"]
        captions_path = path.parent.parent / "s1_caption" / "captions.json"
        captions = (
            json.loads(captions_path.read_text(encoding="utf-8"))
            if captions_path.is_file()
            else {}
        )
        found[name] = Document(name=name, path=path, pages=pages, captions=captions)
    return found
