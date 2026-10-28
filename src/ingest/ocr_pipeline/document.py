"""In-memory representation of an OCR run, decoupled from the SDK models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageRef:
    id: str
    top_left_x: int | None = None
    top_left_y: int | None = None
    bottom_right_x: int | None = None
    bottom_right_y: int | None = None
    image_base64: str | None = None


@dataclass(frozen=True)
class TableRef:
    id: str
    content: str
    format: str | None = None


@dataclass
class Page:
    """One OCR'd page. ``index`` is the 0-based index in the source document."""

    index: int
    markdown: str
    images: tuple[ImageRef, ...] = ()
    tables: tuple[TableRef, ...] = ()
    header: str | None = None
    footer: str | None = None
    dimensions: dict[str, Any] | None = None
    blocks: tuple[dict[str, Any], ...] = ()

    @property
    def number(self) -> int:
        """1-based page number, as a human would cite it."""
        return self.index + 1


@dataclass
class OcrDocument:
    source: Path
    model: str
    pages: list[Page] = field(default_factory=list)
    usage: Mapping[str, Any] | None = None
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def image_count(self) -> int:
        return sum(len(p.images) for p in self.pages)

    @property
    def table_count(self) -> int:
        return sum(len(p.tables) for p in self.pages)


def page_from_raw(raw: Any) -> Page:
    images = tuple(
        ImageRef(
            id=str(img.id),
            top_left_x=img.top_left_x,
            top_left_y=img.top_left_y,
            bottom_right_x=img.bottom_right_x,
            bottom_right_y=img.bottom_right_y,
            image_base64=img.image_base64,
        )
        for img in (raw.images or [])
    )
    tables = tuple(
        TableRef(id=str(t.id), content=t.content, format=getattr(t, "format_", None))
        for t in (raw.tables or [])
    )
    dimensions = raw.dimensions.model_dump() if raw.dimensions is not None else None
    blocks = tuple(
        b.model_dump(by_alias=True, exclude_none=True)
        for b in (getattr(raw, "blocks", None) or [])
    )
    return Page(
        index=raw.index,
        markdown=raw.markdown or "",
        images=images,
        tables=tables,
        header=getattr(raw, "header", None),
        footer=getattr(raw, "footer", None),
        dimensions=dimensions,
        blocks=blocks,
    )
