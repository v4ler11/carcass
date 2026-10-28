"""Page selection (1-based, human style) and request chunk planning."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_RANGE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


class PageSpecError(ValueError):
    """Raised when a page selector cannot be understood."""


def parse_page_spec(spec: str | None) -> list[int] | None:
    """Parse ``"1-3,7"`` (1-based, inclusive) into sorted 0-based page indexes.

    ``None`` or a blank string means "every page" and returns ``None``.
    """
    if spec is None or not spec.strip():
        return None
    pages: set[int] = set()
    for part in spec.split(","):
        if not part.strip():
            continue
        match = _RANGE.match(part)
        if match is None:
            raise PageSpecError(
                f"invalid page selector {part.strip()!r}: expected N or N-M"
            )
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        if start < 1:
            raise PageSpecError(
                f"invalid page selector {part.strip()!r}: pages start at 1"
            )
        if end < start:
            raise PageSpecError(
                f"invalid page selector {part.strip()!r}: range ends before it starts"
            )
        pages.update(range(start - 1, end))
    return sorted(pages)


def format_page_spec(pages: Sequence[int]) -> str:
    """Render 0-based indexes as a compact API selector, e.g. ``"0-9,12"``."""
    ordered = sorted(set(pages))
    if not ordered:
        return ""
    parts: list[str] = []
    start = previous = ordered[0]
    for page in ordered[1:]:
        if page == previous + 1:
            previous = page
            continue
        parts.append(_range_text(start, previous))
        start = previous = page
    parts.append(_range_text(start, previous))
    return ",".join(parts)


def _range_text(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def format_page_numbers(pages: Sequence[int]) -> str:
    """Render 0-based indexes as a 1-based human range, e.g. ``"1-10,13"``."""
    return format_page_spec([page + 1 for page in pages])


def validate_page_range(pages: Iterable[int], total_pages: int) -> None:
    """Raise :class:`PageSpecError` if any selected page is outside the document."""
    if total_pages < 1:
        raise PageSpecError("document has no pages")
    outside = sorted({page + 1 for page in pages if page < 0 or page >= total_pages})
    if outside:
        shown = ", ".join(str(page) for page in outside[:5])
        suffix = "…" if len(outside) > 5 else ""
        raise PageSpecError(
            f"page(s) {shown}{suffix} out of range: document has {total_pages} page(s)"
        )


@dataclass(frozen=True)
class Chunk:
    pages: tuple[int, ...]

    @property
    def api_spec(self) -> str:
        return format_page_spec(self.pages)

    @property
    def human(self) -> str:
        return format_page_numbers(self.pages)


def chunk_pages(pages: Sequence[int], size: int) -> list[Chunk]:
    """Split page indexes into contiguous request-sized chunks."""
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    ordered = list(pages)
    return [
        Chunk(tuple(ordered[start : start + size]))
        for start in range(0, len(ordered), size)
    ]
