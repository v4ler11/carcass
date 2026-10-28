"""Mistral Document AI (OCR) client: PDF transport, chunking, page merge.

API reference: https://docs.mistral.ai/api/endpoint/ocr
OCR model:     DEFAULT_OCR_MODEL (default mistral-ocr-4-1)
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import (
    DEFAULT_OCR_CHUNK_SIZE,
    DEFAULT_OCR_INLINE_MAX_BYTES,
    DEFAULT_OCR_MODEL,
    DEFAULT_OCR_TIMEOUT_S,
    MISTRAL_API_KEY_ENV,
    mistral_api_key,
)

from .document import OcrDocument, TableRef, page_from_raw
from .pages import Chunk, chunk_pages, validate_page_range

#: Above this size a base64 data URL is wasteful; upload once instead.

Progress = Callable[[str], None]


class OcrError(RuntimeError):
    """Configuration or API failure that the CLI can report verbatim."""


@dataclass(frozen=True)
class OcrSettings:
    model: str = DEFAULT_OCR_MODEL
    #: ``None`` keeps tables inline in the page Markdown; ``"markdown"``/``"html"``
    #: returns them separately and we splice them back in on export.
    table_format: str | None = None
    include_images: bool = True
    include_blocks: bool = False
    extract_header: bool = False
    extract_footer: bool = False
    chunk_size: int = DEFAULT_OCR_CHUNK_SIZE
    transport: str = "auto"  # auto | inline | upload
    timeout_s: float = DEFAULT_OCR_TIMEOUT_S

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk size must be >= 1")
        if self.transport not in {"auto", "inline", "upload"}:
            raise ValueError(f"unknown transport {self.transport!r}")
        if self.table_format not in {None, "markdown", "html"}:
            raise ValueError(f"unknown table format {self.table_format!r}")


@dataclass(frozen=True)
class RequestPlan:
    total_pages: int
    selected: tuple[int, ...]
    chunks: tuple[Chunk, ...]
    transport: str
    pdf_bytes: int

    @property
    def request_count(self) -> int:
        return len(self.chunks)


def resolve_api_key(explicit: str | None = None) -> str:
    key = (explicit or mistral_api_key() or "").strip()
    if not key:
        raise OcrError(
            f"{MISTRAL_API_KEY_ENV} is not set: it is missing from the "
            f"environment and from ./.env. Export it (export "
            f"{MISTRAL_API_KEY_ENV}=...), add {MISTRAL_API_KEY_ENV}=<key> "
            "to .env, or pass --api-key."
        )
    return key


def page_sizes(pdf: Path) -> list[tuple[float, float]]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(pdf))
        return [
            (float(page.mediabox.width), float(page.mediabox.height))
            for page in reader.pages
        ]
    except Exception as exc:  # pypdf raises several unrelated error types
        raise OcrError(f"cannot read PDF {pdf}: {exc}") from exc


def count_pages(pdf: Path) -> int:
    return len(page_sizes(pdf))


def plan_requests(
    pdf: Path,
    *,
    pages: Sequence[int] | None = None,
    settings: OcrSettings | None = None,
) -> RequestPlan:
    """Resolve pages and chunk boundaries without calling the API."""
    settings = settings or OcrSettings()
    pdf = Path(pdf)
    if not pdf.is_file():
        raise OcrError(f"PDF not found: {pdf}")
    size = pdf.stat().st_size
    total = count_pages(pdf)
    selected = list(pages) if pages is not None else list(range(total))
    validate_page_range(selected, total)
    if not selected:
        raise OcrError("no pages selected")

    chunks = chunk_pages(selected, settings.chunk_size)
    if settings.transport == "auto":
        transport = (
            "upload"
            if len(chunks) > 1 or size > DEFAULT_OCR_INLINE_MAX_BYTES
            else "inline"
        )
    else:
        transport = settings.transport
    return RequestPlan(
        total_pages=total,
        selected=tuple(selected),
        chunks=tuple(chunks),
        transport=transport,
        pdf_bytes=size,
    )


def ocr_pdf(
    pdf: Path,
    *,
    api_key: str,
    pages: Sequence[int] | None = None,
    settings: OcrSettings | None = None,
    progress: Progress | None = None,
) -> OcrDocument:
    """OCR ``pdf`` with Mistral and return the merged, page-ordered document."""
    from mistralai.client import Mistral

    settings = settings or OcrSettings()
    pdf = Path(pdf)
    plan = plan_requests(pdf, pages=pages, settings=settings)
    client = Mistral(api_key=api_key)

    collected = []
    usage_pages = 0
    usage_bytes = 0

    with _document_source(client, pdf, plan.transport) as document_url:
        for number, chunk in enumerate(plan.chunks, start=1):
            if progress:
                progress(
                    f"ocr chunk {number}/{len(plan.chunks)}: "
                    f"page(s) {chunk.human} [{plan.transport}]"
                )
            response = client.ocr.process(
                **_request_kwargs(document_url, chunk, settings)
            )
            collected.extend(page_from_raw(raw) for raw in (response.pages or []))
            info = getattr(response, "usage_info", None)
            if info is not None:
                usage_pages += getattr(info, "pages_processed", 0) or 0
                usage_bytes += getattr(info, "doc_size_bytes", 0) or 0

    collected.sort(key=lambda page: page.index)
    return OcrDocument(
        source=pdf,
        model=settings.model,
        pages=collected,
        usage={
            "pages_processed": usage_pages or len(collected),
            "doc_size_bytes": usage_bytes or plan.pdf_bytes,
        },
    )


def ocr_image(
    client: Any, payload: bytes, settings: OcrSettings, *, mime: str = "image/jpeg"
) -> tuple[str, list[TableRef]]:
    """OCR one image crop; returns its Markdown and any separately extracted tables."""
    uri = f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "document": {"type": "image_url", "image_url": uri},
        "include_image_base64": False,
        "include_blocks": settings.include_blocks,
        "timeout_ms": int(settings.timeout_s * 1000),
    }
    if settings.table_format:
        kwargs["table_format"] = settings.table_format
    pages = client.ocr.process(**kwargs).pages or []
    if not pages:
        return "", []
    page = pages[0]
    tables = [
        TableRef(
            id=str(table.id),
            content=table.content,
            format=getattr(table, "format_", None),
        )
        for table in (page.tables or [])
    ]
    return page.markdown or "", tables


def _request_kwargs(
    document_url: str, chunk: Chunk, settings: OcrSettings
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "document": {"type": "document_url", "document_url": document_url},
        "pages": list(chunk.pages),
        "include_image_base64": settings.include_images,
        "include_blocks": settings.include_blocks,
        "timeout_ms": int(settings.timeout_s * 1000),
    }
    if settings.table_format:
        kwargs["table_format"] = settings.table_format
    if settings.extract_header:
        kwargs["extract_header"] = True
    if settings.extract_footer:
        kwargs["extract_footer"] = True
    return kwargs


@contextmanager
def _document_source(client: Any, pdf: Path, transport: str) -> Iterator[str]:
    """Yield a document URL: base64 data URL inline, or a signed upload URL."""
    if transport == "inline":
        payload = base64.b64encode(pdf.read_bytes()).decode("ascii")
        yield f"data:application/pdf;base64,{payload}"
        return

    with pdf.open("rb") as handle:
        uploaded = client.files.upload(
            file={"file_name": pdf.name, "content": handle}, purpose="ocr"
        )
    file_id = uploaded.id
    try:
        signed = client.files.get_signed_url(file_id=file_id, expiry=1)
        yield signed.url
    finally:
        try:
            client.files.delete(file_id=file_id)
        except Exception:  # noqa: BLE001,S110 - cleanup must not mask a conversion failure
            pass
