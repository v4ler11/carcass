"""Render OCR pages as one Markdown file with a sidecar asset directory."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .document import ImageRef, OcrDocument, Page, TableRef

_DATA_URI = re.compile(
    r"^data:(?P<mime>[-\w.]+/[-\w.+]+);base64,(?P<data>.*)$", re.DOTALL
)
_MIME_EXT = {
    "image/jpeg": ".jpeg",
    "image/jpg": ".jpeg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
    "image/tiff": ".tiff",
}
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class WriteResult:
    markdown_path: Path
    json_path: Path | None
    asset_dir: Path | None
    images_saved: int


def data_uri_payload(uri: str | None) -> tuple[str, bytes] | None:
    """Decode ``data:image/png;base64,...`` into ``(mime, bytes)``."""
    if not uri:
        return None
    match = _DATA_URI.match(uri.strip())
    if match is None:
        return None
    encoded = "".join(match.group("data").split())
    try:
        payload = base64.b64decode(encoded, validate=True)
    except ValueError, binascii.Error:
        return None
    if not payload:
        return None
    return match.group("mime").lower(), payload


def safe_component(text: str, fallback: str = "file") -> str:
    cleaned = _UNSAFE.sub("_", text).strip("._")
    return cleaned or fallback


def asset_filename(page: Page, image: ImageRef, extension: str) -> str:
    stem = Path(image.id).stem or "img"
    return f"p{page.number:04d}-{safe_component(stem, 'img')}{extension}"


def save_page_images(page: Page, directory: Path) -> dict[str, str]:
    """Write a page's embedded images to ``directory``; return id -> filename."""
    saved: dict[str, str] = {}
    for image in page.images:
        decoded = data_uri_payload(image.image_base64)
        if decoded is None:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        mime, payload = decoded
        name = asset_filename(page, image, _MIME_EXT.get(mime, ".bin"))
        (directory / name).write_bytes(payload)
        saved[image.id] = name
    return saved


def rewrite_image_links(
    markdown: str, links: Mapping[str, str], asset_prefix: str = ""
) -> str:
    for image_id, name in links.items():
        target = f"{asset_prefix}/{name}" if asset_prefix else name
        markdown = markdown.replace(f"]({image_id})", f"]({target})")
    return markdown


def inline_tables(markdown: str, tables: Iterable[TableRef]) -> str:
    """Replace ``[tbl-0.html](tbl-0.html)`` placeholders with the table body."""
    for table in tables:
        markdown = markdown.replace(f"[{table.id}]({table.id})", table.content)
    return markdown


def _blockquote(text: str) -> str:
    lines = text.strip().splitlines() or [""]
    return "\n".join(f"> {line}" if line.strip() else ">" for line in lines)


def render_page(
    page: Page,
    links: Mapping[str, str],
    *,
    asset_prefix: str = "",
    include_header: bool = False,
    include_footer: bool = False,
) -> str:
    body = inline_tables(page.markdown, page.tables)
    body = rewrite_image_links(body, links, asset_prefix)
    parts: list[str] = []
    if include_header and page.header and page.header.strip():
        parts.append(_blockquote(f"*Header:* {page.header}"))
    if body.strip():
        parts.append(body.strip())
    else:
        parts.append("<!-- no text extracted on this page -->")
    if include_footer and page.footer and page.footer.strip():
        parts.append(_blockquote(f"*Footer:* {page.footer}"))
    return "\n\n".join(parts)


def render_markdown(
    document: OcrDocument,
    *,
    links_by_page: Mapping[int, Mapping[str, str]],
    asset_prefix: str = "",
    include_header: bool = False,
    include_footer: bool = False,
) -> str:
    """Assemble the full document Markdown, one ``<!-- page: N -->`` block per page."""
    from .pages import format_page_numbers

    frontmatter = [
        "---",
        f"source: {json.dumps(str(document.source))}",
        f"model: {json.dumps(document.model or 'unknown')}",
        f"pages: {json.dumps(format_page_numbers([p.index for p in document.pages]))}",
        f"page_count: {document.page_count}",
        f"generated_at: {json.dumps(document.generated_at.isoformat(timespec='seconds'))}",
    ]
    frontmatter.append("---")

    blocks = ["\n".join(frontmatter)]
    for page in document.pages:
        body = render_page(
            page,
            links_by_page.get(page.index, {}),
            asset_prefix=asset_prefix,
            include_header=include_header,
            include_footer=include_footer,
        )
        blocks.append(f"<!-- page: {page.number} -->\n\n{body}")
    return "\n\n".join(blocks) + "\n"


def _maybe_json(text: str | None) -> Any:
    if text is None:
        return None
    try:
        return json.loads(text)
    except TypeError, ValueError:
        return text


def build_json_artifact(
    document: OcrDocument,
    *,
    links_by_page: Mapping[int, Mapping[str, str]],
) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []
    for page in document.pages:
        images = []
        for image in page.images:
            entry: dict[str, Any] = {
                "id": image.id,
                "top_left_x": image.top_left_x,
                "top_left_y": image.top_left_y,
                "bottom_right_x": image.bottom_right_x,
                "bottom_right_y": image.bottom_right_y,
            }
            if image.id in links_by_page.get(page.index, {}):
                entry["path"] = links_by_page[page.index][image.id]
            elif image.image_base64 and not image.image_base64.startswith("data:"):
                entry["url"] = image.image_base64
            images.append(entry)
        pages.append(
            {
                "page": page.number,
                "markdown": page.markdown,
                "images": images,
                "tables": [
                    {"id": t.id, "format": t.format, "content": t.content}
                    for t in page.tables
                ],
                "header": page.header,
                "footer": page.footer,
                "dimensions": page.dimensions,
                "blocks": list(page.blocks),
            }
        )
    return {
        "source": str(document.source),
        "model": document.model,
        "generated_at": document.generated_at.isoformat(timespec="seconds"),
        "page_count": document.page_count,
        "usage": dict(document.usage) if document.usage else None,
        "pages": pages,
    }


def default_output_path(pdf: Path, out_dir: Path | None = None) -> Path:
    stem = re.sub(r"\s*\(\d+\)\s*$", "", pdf.stem)
    stem = safe_component(stem, "document")
    return (out_dir or Path("out")) / f"{stem}.md"


def assets_dir_for(markdown_path: Path) -> Path:
    return markdown_path.with_name(f"{markdown_path.stem}.assets")


def write_document(
    document: OcrDocument,
    markdown_path: Path,
    *,
    save_json: bool = True,
    save_assets: bool = True,
    include_header: bool = False,
    include_footer: bool = False,
) -> WriteResult:
    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = assets_dir_for(markdown_path) if save_assets else None

    links_by_page: dict[int, dict[str, str]] = {}
    images_saved = 0
    for page in document.pages:
        if asset_dir is None:
            continue
        links = save_page_images(page, asset_dir)
        if links:
            links_by_page[page.index] = links
            images_saved += len(links)

    if asset_dir is not None and not asset_dir.exists():
        asset_dir = None  # nothing embedded in the Markdown, keep the tree clean
    asset_prefix = asset_dir.name if asset_dir is not None else ""
    text = render_markdown(
        document,
        links_by_page=links_by_page,
        asset_prefix=asset_prefix,
        include_header=include_header,
        include_footer=include_footer,
    )
    _atomic_write(markdown_path, text)

    json_path: Path | None = None
    if save_json:
        json_path = markdown_path.with_suffix(".ocr.json")
        artifact = build_json_artifact(document, links_by_page=links_by_page)
        _atomic_write(
            json_path,
            json.dumps(artifact, ensure_ascii=False, indent=1) + "\n",
        )

    return WriteResult(
        markdown_path=markdown_path,
        json_path=json_path,
        asset_dir=asset_dir,
        images_saved=images_saved,
    )


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
