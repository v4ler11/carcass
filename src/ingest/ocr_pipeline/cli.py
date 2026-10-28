"""Command line entry point: ``python -m ingest.ocr_pipeline.cli PDF [-o OUT.md]``.

The ``ingest_pipeline`` console script runs this stage as part of the full
corpus build; invoke it directly only to OCR one PDF.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from core.config import (
    DEFAULT_OCR_CHUNK_SIZE,
    DEFAULT_OCR_MODEL,
    DEFAULT_OCR_TIMEOUT_S,
    MISTRAL_API_KEY_ENV,
)

from .document import OcrDocument
from .markdown import (
    assets_dir_for,
    default_output_path,
    write_document,
)
from .mistral_ocr import (
    OcrError,
    OcrSettings,
    ocr_pdf,
    plan_requests,
    resolve_api_key,
)
from .pages import PageSpecError, format_page_numbers, parse_page_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mistral-ocr",
        description=(
            "Convert a PDF to Markdown with Mistral Document AI OCR "
            f"(default model: {DEFAULT_OCR_MODEL}), saving images and a JSON sidecar."
        ),
    )
    parser.add_argument("pdf", help="path to the source PDF")
    parser.add_argument(
        "-o",
        "--output",
        help="Markdown output path (default: out/<document>.md)",
    )
    parser.add_argument(
        "--pages",
        help="1-based page selection, e.g. '1-8,12' (default: all pages)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_OCR_CHUNK_SIZE,
        help=f"pages per API request (default: {DEFAULT_OCR_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_OCR_MODEL,
        help=f"OCR model id (default: {DEFAULT_OCR_MODEL})",
    )
    parser.add_argument(
        "--table-format",
        choices=("markdown", "html"),
        help="extract tables separately in this format, then inline them in the Markdown",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="do not request image payloads (image links stay unresolved)",
    )
    parser.add_argument(
        "--blocks",
        action="store_true",
        help="keep paragraph-level blocks and bounding boxes in the JSON sidecar",
    )
    parser.add_argument(
        "--extract-header",
        action="store_true",
        help="extract page headers into the Markdown as blockquotes",
    )
    parser.add_argument(
        "--extract-footer",
        action="store_true",
        help="extract page footers into the Markdown as blockquotes",
    )
    parser.add_argument(
        "--transport",
        choices=("auto", "inline", "upload"),
        default="auto",
        help="how to send the PDF: base64 inline, file upload, or auto (default)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_OCR_TIMEOUT_S,
        help=f"per-request timeout in seconds (default: {DEFAULT_OCR_TIMEOUT_S:g})",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="skip the .ocr.json sidecar",
    )
    parser.add_argument(
        "--api-key",
        help=f"Mistral API key (default: {MISTRAL_API_KEY_ENV} from the environment)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the request plan without calling the API",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress per-chunk progress on stderr",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pdf = Path(args.pdf)
    if not pdf.is_file():
        return _fail(f"PDF not found: {pdf}", 2)

    try:
        settings = OcrSettings(
            model=args.model,
            table_format=args.table_format,
            include_images=not args.no_images,
            include_blocks=args.blocks,
            extract_header=args.extract_header,
            extract_footer=args.extract_footer,
            chunk_size=args.chunk_size,
            transport=args.transport,
            timeout_s=args.timeout,
        )
        selected = parse_page_spec(args.pages)
        plan = plan_requests(pdf, pages=selected, settings=settings)
    except (OcrError, PageSpecError, ValueError) as exc:
        return _fail(str(exc), 2)

    output = Path(args.output) if args.output else default_output_path(pdf)
    if args.dry_run:
        _print_plan(pdf, plan, settings, output)
        return 0

    try:
        api_key = resolve_api_key(args.api_key)
    except OcrError as exc:
        return _fail(str(exc), 2)

    progress = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    try:
        document = ocr_pdf(
            pdf,
            api_key=api_key,
            pages=plan.selected,
            settings=settings,
            progress=progress,
        )
    except OcrError as exc:
        return _fail(str(exc), 1)
    except Exception as exc:  # noqa: BLE001 - SDK/network errors: report, do not traceback
        return _fail(f"OCR request failed: {type(exc).__name__}: {exc}", 1)

    result = write_document(
        document,
        output,
        save_json=not args.no_json,
        include_header=args.extract_header,
        include_footer=args.extract_footer,
    )
    _print_summary(document, result, settings)
    return 0


def _print_plan(pdf: Path, plan, settings: OcrSettings, output: Path) -> None:
    mib = plan.pdf_bytes / (1024 * 1024)
    print(f"source: {pdf} ({plan.total_pages} pages, {mib:.1f} MiB)")
    print(f"model: {settings.model}")
    print(f"pages: {format_page_numbers(plan.selected)}")
    chunks = ", ".join(chunk.human for chunk in plan.chunks)
    print(f"chunks: {len(plan.chunks)} ({chunks})")
    print(f"transport: {plan.transport}")
    print(f"requests: {plan.request_count}")
    print(f"output: {output}")
    print(f"assets: {assets_dir_for(output)}")


def _print_summary(document: OcrDocument, result, settings: OcrSettings) -> None:
    print(f"wrote {result.markdown_path}")
    print(
        f"pages: {document.page_count} | images: {document.image_count} "
        f"({result.images_saved} saved) | tables: {document.table_count}"
    )
    if result.asset_dir is not None:
        print(f"assets: {result.asset_dir}")
    if result.json_path is not None:
        print(f"raw: {result.json_path}")
    usage = document.usage or {}
    if usage.get("pages_processed"):
        print(f"billed pages: {usage['pages_processed']}")


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
