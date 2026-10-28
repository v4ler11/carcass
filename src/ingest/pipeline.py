"""End-to-end ingestion for the PLU corpus.

Reads the source PDFs in ``files/`` and rebuilds the ``ingest/`` tree:

    ingest/oap/s0_ingest/             OCR of the OAP PDF
    ingest/oap/s1_caption/            captions for the OAP images
    ingest/plu_reglement/s0_ingest/   OCR of the written règlement
    ingest/plu_reglement/s1_caption/  captions for its images
    ingest/plu_graph/document/        annex panels + OCR of the reserve tables
    ingest/plu_graph/parcel/          parcel lookup, zooms, annotated sheet, WKT

Run ``python -m ingest.pipeline`` or the ``ingest_pipeline`` console script.
Use ``--out`` to write a comparison copy somewhere else.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

import core.logger  # noqa: F401 - importing configures loguru
from chat.clients import Client
from core import BASE_DIR
from core.config import (
    DEFAULT_CAPTION_MODEL,
    DEFAULT_PARCEL_ADDRESS,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_EXTRA_BODY,
    OPENROUTER_URL,
    openrouter_api_key,
)
from ingest.oap.s1_caption import caption_images
from ingest.ocr_pipeline.markdown import default_output_path, write_document
from ingest.ocr_pipeline.mistral_ocr import OcrSettings, ocr_pdf, resolve_api_key
from ingest.plu_graph import pipeline as plu_graph

DEFAULT_FILES = BASE_DIR / "files"
DEFAULT_OUT = BASE_DIR / "ingest"

OAP_PATTERN = "oap_*.pdf"
REGLEMENT_PATTERN = "plu_reglement_*.pdf"
SHEET_PATTERN = "plu_reglement_graphique*.pdf"


def find_pdf(directory: Path, pattern: str, *, exclude: str | None = None) -> Path:
    hits = sorted(
        path
        for path in directory.glob(pattern)
        if exclude is None or exclude not in path.name
    )
    if not hits:
        raise SystemExit(f"missing {pattern!r} in {directory}")
    if len(hits) > 1:
        names = ", ".join(path.name for path in hits)
        raise SystemExit(f"ambiguous {pattern!r} in {directory}: {names}")
    return hits[0]


def run_text_ocr(pdf: Path, out_dir: Path, *, api_key: str) -> Path:
    """OCR one PDF into ``out_dir`` and return its Markdown path."""
    logger.info("ocr {}", pdf.name)
    document = ocr_pdf(
        pdf,
        api_key=api_key,
        settings=OcrSettings(),
        progress=lambda message: logger.info("{}", message),
    )
    markdown_path = default_output_path(pdf, out_dir)
    result = write_document(document, markdown_path)
    logger.info(
        "wrote {} ({} pages, {} images)",
        result.markdown_path,
        document.page_count,
        result.images_saved,
    )
    return result.markdown_path


async def run_captions(
    ocr_dir: Path,
    captions_path: Path,
    *,
    model: str = DEFAULT_CAPTION_MODEL,
) -> None:
    api_key = openrouter_api_key()
    if not api_key:
        raise SystemExit(
            f"{OPENROUTER_API_KEY_ENV} is not set; captioning needs the "
            "OpenRouter key used by the rest of the agent pipeline"
        )
    async with Client(
        base_url=OPENROUTER_URL,
        api_key=api_key,
        extra_body=OPENROUTER_EXTRA_BODY,
    ) as client:
        await caption_images(
            client=client,
            ocr_dir=ocr_dir,
            captions_path=captions_path,
            model=model,
        )


def run_ingest(
    *,
    files: Path,
    out: Path,
    address: str,
    api_key: str | None = None,
    caption_model: str = DEFAULT_CAPTION_MODEL,
    skip_ocr: bool = False,
    skip_captions: bool = False,
) -> None:
    out = Path(out)
    files = Path(files)
    if skip_ocr:
        skip_captions = True
    if not skip_ocr and not api_key:
        api_key = resolve_api_key()
    if not skip_captions and not openrouter_api_key():
        raise SystemExit(
            f"{OPENROUTER_API_KEY_ENV} is not set; pass --skip-captions "
            "to stop after OCR"
        )

    for name, pattern, exclude in (
        ("oap", OAP_PATTERN, None),
        ("plu_reglement", REGLEMENT_PATTERN, "graphique"),
    ):
        pdf = find_pdf(files, pattern, exclude=exclude)
        stage = out / name
        ocr_dir = stage / "s0_ingest"
        if not skip_ocr:
            markdown_path = run_text_ocr(pdf, ocr_dir, api_key=api_key or "")
            if not skip_captions:
                asyncio.run(
                    run_captions(
                        markdown_path.parent,
                        stage / "s1_caption" / "captions.json",
                        model=caption_model,
                    )
                )

    sheet = find_pdf(files, SHEET_PATTERN)
    graph_dir = out / "plu_graph"
    plu_graph.run_document(
        sheet,
        graph_dir / "document",
        api_key=api_key if not skip_ocr else None,
        skip_ocr=skip_ocr,
    )
    plu_graph.run_parcel(address, sheet, graph_dir / "parcel")
    logger.info("ingest complete -> {}", out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest_pipeline",
        description="rebuild the ingest tree from the source PDFs",
    )
    parser.add_argument(
        "--files",
        default=str(DEFAULT_FILES),
        help=f"source PDF directory (default: {DEFAULT_FILES})",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=str(DEFAULT_OUT),
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--address",
        default=DEFAULT_PARCEL_ADDRESS,
        help=f"parcel address for plu_graph (default: {DEFAULT_PARCEL_ADDRESS})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Mistral API key (default: MISTRAL_API_KEY or .env)",
    )
    parser.add_argument(
        "--caption-model",
        default=DEFAULT_CAPTION_MODEL,
        help=f"vision model for captions (default: {DEFAULT_CAPTION_MODEL})",
    )
    parser.add_argument(
        "--skip-captions",
        action="store_true",
        help="run OCR but skip the image-caption model calls",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="skip all Mistral OCR and caption calls (panels and parcel only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_ingest(
        files=Path(args.files),
        out=Path(args.out),
        address=args.address,
        api_key=args.api_key,
        caption_model=args.caption_model,
        skip_ocr=args.skip_ocr,
        skip_captions=args.skip_captions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
