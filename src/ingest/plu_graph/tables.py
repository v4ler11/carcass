"""OCR the reserve tables of the PLU sheet into Markdown, one document each.

Only tables are transcribed.  The legend, the zone key and the ``elements
recenses`` note stay as images: their meaning lives in the swatches and the
locate inset, which OCR cannot carry, so ``01_legende.png`` etc. remain the
authoritative artifacts.

    uv run python -m ingest.plu_graph.tables                 # ingest/plu_graph/document/parsed/s1_raw tables -> *.md
    uv run python -m ingest.plu_graph.tables --dry-run
    uv run python -m ingest.plu_graph.tables --kind all      # every panel, for inspection

Table cells are requested as HTML: the reserve tables merge cells (a beneficiary
spanning two rows, two superficies in one cell) which a pipe table cannot hold.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from core.config import DEFAULT_OCR_MODEL, DEFAULT_TABLE_MAX_PIXELS
from ingest.ocr_pipeline.markdown import inline_tables
from ingest.ocr_pipeline.mistral_ocr import (
    OcrSettings,
    ocr_image,
    resolve_api_key,
)

DEFAULT_DIR = "ingest/plu_graph/document/parsed/s1_raw"
MANIFEST = "panels.json"
#: Panels are readable well above the model's ~1 MP page budget; a larger image
#: is downscaled before it is sent and the fact is recorded in the front matter.
DEFAULT_MAX_PIXELS = DEFAULT_TABLE_MAX_PIXELS
#: Proof renders that live next to the panels but are not panels.
SKIP_STEMS = ("overlay",)


@dataclass
class Result:
    source: Path
    markdown: str
    tables: int
    image_px: tuple[int, int]
    sent_px: tuple[int, int]
    elapsed_s: float
    joins: int = 0


def select_panels(
    paths: list[Path], kinds: tuple[str, ...], manifest: Path | None
) -> list[Path]:
    """Keep the panels whose kind is wanted, using the extractor's manifest.

    Without a manifest the selection cannot be trusted, so every image is kept
    and the caller sees what it got.
    """
    if not manifest or not manifest.is_file():
        return paths
    data = json.loads(manifest.read_text())
    wanted = {
        panel["name"] for panel in data.get("panels", []) if panel.get("kind") in kinds
    }
    return [path for path in paths if path.stem in wanted]


def payload_of(
    image: Image.Image, max_pixels: int, scale: float = 1.0
) -> tuple[bytes, tuple[int, int]]:
    """Encode as PNG at native size, downscaling only if it is over budget."""
    if scale != 1.0:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    width, height = image.size
    if width * height > max_pixels:
        factor = (max_pixels / float(width * height)) ** 0.5
        image = image.resize(
            (max(1, int(width * factor)), max(1, int(height * factor))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), image.size


def count_cell_joins(body: str) -> int:
    """Occurrences of a section code fused to the preceding number.

    The model occasionally drops the line break inside a cell and turns
    ``48, 50`` + ``BD206`` into ``50BD206``.  Counted so the file says so; the
    text is never rewritten.
    """
    return len(re.findall(r"(?<=\d)[A-Z]{2}(?=\d)", body))


def front_matter(result: Result, settings: OcrSettings) -> str:
    sent = f"{result.sent_px[0]}x{result.sent_px[1]}"
    native = f"{result.image_px[0]}x{result.image_px[1]}"
    joins = f"possible_cell_joins: {result.joins}\n" if result.joins else ""
    return (
        "---\n"
        f'source: "{result.source.as_posix()}"\n'
        f'model: "{settings.model}"\n'
        f"image: {native} px\n"
        f"sent: {sent} px" + ("  (downscaled)" if sent != native else "") + "\n"
        f"table_format: {settings.table_format or 'inline'}\n"
        f"tables: {result.tables}\n"
        + joins
        + f'generated_at: "{datetime.now().isoformat(timespec="seconds")}"\n'
        "---\n\n"
    )


def run_one(
    path: Path, settings: OcrSettings, max_pixels: int, client, scale: float
) -> Result:
    image = Image.open(path).convert("RGB")
    payload, sent = payload_of(image, max_pixels, scale)
    started = time.time()
    markdown, tables = ocr_image(client, payload, settings, mime="image/png")
    elapsed = time.time() - started
    body = inline_tables(markdown, tables).strip()
    return Result(
        source=path,
        markdown=body,
        tables=len(tables),
        image_px=image.size,
        sent_px=sent,
        elapsed_s=elapsed,
        joins=count_cell_joins(body),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OCR PLU reserve tables.")
    parser.add_argument(
        "images", nargs="*", help=f"PNG files (default: {DEFAULT_DIR}/*.png)"
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="output directory (default: next to the image)",
    )
    parser.add_argument("--model", default=DEFAULT_OCR_MODEL)
    parser.add_argument(
        "--kind",
        default="table",
        help="panel kinds to transcribe, comma separated (table, figure, all)",
    )
    parser.add_argument(
        "--table-format",
        choices=("html", "markdown", "none"),
        default="html",
        help="html keeps merged cells; none leaves tables inline",
    )
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--scale", type=float, default=1.0, help="upscale before sending"
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--manifest", default=None, help=f"panel manifest (default: <dir>/{MANIFEST})"
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = [Path(p) for p in args.images] or sorted(Path(DEFAULT_DIR).glob("*.png"))
    paths = [
        p
        for p in paths
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and not p.stem.startswith(SKIP_STEMS)
    ]
    if not paths:
        raise SystemExit("no images to OCR")

    kinds = tuple(k.strip() for k in args.kind.split(","))
    manifest = Path(args.manifest) if args.manifest else None
    if manifest is None and not args.images:
        manifest = Path(DEFAULT_DIR) / MANIFEST
    chosen = paths if "all" in kinds else select_panels(paths, kinds, manifest)
    if not chosen:
        raise SystemExit(f"no panel of kind {args.kind} found (manifest: {manifest})")

    settings = OcrSettings(
        model=args.model,
        table_format=None if args.table_format == "none" else args.table_format,
        include_images=False,
        timeout_s=600.0,
    )

    if args.dry_run:
        for path in chosen:
            width, height = Image.open(path).size
            factor = (
                min(1.0, (args.max_pixels / float(width * height)) ** 0.5) * args.scale
            )
            print(
                f"{path.name:40s} {width}x{height} px -> {int(width * factor)}x{int(height * factor)} px"
            )
        print(f"\n{len(chosen)} request(s), kind={args.kind}, model {settings.model}")
        return 0

    from mistralai.client import Mistral

    client = Mistral(api_key=resolve_api_key(args.api_key))
    results: list[Result] = []

    def job(path: Path) -> Result:
        return run_one(path, settings, args.max_pixels, client, args.scale)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        for result in pool.map(job, chosen):
            out_dir = Path(args.out) if args.out else result.source.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{result.source.stem}.md"
            target.write_text(front_matter(result, settings) + result.markdown + "\n")
            results.append(result)
            print(
                f"{target.name:44s} {len(result.markdown):6d} chars  "
                f"{result.tables} table(s)  {result.elapsed_s:5.1f}s"
                + (f"  {result.joins} cell join(s)" if result.joins else "")
            )

    sidecar = (Path(args.out) if args.out else chosen[0].parent) / "tables.ocr.json"
    sidecar.write_text(
        json.dumps(
            {
                "model": settings.model,
                "table_format": settings.table_format,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "tables": [
                    {
                        "source": r.source.as_posix(),
                        "markdown_file": f"{r.source.stem}.md",
                        "image_px": list(r.image_px),
                        "sent_px": list(r.sent_px),
                        "tables": r.tables,
                        "chars": len(r.markdown),
                        "possible_cell_joins": r.joins,
                        "elapsed_s": round(r.elapsed_s, 1),
                    }
                    for r in results
                ],
            },
            indent=2,
        )
        + "\n"
    )
    empty = [r.source.name for r in results if not r.markdown]
    print(f"\n{len(results)} table document(s) + {sidecar}")
    return 1 if empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
