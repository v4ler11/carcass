#!/usr/bin/env python3
"""End-to-end: an address -> its parcel located on the PLU graphic sheet.

    uv run python -m ingest.plu_graph.pipeline all --address "4 rue Joliot Curie 91190 Gif-sur-Yvette"

or run the whole corpus build, which includes this stage::

    uv run ingest_pipeline

Stages, each a module of this package:

1. ``geo``     - BAN geocode, cadastral parcel, GPU zoning and prescriptions;
2. ``overlay`` - solve the sheet georeference and place the parcel;
3. ``pipeline`` - render the whole sheet plus three zoom levels;
4. ``panels``  - cut the annex panels (legend, key, tables);
5. ``tables``  - transcribe the three reserve tables.

Output tree (``ingest/plu_graph`` by default)::

    metadata.json          registry answer for the address and the parcel
    zoom-0.png             whole sheet, parcel outlined
    zoom-1.png             wide: several zones around the parcel
    zoom-2.png             the neighbouring zones
    zoom-3.png             the parcel's own zone only
    parcel.pdf             the annotated sheet (vector outline, A0)
    parsed/
      legend.png           legend, as an image
      zones_legend.png     zone key, as an image
      tables/*.md          the three reserve tables, transcribed
      s1_raw/              panels before OCR (+ panels.json)
      s2_ocr/              panels after OCR (.md for tables)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

from . import geo as geodata
from . import overlay
from . import tables as table_ocr
from .panels import extract as extract_panels

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_OUT = "ingest/plu_graph"
PDF_GLOB = "files/plu_reglement_graphique*.pdf"
DPI = 150
MAP_WIDTH = 2400
ZOOM_WIDTH = 1400
ZOOM_MAX_DPI = 600
L93_TO_WGS84 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

#: metres of sheet covered by zoom-1 (wide) and zoom-3 (tight, on the parcel)
ZOOM1_M = 2500.0
ZOOM3_M = 380.0
#: fallback for zoom-2 when the zone polygon is unavailable
ZOOM2_MIN_M = 600.0


def find_sheet(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"missing PDF: {path}")
        return path
    hits = sorted(REPO_ROOT.glob(PDF_GLOB))
    if not hits:
        raise SystemExit(f"no graphic-regulation sheet found ({PDF_GLOB})")
    return hits[0]


def zone_of(features: list[dict]) -> tuple[dict | None, list[dict]]:
    """Split the GPU feature-info answer into its zone and its prescriptions.

    The feature type is carried by the id (``zone_urba.4356473``), not by the
    properties, and the zone's own geometry comes back as null.
    """
    zone, others = None, []
    for feature in features:
        props = feature.get("properties", {})
        marker = feature.get("id", "").split(".")[0] or props.get("typename", "")
        if marker == "zone_urba" or "typezone" in props:
            zone = feature
        else:
            others.append(feature)
    return zone, others


def with_retry(call, attempts: int = 3, delay_s: float = 1.5):
    """The GPU feature-info endpoint returns sporadic 500s."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - retried, then reported
            last = error
            time.sleep(delay_s * (attempt + 1))
    if last is None:
        raise RuntimeError("retry called without an attempt")
    raise last


def zone_rings_px(
    code: str, centre_lonlat: tuple[float, float], geo: overlay.Georef
) -> list[np.ndarray]:
    """The zoning polygon in sheet pixels.

    ``feature-info/du`` answers with ``geometry: null`` for the zone, so the
    polygon comes from the GPU geometry route, queried at the parcel centroid.
    """
    lon, lat = centre_lonlat
    data = geodata.get(
        f"{geodata.APICARTO}/gpu/zone-urba",
        {"geom": json.dumps({"type": "Point", "coordinates": [lon, lat]})},
    )
    rings: list[np.ndarray] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        if code and props.get("libelle") != code:
            continue
        geometry = feature.get("geometry") or {}
        polygons = geometry.get("coordinates") or []
        if geometry.get("type") == "Polygon":
            polygons = [polygons]
        for polygon in polygons:
            for ring in polygon[:1]:  # outer ring of each part
                l93 = np.array([overlay.WGS84_TO_L93.transform(x, y) for x, y in ring])
                rings.append(geo.to_px(l93))
    return rings


def window_around(cx: float, cy: float, half_px: float, size: tuple[int, int]):
    width, height = size
    x0 = max(0, round(cx - half_px))
    y0 = max(0, round(cy - half_px))
    x1 = min(width, round(cx + half_px))
    y1 = min(height, round(cy + half_px))
    return x0, y0, x1, y1


def bbox_window(points: list[np.ndarray], pad: float, size: tuple[int, int]):
    stacked = np.concatenate(points)
    x0, y0 = stacked[:, 0].min(), stacked[:, 1].min()
    x1, y1 = stacked[:, 0].max(), stacked[:, 1].max()
    half = max(x1 - x0, y1 - y0) * (0.5 + pad)
    return window_around((x0 + x1) / 2, (y0 + y1) / 2, half, size)


def font_for(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def annotate(
    image: Image.Image,
    geo: overlay.Georef,
    window,
    label: str,
    *,
    emphasis: float = 1.0,
) -> Image.Image:
    """Draw the parcel and its label in *image* pixels, so the line and the text
    keep the same weight whether the crop is 150 m or 900 m wide."""
    x0, y0, x1, y1 = window
    draw = ImageDraw.Draw(image)
    sx = image.width / max(1, (x1 - x0))
    sy = image.height / max(1, (y1 - y0))
    width = max(2, round(image.width / 400 * emphasis))

    boxes = []
    for ring in geo.parcel["rings_l93"]:
        pts = geo.to_px(ring)
        points = [((px - x0) * sx, (py - y0) * sy) for px, py in pts]
        draw.line(points + [points[0]], fill=(221, 13, 13), width=width, joint="curve")
        boxes.append(np.array(points))

    stacked = np.concatenate(boxes)
    text = label
    # text sized to the parcel as drawn, not to the image: 56 px on a 900 m
    # crop and 12 px on the whole sheet, so it neither swamps nor vanishes
    bbox_height = float(stacked[:, 1].max() - stacked[:, 1].min())
    size = max(12, min(56, round(0.35 * bbox_height)))
    font = font_for(size)
    left = float(stacked[:, 0].min())
    top = float(stacked[:, 1].min()) - size * 1.4
    if top < size:
        top = float(stacked[:, 1].max()) + size * 0.3
    draw.text(
        (left, top),
        text,
        font=font,
        fill=(221, 13, 13),
        stroke_width=max(2, size // 8),
        stroke_fill=(255, 255, 255),
    )
    return image


def render_window(
    page,
    rect,
    window,
    pt_per_px: float,
    out: Path,
    geo,
    label: str,
    *,
    target_width: int = ZOOM_WIDTH,
    max_dpi: int = ZOOM_MAX_DPI,
    emphasis: float = 1.0,
) -> tuple[int, int]:
    """Render a raster-pixel window as a PNG and annotate it."""
    x0, y0, x1, y1 = window
    clip = pymupdf.Rect(
        rect.x0 + x0 * pt_per_px,
        rect.y0 + y0 * pt_per_px,
        rect.x0 + x1 * pt_per_px,
        rect.y0 + y1 * pt_per_px,
    )
    dpi = min(max_dpi, max(36, int(target_width / (clip.width / 72))))
    pixmap = page.get_pixmap(clip=clip, dpi=dpi)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    annotate(image, geo, window, label, emphasis=emphasis).save(out)
    return image.size


def run_document(
    pdf: Path, out: Path, *, api_key: str | None = None, skip_ocr: bool = False
) -> int:
    """Stage 1: cut the annex panels of the sheet and OCR the reserve tables."""
    parsed = out / "parsed"
    raw_dir = parsed / "s1_raw"
    ocr_dir = parsed / "s2_ocr"
    for path in (out, parsed, raw_dir, ocr_dir):
        path.mkdir(parents=True, exist_ok=True)

    page = pymupdf.open(pdf)[0]
    page_mm = [
        round(page.rect.width * 25.4 / 72, 1),
        round(page.rect.height * 25.4 / 72, 1),
    ]
    raster_px = [round(page.rect.width / 72 * DPI), round(page.rect.height / 72 * DPI)]

    panels = extract_panels(pdf, dpi=DPI, out_dir=raw_dir)
    (raw_dir / "panels.json").write_text(
        json.dumps(
            {
                "source": str(pdf),
                "page": 1,
                "dpi": DPI,
                "panels": [panel.as_dict(DPI) for panel in panels],
            },
            indent=2,
        )
        + "\n"
    )
    figures = [panel for panel in panels if panel.kind == "figure"]
    tables = [panel for panel in panels if panel.kind == "table"]

    settings = table_ocr.OcrSettings(table_format="html", include_images=False)
    transcribed: list[dict] = []
    if tables and not skip_ocr:
        from mistralai.client import Mistral

        client = Mistral(api_key=table_ocr.resolve_api_key(api_key))
        for panel in tables:
            source = raw_dir / f"{panel.name}.png"
            result = table_ocr.run_one(
                source, settings, table_ocr.DEFAULT_MAX_PIXELS, client, 1.0
            )
            target = ocr_dir / f"{panel.name}.md"
            target.write_text(
                table_ocr.front_matter(result, settings) + result.markdown + "\n"
            )
            transcribed.append(
                {
                    "name": panel.name,
                    "chars": len(result.markdown),
                    "tables": result.tables,
                    "joins": result.joins,
                    "elapsed_s": round(result.elapsed_s, 1),
                }
            )
            print(f"ocr            {target.name}  {len(result.markdown)} chars")

    for panel in figures:  # the legend is final as an image
        shutil.copy2(raw_dir / f"{panel.name}.png", ocr_dir / f"{panel.name}.png")

    shutil.copy2(pdf, out / "reference.pdf")

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "pdf": pdf.name,
            "reference_copy": "reference.pdf",
            "page": 1,
            "page_mm": page_mm,
            "raster_px": raster_px,
            "dpi": DPI,
        },
        "panels": {
            "directory": "parsed/s1_raw",
            "count": len(panels),
            "items": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "box_px": p.box.as_dict(),
                    "check": p.check,
                }
                for p in panels
            ],
        },
        "tables_ocr": {"directory": "parsed/s2_ocr", "items": transcribed},
        "figures_ocr": {
            "directory": "parsed/s2_ocr",
            "items": [p.name for p in figures],
        },
    }
    (out / "metadata_parsing.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"\ndocument -> {out}/")
    print("  reference.pdf  metadata_parsing.json")
    print(
        f"  parsed/s1_raw/ ({len(panels)} panels)   parsed/s2_ocr/ "
        f"({len(transcribed)} md + {len(figures)} images)"
    )
    return 0


def write_parcel_wkt(geo: overlay.Georef, path: Path) -> None:
    """Write the parcel boundary in Lambert-93 as WKT for downstream tools."""

    def ring_text(ring) -> str:
        return ", ".join(f"{float(x):.3f} {float(y):.3f}" for x, y in ring)

    rings = geo.parcel["rings_l93"]
    if len(rings) == 1:
        text = f"POLYGON (({ring_text(rings[0])}))"
    else:
        parts = ", ".join(f"(({ring_text(ring)}))" for ring in rings)
        text = f"MULTIPOLYGON ({parts})"
    path.write_text(text + "\n")


def run_parcel(
    address_text: str, pdf: Path, out: Path, *, api_key: str | None = None
) -> int:
    """Stage 2: look the parcel up, place it on the sheet, render the zooms."""
    out.mkdir(parents=True, exist_ok=True)

    address = geodata.geocode(address_text)
    idu = geodata.parcel_idu_for_address(address)
    try:
        pid, features, sup = with_retry(lambda: geodata.parcel_dossier(idu))
    except Exception as error:  # noqa: BLE001 - zoning is optional
        print(f"warning        GPU feature-info unavailable ({error}); zoning omitted")
        pid, features, sup = None, [], []
    zone, others = zone_of(features)
    zone_props = (zone or {}).get("properties", {})

    geo = overlay.georeference(pdf, idu, progress=print)
    print(f"address        {address['label']}  ({address['address_id']})")
    print(
        f"zone           {zone_props.get('libelle', 'unavailable')} "
        f"({zone_props.get('libellelong') or zone_props.get('libelong') or 'type ' + str(zone_props.get('typezone'))})"
    )

    overlay.draw_parcel(geo)
    parcel_pdf = out / "parcel.pdf"
    geo.doc.save(parcel_pdf, garbage=3, deflate=True)
    write_parcel_wkt(geo, out / "parcel.wkt")

    page = pymupdf.open(pdf)[0]  # clean page: the overlay is drawn per image
    width, height = geo.size
    pt_per_px = geo.rect.width / width

    anchor = geo.to_px(geo.parcel["rings_l93"][0])
    cx, cy = float(anchor[:, 0].mean()), float(anchor[:, 1].mean())
    l93 = np.concatenate(geo.parcel["rings_l93"])
    label = f"{geo.parcel['idu'][8:10]} {geo.parcel['idu'][10:]}  {geo.parcel['area_m2']:,} m2"

    sheet_window = (0, 0, width, height)
    sizes = {
        "zoom-0.png": render_window(
            page,
            geo.rect,
            sheet_window,
            pt_per_px,
            out / "zoom-0.png",
            geo,
            label,
            target_width=MAP_WIDTH,
        )
    }

    px_per_m = geo.scale
    windows = {
        "zoom-1.png": window_around(cx, cy, ZOOM1_M / 2 * px_per_m, (width, height))
    }
    centroid_lonlat = L93_TO_WGS84.transform(
        float(l93[:, 0].mean()), float(l93[:, 1].mean())
    )
    try:
        zone_rings = (
            zone_rings_px(zone_props.get("libelle", ""), centroid_lonlat, geo)
            if zone
            else []
        )
    except Exception as error:  # noqa: BLE001 - fall back to a plain zoom
        print(
            f"warning        zone geometry unavailable ({error}); zoom-2 falls back to {ZOOM2_MIN_M:.0f} m"
        )
        zone_rings = []
    if zone_rings:
        windows["zoom-2.png"] = bbox_window(zone_rings, 0.08, (width, height))
    else:
        windows["zoom-2.png"] = window_around(
            cx, cy, ZOOM2_MIN_M / 2 * px_per_m, (width, height)
        )
    windows["zoom-3.png"] = window_around(
        cx, cy, ZOOM3_M / 2 * px_per_m, (width, height)
    )

    for name, window in windows.items():
        # the tight crop gets a heavier outline: the parcel is the subject there
        emphasis = 2.0 if name == "zoom-3.png" else 1.0
        sizes[name] = render_window(
            page, geo.rect, window, pt_per_px, out / name, geo, label, emphasis=emphasis
        )
    for name in ("zoom-0.png", *windows):
        print(f"written        {out / name}  {sizes[name][0]}x{sizes[name][1]}")

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "pdf": pdf.name,
            "path": str(pdf),
            "reference_copy": "../document/reference.pdf"
            if (out.parent / "document" / "reference.pdf").is_file()
            else None,
            "page": 1,
            "page_mm": [
                round(geo.rect.width * 25.4 / 72, 1),
                round(geo.rect.height * 25.4 / 72, 1),
            ],
            "raster_px": [width, height],
        },
        "address": address,
        "parcel": {
            "idu": geo.parcel["idu"],
            "section": geo.parcel["section"],
            "numero": geo.parcel["numero"],
            "area_m2": geo.parcel["area_m2"],
            "commune": geo.parcel["commune"],
            "gpu_id": pid,
            "bbox_l93": [round(float(v), 1) for v in l93.min(axis=0)]
            + [round(float(v), 1) for v in l93.max(axis=0)],
            "centroid_px": [round(cx, 1), round(cy, 1)],
        },
        "zone": {
            "code": zone_props.get("libelle"),
            "type": zone_props.get("typezone"),
            "nom": zone_props.get("nomfic") or zone_props.get("nomfich"),
            "url": zone_props.get("urlfic"),
        },
        "prescriptions": [
            {
                "typename": f["properties"].get("typename"),
                "type": f["properties"].get("type"),
                "label": f["properties"].get("libelle"),
            }
            for f in others
        ],
        "servitudes": [
            {
                "typename": f["properties"].get("typename"),
                "label": f["properties"].get("libelle"),
            }
            for f in sup
        ],
        "georeference": {
            "scale_px_per_m": round(geo.scale, 5),
            "nominal_px_per_m": round(geo.nominal, 5),
            "rotation_deg": round(float(np.degrees(geo.angle)), 2),
            "commune_iou": round(geo.iou, 4),
            "snap_px": round(geo.snap_px, 2),
            "parcel_edge_residual_median_m": round(
                geo.residual_median_px / geo.scale, 2
            ),
            "parcel_edge_residual_mean_m": round(geo.residual_mean_px / geo.scale, 2),
            "note": "sheet raster is 150 dpi (0.84 px/m); zooms are rendered above native "
            "resolution, so the parcel outline is sharp and the map beneath is soft",
        },
        "images": {
            "sheet": {
                "file": "zoom-0.png",
                "window_px": list(sheet_window),
                "size_px": sizes["zoom-0.png"],
            },
            "zooms": {
                name: {
                    "file": name,
                    "window_px": list(w),
                    "size_px": sizes[name],
                    "metres": round((w[2] - w[0]) / geo.scale),
                }
                for name, w in windows.items()
            },
            "annotated_pdf": parcel_pdf.name,
        },
    }
    (out / "metadata_parcel.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )

    print(f"\nparcel -> {out}/")
    print("  metadata_parcel.json  parcel.wkt  zoom-0..3.png  parcel.pdf")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plu-graph",
        description="PLU graphic-regulation sheet: two stages over one sheet.",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--pdf", default=None, help=f"sheet (default: {PDF_GLOB})")
        sub.add_argument(
            "-o",
            "--out",
            default=None,
            help="output directory (default: ingest/plu_graph/<stage>)",
        )
        sub.add_argument("--api-key", default=None)

    document = stages.add_parser(
        "document", help="extract the annex panels and OCR the reserve tables"
    )
    common(document)
    document.add_argument(
        "--skip-ocr", action="store_true", help="extract panels, do not transcribe"
    )

    parcel = stages.add_parser(
        "parcel", help="look up the parcel, place it, render the zooms"
    )
    common(parcel)
    parcel.add_argument("--address", required=True)

    both = stages.add_parser("all", help="document, then parcel")
    common(both)
    both.add_argument("--address", required=True)
    both.add_argument("--skip-ocr", action="store_true")

    args = parser.parse_args(argv)
    pdf = find_sheet(args.pdf)

    if args.stage == "document":
        out = Path(args.out) if args.out else Path(DEFAULT_OUT) / "document"
        return run_document(pdf, out, api_key=args.api_key, skip_ocr=args.skip_ocr)

    if args.stage == "parcel":
        out = Path(args.out) if args.out else Path(DEFAULT_OUT) / "parcel"
        return run_parcel(args.address, pdf, out, api_key=args.api_key)

    base = Path(args.out) if args.out else Path(DEFAULT_OUT)
    status = run_document(
        pdf, base / "document", api_key=args.api_key, skip_ocr=args.skip_ocr
    )
    return (
        run_parcel(args.address, pdf, base / "parcel", api_key=args.api_key) or status
    )


if __name__ == "__main__":
    raise SystemExit(main())
