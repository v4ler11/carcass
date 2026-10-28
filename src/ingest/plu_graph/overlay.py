#!/usr/bin/env python3
"""Locate a cadastral parcel on the PLU graphic-regulation sheet and draw it.

The sheet carries exactly one 7021x4966 JPEG on an A0 page (1189x841 mm) and has
no text layer and no vector content, so the georeference has to be solved from
the image itself:

1. the commune footprint is segmented out of the raster (coloured zone fills on
   white paper);
2. the rotation of the map is found by matching the commune polygon from
   geo.api.gouv.fr against that footprint over 360 degrees;
3. the scale and position are then fitted by maximising the IoU between the two;
4. the fitted scale is checked against the scale printed on the sheet
   (1:7000 -> 0.84364 px/m at 7021 px over 3370.08 pt).

Measured on this sheet: rotation 0.0 deg, scale 0.84364 px/m (0.00 % from
nominal), commune IoU 0.99, and the projected parcel boundary sits a median of
1 px (1.2 m) from the cadastral lines drawn on the map.

    uv run python -m ingest.plu_graph.overlay
    uv run python -m ingest.plu_graph.overlay --parcel 91272000CR0247
    uv run python -m ingest.plu_graph.overlay --address "1 rue de la Gruerie 91190 Gif-sur-Yvette"

Outputs: out/plu_parcel_<idu>.pdf and out/plu_parcel_<idu>_zoom.png.
"""

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

from core.config import DEFAULT_PARCEL_ADDRESS

from . import geo
from .geo import WGS84_TO_L93

DEFAULT_PDF = "files/plu_reglement_graphique_91272_20210706 (1).pdf"

FIT_RES = 4  # fitting resolution: raster pixels per cell
PAD = 48  # canvas padding so the rotated template always fits


def commune_ring(insee):
    g = geo.get(f"https://geo.api.gouv.fr/communes/{insee}", {"fields": "nom,contour"})[
        "contour"
    ]
    poly = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
    return np.array([WGS84_TO_L93.transform(x, y) for x, y in poly])


def page_raster(page):
    """Native embedded image and the rect it is painted into."""
    xref = page.get_images()[0][0]
    info = page.parent.extract_image(xref)
    raster = np.asarray(Image.open(io.BytesIO(info["image"])).convert("RGB"))
    return raster, page.get_image_rects(xref)[0]


def commune_silhouette(rgb):
    """Coloured zone fills: everything inked, minus the detached title block."""
    ink = (rgb.min(axis=2) < 245).astype(np.uint8)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8)


def project(pts_l93, scale, angle, origin):
    """Lambert-93 metres -> raster pixels; north up, y axis downward."""
    th = np.deg2rad(angle)
    c, s = np.cos(th), np.sin(th)
    x = pts_l93[:, 0] - origin[0]
    y = pts_l93[:, 1] - origin[1]
    return np.stack(
        [(c * x - s * y) * scale + origin[2], -(s * x + c * y) * scale + origin[3]],
        axis=1,
    )


def polygon_centroid(pts):
    """Area centroid of a closed ring (shoelace), not the mean of its vertices."""
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = cross.sum() / 2
    if a == 0:
        return float(x.mean()), float(y.mean())
    return float(((x + xn) * cross).sum() / (6 * a)), float(
        ((y + yn) * cross).sum() / (6 * a)
    )


def mask_centroid(mask):
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def align(ring, target, target_centre, scale, angle, origin_m):
    """IoU of the projected ring against `target`, with position optimised.

    The ring is first moved so its area centroid sits on the target centroid;
    correlation of the two binary masks then equals their intersection, so the
    IoU falls straight out of the matchTemplate peak. Returns (iou, dx, dy) where
    (dx, dy) is the residual shift, in target pixels, of that centroid.
    """
    pts = project(
        ring,
        scale,
        angle,
        (origin_m[0], origin_m[1], target_centre[0], target_centre[1]),
    )
    cx, cy = polygon_centroid(pts)
    pts = pts + (target_centre[0] - cx, target_centre[1] - cy)
    m = np.zeros(target.shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    coords = np.argwhere(m)
    if coords.size == 0:
        return 0.0, 0, 0
    ys, xs = coords[:, 0], coords[:, 1]
    if ys.size == 0:
        return 0.0, 0, 0
    tmpl = m[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
    canvas = np.pad(target, PAD).astype(np.float32)
    if tmpl.shape[0] > canvas.shape[0] or tmpl.shape[1] > canvas.shape[1]:
        return 0.0, 0, 0
    res = cv2.matchTemplate(canvas, tmpl, cv2.TM_CCORR)
    _, peak, _, loc = cv2.minMaxLoc(res)
    union = float(tmpl.sum()) + float(target.sum()) - peak
    return (
        (peak / union if union > 0 else 0.0),
        int(loc[0] - PAD - xs.min()),
        int(loc[1] - PAD - ys.min()),
    )


def solve_rotation(ring, target, centre, scale):
    best = (0.0, 0.0)
    for angle in np.arange(-180.0, 180.0, 1.0):
        v, _, _ = align(ring, target, centre, scale, angle, (0.0, 0.0))
        if v > best[0]:
            best = (v, float(angle))
    return best


def solve_scale(ring, target, centre, angle, nominal, span=0.02, step=0.0005):
    best = (0.0, nominal)
    for factor in 1.0 + np.arange(-span, span + step / 2, step):
        v, _, _ = align(ring, target, centre, nominal * factor, angle, (0.0, 0.0))
        if v > best[0]:
            best = (v, nominal * factor)
    return best


def refine_position(
    ring, edge_dt, origin_px, scale, angle, radius=6.0, step=0.5, spacing=2.0
):
    """Snap the fit to the drawn commune limit at full resolution.

    Samples the projected commune boundary every `spacing` metres and minimises
    its mean distance to the silhouette edge, which resolves the sub-pixel
    offset left by the half-resolution IoU peak.
    """
    pts = project(ring, scale, angle, origin_px)
    segs = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        n = max(int(np.hypot(*(b - a)) / (spacing * scale)), 2)
        segs.append(np.linspace(a, b, n, endpoint=False))
    samples = np.concatenate(segs)

    def cost(dx, dy):
        q = np.round(samples + (dx, dy)).astype(int)
        ok = (
            (q[:, 0] >= 0)
            & (q[:, 0] < edge_dt.shape[1])
            & (q[:, 1] >= 0)
            & (q[:, 1] < edge_dt.shape[0])
        )
        return float(edge_dt[q[ok, 1], q[ok, 0]].mean()) if ok.any() else 1e9

    best = (cost(0, 0), 0.0, 0.0)
    for dy in np.arange(-radius, radius + step / 2, step):
        for dx in np.arange(-radius, radius + step / 2, step):
            c = cost(dx, dy)
            if c < best[0]:
                best = (c, float(dx), float(dy))
    return best[1], best[2], best[0]


@dataclass
class Georef:
    """Everything the caller needs to place geometry on the sheet."""

    pdf: Path
    doc: pymupdf.Document
    page: pymupdf.Page
    rect: pymupdf.Rect
    rgb: np.ndarray
    parcel: dict
    ring: np.ndarray
    scale: float  # px per metre on the raster
    angle: float  # radians, 0 for a north-up sheet
    origin: tuple  # (x_l93, y_l93, px, py) of the pinned point
    nominal: float  # px per metre as printed (1:7000)
    iou: float
    snap_px: float
    residual_median_px: float
    residual_mean_px: float

    @property
    def size(self) -> tuple[int, int]:
        return self.rgb.shape[1], self.rgb.shape[0]

    def to_px(self, pts_l93) -> np.ndarray:
        return project(
            np.asarray(pts_l93, dtype=float), self.scale, self.angle, self.origin
        )

    def metres(self, px: float) -> float:
        return px / self.scale if self.scale else 0.0


def georeference(pdf_path, idu: str, progress=None) -> Georef:
    """Solve the sheet's georeference and return the parcel in pixel space."""
    say = progress or (lambda *_: None)
    parcel = geo.parcel_l93(idu)
    ring = commune_ring(idu[:5])

    doc = pymupdf.open(pdf_path)
    page = doc[0]
    rgb, rect = page_raster(page)
    h, w = rgb.shape[:2]
    say(
        f"sheet          {Path(pdf_path).name}  {rect.width * 25.4 / 72:.1f} x {rect.height * 25.4 / 72:.1f} mm"
    )
    say(f"raster         {w} x {h} px @ {w / rect.width * 72:.0f} dpi")
    say(
        f"parcel         {parcel['idu']}  {parcel['area_m2']:,} m2  {parcel['commune']}"
    )

    nominal = (1 / (7000 * 0.0254)) * 72 * (w / rect.width)
    origin_m = polygon_centroid(ring)

    sil = commune_silhouette(rgb)
    sil8 = cv2.resize(sil, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    sil8 = (sil8 > 0.4).astype(np.uint8)
    _, angle = solve_rotation(ring, sil8, mask_centroid(sil8), nominal / 8)

    sil4 = cv2.resize(sil, (w // FIT_RES, h // FIT_RES), interpolation=cv2.INTER_AREA)
    sil4 = (sil4 > 0.4).astype(np.uint8)
    centre4 = mask_centroid(sil4)
    _, scale4 = solve_scale(ring, sil4, centre4, angle, nominal / FIT_RES)
    _, scale4 = solve_scale(
        ring, sil4, centre4, angle, scale4, span=0.001, step=0.00005
    )
    scale = scale4 * FIT_RES

    sil2 = cv2.resize(sil, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    sil2 = (sil2 > 0.4).astype(np.uint8)
    centre2 = mask_centroid(sil2)
    assert centre2 is not None
    iou2, dx, dy = align(ring, sil2, centre2, scale / 2, angle, (0.0, 0.0))
    origin_px = [origin_m[0], origin_m[1], (centre2[0] + dx) * 2, (centre2[1] + dy) * 2]

    edge = (sil ^ cv2.erode(sil, np.ones((3, 3), np.uint8))).astype(np.uint8)
    edge_dt = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 5)
    rdx, rdy, rcost = refine_position(ring, edge_dt, origin_px, scale, angle)
    origin_px[2] += rdx
    origin_px[3] += rdy
    origin = tuple(origin_px)

    say(
        f"georeference   rotation {angle:+.1f} deg, {scale:.5f} px/m "
        f"({(scale / nominal - 1) * 100:+.2f} % vs printed 1:7000), commune IoU {iou2:.4f}"
    )
    say(
        f"snap           {rdx:+.1f}, {rdy:+.1f} px, commune outline off by {rcost:.2f} px "
        f"({rcost / scale:.1f} m)"
    )

    def to_px(pts):
        return project(np.asarray(pts, dtype=float), scale, angle, origin)

    ink = (rgb.max(axis=2) < 110).astype(np.uint8)
    dt = cv2.distanceTransform(1 - ink, cv2.DIST_L2, 5)
    samples = []
    for r in parcel["rings_l93"]:
        px = to_px(r)
        for i in range(len(px) - 1):
            a, b = px[i], px[i + 1]
            n = max(int(np.hypot(*(b - a))), 2)
            samples.append(np.linspace(a, b, n, endpoint=False))
    q = np.round(np.concatenate(samples)).astype(int)
    q = q[(q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)]
    res = dt[q[:, 1], q[:, 0]]
    say(
        f"edge residual  median {np.median(res):.2f} px ({np.median(res) / scale:.1f} m), "
        f"mean {res.mean():.2f} px ({res.mean() / scale:.1f} m) over {len(res)} samples"
    )

    return Georef(
        pdf=Path(pdf_path),
        doc=doc,
        page=page,
        rect=rect,
        rgb=rgb,
        parcel=parcel,
        ring=ring,
        scale=scale,
        angle=angle,
        origin=origin,
        nominal=nominal,
        iou=iou2,
        snap_px=rcost,
        residual_median_px=float(np.median(res)),
        residual_mean_px=float(res.mean()),
    )


def draw_parcel(geo: Georef) -> None:
    """Outline the parcel on the PDF page (kept in page points, not pixels)."""
    pt_per_px = geo.rect.width / geo.size[0]
    shape = geo.page.new_shape()
    for r in geo.parcel["rings_l93"]:
        pts = [
            pymupdf.Point(
                geo.rect.x0 + p[0] * pt_per_px, geo.rect.y0 + p[1] * pt_per_px
            )
            for p in geo.to_px(r)
        ]
        shape.draw_polyline(pts + [pts[0]])
    shape.finish(color=(0.85, 0.05, 0.05), width=2.2, closePath=True)
    shape.commit()

    anchor = geo.to_px(geo.parcel["rings_l93"][0])
    label_at = pymupdf.Point(
        geo.rect.x0 + (float(anchor[:, 0].min()) - 6) * pt_per_px,
        geo.rect.y0 + (float(anchor[:, 1].min()) - 4) * pt_per_px,
    )
    geo.page.insert_text(
        label_at,
        f"{geo.parcel['idu'][8:10]} {geo.parcel['idu'][10:]}  "
        f"{geo.parcel['area_m2']:,} m2",
        fontsize=13,
        color=(0.85, 0.05, 0.05),
        rotate=0,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--address", default=None)
    ap.add_argument("--parcel", default=None, help="cadastral idu, e.g. 91272000CR0247")
    ap.add_argument("--out-dir", default="ingest/plu_graph/parcel")
    args = ap.parse_args(argv)

    address = None
    if args.parcel:
        idu = args.parcel.upper()
    else:
        address = geo.geocode(args.address or DEFAULT_PARCEL_ADDRESS)
        idu = geo.parcel_idu_for_address(address)
        print(f"address        {address['label']}  ({address['address_id']})")

    georef = georeference(args.pdf, idu, progress=print)
    draw_parcel(georef)
    anchor = georef.to_px(georef.parcel["rings_l93"][0])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = out_dir / f"plu_parcel_{georef.parcel['idu']}.pdf"
    georef.doc.save(pdf_out, garbage=3, deflate=True)

    saved = pymupdf.open(pdf_out)
    pt_per_px = georef.rect.width / georef.size[0]
    m = 110
    clip = pymupdf.Rect(
        georef.rect.x0 + (float(anchor[:, 0].min()) - m) * pt_per_px,
        georef.rect.y0 + (float(anchor[:, 1].min()) - m) * pt_per_px,
        georef.rect.x0 + (float(anchor[:, 0].max()) + m * 1.6) * pt_per_px,
        georef.rect.y0 + (float(anchor[:, 1].max()) + m) * pt_per_px,
    )
    zoom = out_dir / f"plu_parcel_{georef.parcel['idu']}_zoom.png"
    saved[0].get_pixmap(clip=clip, dpi=300).save(zoom)
    whole = out_dir / f"plu_parcel_{georef.parcel['idu']}_sheet.png"
    saved[0].get_pixmap(dpi=60).save(whole)
    print(f"written        {pdf_out}")
    print(f"written        {zoom}")
    print(f"written        {whole}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
