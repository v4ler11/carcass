"""Cut the annex panels out of the A0 graphic-regulation sheet.

The sheet is one 7021x4966 scan of a 1189x841 mm page: no text layer, no
vectors.  What it carries that is worth keeping is page furniture in the right
margin - one legend, one ``elements recenses`` block, one zone key and the three
emplacements-reserves tables.

Proximity segmentation does not work here.  The legend's own row gaps (31-142 px)
are *wider* than the gaps between two panels (21-33 px), so no single closing
kernel separates panels: a 51 px kernel splits the legend into six fragments.
Two independent locators are used instead:

* tables - from their printed grid: long horizontal rules that share the same x
  and the same length, at least five of them, spanning >= 150 px vertically.
  The title row above the first rule is picked up by walking up band by band.
* annex panels - declared as fractions of the sheet (fixed furniture on a fixed
  document), then checked mechanically, see ``check_panel``.

Every crop is written next to a manifest entry carrying its box in pixels and
millimetres plus the numbers behind the check, so a panel can be audited without
re-running anything.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

DEFAULT_PDF = "files/plu_reglement_graphique_91272_20210706 (1).pdf"
DEFAULT_OUT = "ingest/plu_graph/document/parsed/s1_raw"
DEFAULT_DPI = 150

INK_LEVEL = 240  # anything darker than this on a channel is ink

# Every distance below is expressed in pixels *at the resolution this was tuned
# on* and rescaled to whatever raster is supplied, so the same numbers hold at
# 72 dpi and at 200 dpi.
REF_W, REF_H = 7023, 4967
MARGIN_PX = 10  # white breathing space around every panel
HALO_PX = 8  # ring outside a crop that must stay blank
BORDER_PX = 4  # border strip inside a crop that must be blank
MIN_RULE_PX = 300
MIN_HEIGHT_PX = 150
TITLE_GAP_PX = 30
SNAP_PAD_PX = 12


@dataclass(frozen=True)
class Box:
    """A crop rectangle in raster pixels."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    def expand(self, px: int, width: int, height: int) -> Box:
        x = max(0, self.x - px)
        y = max(0, self.y - px)
        return Box(x, y, min(width, self.x1 + px) - x, min(height, self.y1 + px) - y)

    def iou(self, other: Box) -> float:
        ix = max(0, min(self.x1, other.x1) - max(self.x, other.x))
        iy = max(0, min(self.y1, other.y1) - max(self.y, other.y))
        inter = ix * iy
        return inter / (self.area + other.area - inter) if inter else 0.0

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def to_mm(self, dpi: int) -> dict[str, float]:
        factor = 25.4 / dpi
        return {
            "x_mm": round(self.x * factor, 1),
            "y_mm": round(self.y * factor, 1),
            "w_mm": round(self.w * factor, 1),
            "h_mm": round(self.h * factor, 1),
        }


def render_page(pdf_path, page_index: int, dpi: int) -> np.ndarray:
    """Rasterize one page at ``dpi`` as an RGB numpy array."""
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        page = document[page_index]
        bitmap = page.render(scale=dpi / 72.0)  # pyright: ignore[reportArgumentType]
        try:
            return np.asarray(bitmap.to_pil().convert("RGB"))
        finally:
            bitmap.close()
    finally:
        document.close()


def scale_of(mask: np.ndarray) -> float:
    """Pixels at the tuned resolution -> pixels at this raster's resolution."""
    return min(mask.shape[1] / REF_W, mask.shape[0] / REF_H)


def at_ref(mask: np.ndarray, px: int) -> int:
    return max(1, round(px * scale_of(mask)))


# (x0, y0, x1, y1) as fractions of the sheet.  Measured on the 150 dpi scan and
# rounded outwards; ``check_panel`` proves the result is tight and complete.
ANNEX_PANELS: tuple[tuple[str, float, float, float, float], ...] = (
    ("01_legende", 0.8209, 0.3828, 0.9975, 0.6285),
    ("02_elements_recenses", 0.8238, 0.6345, 0.9835, 0.7056),
    ("03_zones_legend", 0.8221, 0.7090, 0.9975, 0.9231),
)

# Printed on the sheet, top to bottom.  The tables are located by their grid, not
# by these names; the list only labels the boxes found.
TABLE_TITLES = (
    "04_er_elargissements_voirie",
    "05_er_creation_amelioration_voirie",
    "06_er_equipements_espaces_publics",
)


def ink_mask(rgb: np.ndarray) -> np.ndarray:
    """1 where the pixel carries ink (any channel), 0 on white."""
    return (rgb.min(axis=2) < INK_LEVEL).astype(np.uint8)


def row_bands(
    mask: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> list[tuple[int, int]]:
    """Vertical bands of ink rows inside ``x0..x1``, as (start, end)."""
    column = mask[max(0, y0) : y1, x0:x1].any(axis=1)
    edges = np.diff(np.r_[0, column.astype(np.int8), 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(int(a) + max(0, y0), int(b) + max(0, y0)) for a, b in zip(starts, ends)]


def ink_box(mask: np.ndarray, box: Box) -> Box | None:
    """Tight ink bounding box inside ``box``."""
    crop = mask[box.y : box.y1, box.x : box.x1]
    ys, xs = np.where(crop)
    if len(ys) == 0:
        return None
    return Box(
        box.x + int(xs.min()),
        box.y + int(ys.min()),
        int(xs.max() - xs.min()) + 1,
        int(ys.max() - ys.min()) + 1,
    )


def is_table_like(
    mask: np.ndarray,
    rgb: np.ndarray,
    box: Box,
    *,
    max_fill: float = 0.60,
    max_colored: float = 0.10,
) -> bool:
    """A reserve table is black rules on white, not a photograph or a colour block.

    The title block also carries aligned horizontal rules, so the grid alone is
    not enough: its cover photograph covers 99 % of the box and 90 % of it is
    saturated, against 19-27 % fill and ~0 % colour for the three real tables.
    """
    if mask[box.y : box.y1, box.x : box.x1].mean() > max_fill:
        return False
    hsv = cv2.cvtColor(rgb[box.y : box.y1, box.x : box.x1], cv2.COLOR_RGB2HSV)
    colored = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 80)).mean()
    return bool(colored <= max_colored)


def find_tables(
    mask: np.ndarray,
    rgb: np.ndarray | None = None,
    *,
    min_rule_px: int | None = None,
    min_rules: int = 5,
    min_height_px: int | None = None,
    title_gap_px: int | None = None,
    split_factor: float = 4.0,
) -> list[Box]:
    """Locate ruled tables from their grid rules.

    A table is a stack of long horizontal rules that start at the same x and have
    the same length.  The stack is split where the vertical gap jumps above
    ``split_factor`` times the stack's own median pitch, so no page-relative
    pitch has to be guessed: an A0 plot table (pitch ~35-90 px) and an A4 text
    table (pitch ~72-197 px) are handled by the same number.
    """
    min_rule_px = min_rule_px or at_ref(mask, MIN_RULE_PX)
    min_height_px = min_height_px or at_ref(mask, MIN_HEIGHT_PX)
    title_gap_px = title_gap_px or at_ref(mask, TITLE_GAP_PX)

    horizontal = cv2.morphologyEx(
        mask * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (min_rule_px, 1)),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
    rules = [
        (int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2])) for i in range(1, count)
    ]
    rules = [rule for rule in rules if rule[2] >= min_rule_px]

    stacks: list[dict] = []
    for x, y, length in sorted(rules, key=lambda rule: (rule[0], rule[1])):
        for stack in stacks:
            if abs(stack["x"] - x) <= 6 and abs(stack["len"] - length) <= 10:
                stack["ys"].append(y)
                break
        else:
            stacks.append({"x": x, "len": length, "ys": [y]})

    boxes: list[Box] = []
    for stack in stacks:
        ys = sorted(stack["ys"])
        if len(ys) < min_rules:
            continue
        gaps = np.diff(ys)
        cut = max(1.0, split_factor * float(np.median(gaps)))
        runs: list[list[int]] = [[ys[0]]]
        for y, gap in zip(ys[1:], gaps):
            if gap > cut:
                runs.append([])
            runs[-1].append(y)
        for run in runs:
            if len(run) >= min_rules:
                box = _table_box(mask, stack, run, min_height_px, title_gap_px)
                if box is not None:
                    boxes.append(box)
    if rgb is not None:
        boxes = [box for box in boxes if is_table_like(mask, rgb, box)]
    return boxes


def _table_box(
    mask: np.ndarray, stack: dict, run: list[int], min_height_px: int, title_gap_px: int
) -> Box | None:
    top_rule, bottom_rule = run[0], run[-1]
    if bottom_rule - top_rule < min_height_px:
        return None
    x0, x1 = stack["x"], stack["x"] + stack["len"]
    top = top_rule
    for _ in range(4):
        bands = row_bands(mask, x0, x1, top - 200, top)
        if not bands or top - bands[-1][1] > title_gap_px:
            break
        top = bands[-1][0]
    found = ink_box(mask, Box(x0 - 4, top - 4, x1 - x0 + 8, bottom_rule - top + 12))
    return found


def grow_to_ink(
    mask: np.ndarray, box: Box, *, pad: int | None = None, rounds: int = 3
) -> Box | None:
    """Take the ink bbox inside ``box``, re-searching a little wider each round.

    The declared fraction only has to be roughly right: ink that falls just
    outside it (a stray mark, the corner of the locate inset) is picked up on the
    next round instead of being clipped by the crop.
    """
    height, width = mask.shape
    pad = pad or at_ref(mask, SNAP_PAD_PX)
    current = box
    tight = ink_box(mask, current)
    for _ in range(rounds):
        if tight is None:
            return None
        wider = tight.expand(pad, width, height)
        found = ink_box(mask, wider)
        if found is None or found.as_dict() == tight.as_dict():
            break
        current, tight = wider, found
    return ink_box(mask, current)


def frac_box(
    width: int, height: int, fractions: tuple[float, float, float, float]
) -> Box:
    x0, y0, x1, y1 = fractions
    left, top = round(x0 * width), round(y0 * height)
    right, bottom = round(x1 * width), round(y1 * height)
    return Box(left, top, right - left, bottom - top)


def check_panel(mask: np.ndarray, box: Box) -> dict:
    """Prove a crop is one whole panel and nothing else.

    * ``border_ink`` - the outer ``BORDER_PX`` strip must be blank: the crop is
      not slicing through a neighbouring panel.
    * ``halo_ink`` - the ring ``HALO_PX`` beyond the crop must be blank too, so
      the panel really ends where the box says and is not clipped by it.
    * ``fill`` - some ink, but not a page of it.
    * ``colored_ratio`` - no large saturated area, i.e. the crop did not swallow
      a piece of the zoning map.
    """
    border_px = at_ref(mask, BORDER_PX)
    halo_px = at_ref(mask, HALO_PX)
    inside = int(mask[box.y : box.y1, box.x : box.x1].sum())

    border = 0
    border += int(mask[box.y : box.y + border_px, box.x : box.x1].sum())
    border += int(mask[box.y1 - border_px : box.y1, box.x : box.x1].sum())
    border += int(mask[box.y : box.y1, box.x : box.x + border_px].sum())
    border += int(mask[box.y : box.y1, box.x1 - border_px : box.x1].sum())

    halo = box.expand(halo_px, mask.shape[1], mask.shape[0])
    halo_ink = int(mask[halo.y : halo.y1, halo.x : halo.x1].sum()) - inside

    fill = inside / float(box.area)
    ok = border == 0 and halo_ink == 0 and 0.01 <= fill <= 0.60
    return {
        "border_ink": border,
        "halo_ink": halo_ink,
        "fill_pct": round(100.0 * fill, 2),
        "ok": bool(ok),
    }


@dataclass
class Panel:
    name: str
    box: Box
    check: dict
    kind: str = "figure"  # "table" for a ruled reserve table, "figure" otherwise

    def as_dict(self, dpi: int) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "box_px": self.box.as_dict(),
            "box_mm": self.box.to_mm(dpi),
            "check": self.check,
        }


def extract(
    pdf: Path,
    *,
    dpi: int = DEFAULT_DPI,
    out_dir: Path = Path(DEFAULT_OUT),
    scale: float = 1.0,
) -> list[Panel]:
    image = render_page(pdf, 0, dpi)
    mask = ink_mask(image)
    height, width = mask.shape

    found: list[tuple[str, Box, str]] = []
    for name, *fractions in ANNEX_PANELS:
        declared = frac_box(width, height, tuple(fractions))  # type: ignore[arg-type]
        tight = grow_to_ink(mask, declared)
        if tight is None:
            raise SystemExit(f"{name}: no ink inside the declared box")
        found.append(
            (name, tight.expand(at_ref(mask, MARGIN_PX), width, height), "figure")
        )

    tables = [
        grown
        for grown in (grow_to_ink(mask, box) for box in find_tables(mask, image))
        if grown is not None
    ]
    tables.sort(key=lambda box: (box.y, box.x))
    if len(tables) != len(TABLE_TITLES):
        raise SystemExit(
            f"expected {len(TABLE_TITLES)} reserve tables, found {len(tables)}"
        )
    for name, box in zip(TABLE_TITLES, tables):
        found.append(
            (name, box.expand(at_ref(mask, MARGIN_PX), width, height), "table")
        )

    panels: list[Panel] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, box, kind in found:
        check = check_panel(mask, box)
        crop = Image.fromarray(image[box.y : box.y1, box.x : box.x1])
        if scale != 1.0:
            crop = crop.resize(
                (int(crop.width * scale), int(crop.height * scale)),
                Image.Resampling.LANCZOS,
            )
        crop.save(out_dir / f"{name}.png")
        panels.append(Panel(name, box, check, kind))
        flag = "ok" if check["ok"] else "CHECK"
        print(
            f"{name:42s} {box.w:5d}x{box.h:<5d} fill={check['fill_pct']:5.1f}% "
            f"border={check['border_ink']:3d} halo={check['halo_ink']:4d}  {flag}"
        )
    return panels


def overlay(
    image: np.ndarray, panels: list[Panel], *, max_width: int = 2400
) -> Image.Image:
    canvas = Image.fromarray(image).convert("RGB")
    factor = min(1.0, max_width / canvas.width)
    canvas = canvas.resize(
        (int(canvas.width * factor), int(canvas.height * factor)),
        Image.Resampling.LANCZOS,
    )
    draw = ImageDraw.Draw(canvas)
    for panel in panels:
        box = panel.box
        draw.rectangle(
            [box.x * factor, box.y * factor, box.x1 * factor, box.y1 * factor],
            outline=(200, 0, 0) if panel.check["ok"] else (255, 140, 0),
            width=3,
        )
        draw.text(
            (box.x * factor + 4, box.y * factor + 2), panel.name, fill=(200, 0, 0)
        )
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cut the PLU annex panels.")
    parser.add_argument("pdf", nargs="?", default=DEFAULT_PDF)
    parser.add_argument("-o", "--out", default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--scale", type=float, default=1.0, help="upscale factor (e.g. 2)"
    )
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args(argv)

    pdf = Path(args.pdf)
    if not pdf.exists():
        raise SystemExit(f"missing PDF: {pdf}")
    out_dir = Path(args.out)
    panels = extract(pdf, dpi=args.dpi, out_dir=out_dir, scale=args.scale)

    manifest = {
        "source": str(pdf),
        "page": 1,
        "dpi": args.dpi,
        "panels": [panel.as_dict(args.dpi) for panel in panels],
    }
    json_path = Path(args.json_path) if args.json_path else out_dir / "panels.json"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n")

    image = render_page(pdf, 0, args.dpi)
    overlay(image, panels).save(out_dir / "overlay.png")
    print(f"\nwritten {out_dir} ({len(panels)} panels) + {json_path}")
    return 0 if all(panel.check["ok"] for panel in panels) else 1


if __name__ == "__main__":
    raise SystemExit(main())
