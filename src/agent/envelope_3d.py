"""Draw every envelope run of the study as a 3D prism on the parcel.

The inputs are the `compute_buildable_envelope` calls S4 made, read from its
trace: parcel polygon, setbacks, height caps and floor height are the study's
own. Each call is re-run and must reproduce, in order, the footprint, height
and volume of the run table in `out/s4/result.md`, so the figure cannot drift
from the report. It shows the simplified whole-parcel bound, not two buildings.
"""

import itertools
import json
import re
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from shapely import wkt
from shapely.geometry import Polygon

from agent.paths import S4_MARKDOWN_PATH
from agent.tools import EnvelopeCalcInput, compute_buildable_envelope
from conversations import Node, list_conversations
from core import BASE_DIR

IMAGE_PATH = BASE_DIR / "docs" / "envelope_3d.png"
TOOL_NAME = "compute_buildable_envelope"
Z_MAX = 25.0
COLOR, EDGE = "#4c72b0", "#2a3f66"

_ROW = re.compile(
    r"^\| ([AB]\d [^|]*?) \|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ReportRun:
    label: str
    footprint_m2: float
    height_m: float
    volume_m3: float


def report_runs() -> list[ReportRun]:
    """The run table of the report, in order."""
    text = S4_MARKDOWN_PATH.read_text(encoding="utf-8")
    runs = [
        ReportRun(label, float(footprint), float(height), float(volume))
        for label, footprint, height, volume in _ROW.findall(text)
    ]
    if not runs:
        raise ValueError(f"no envelope run table found in {S4_MARKDOWN_PATH}")
    return runs


def _walk(node: Node) -> Iterator[Node]:
    yield node
    for child in node.children:
        yield from _walk(child)


def study_inputs() -> list[EnvelopeCalcInput]:
    """The envelope tool inputs of the latest S4 conversation that called it."""
    for conversation in list_conversations():  # most recent first
        if conversation.metadata.get("stage") != "s4":
            continue
        conversation.populate()
        calls: list[EnvelopeCalcInput] = []
        for root in conversation.tree:
            for node in _walk(root):
                if node.attributes.get("gen_ai.tool.name") != TOOL_NAME:
                    continue
                arguments = node.attributes["gen_ai.tool.call.arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append(EnvelopeCalcInput.model_validate(arguments))
        if calls:
            return calls
    raise ValueError("no S4 conversation with envelope tool calls in the traces")


def _polygons(footprint_wkt: str) -> list[Polygon]:
    geometry = wkt.loads(footprint_wkt)
    if isinstance(geometry, Polygon):
        return [geometry]
    return list(geometry.geoms)  # pyright: ignore[reportAttributeAccessIssue]


def _prism(ax, polygon: Polygon, height: float, origin: tuple[float, float]) -> None:
    ring = [(x - origin[0], y - origin[1]) for x, y in polygon.exterior.coords]
    sides = [
        [(*a, 0.0), (*b, 0.0), (*b, height), (*a, height)]
        for a, b in itertools.pairwise(ring)
    ]
    ax.add_collection3d(
        Poly3DCollection(sides, facecolor=COLOR, edgecolor=EDGE, alpha=0.35, linewidth=0.4)
    )
    roof = [[(*p, height) for p in ring]]
    ax.add_collection3d(Poly3DCollection(roof, facecolor=COLOR, edgecolor=EDGE, alpha=0.6))


def plot() -> None:
    runs = report_runs()
    inputs = study_inputs()
    if len(inputs) != len(runs):
        raise ValueError(f"trace has {len(inputs)} envelope calls, report lists {len(runs)} runs")

    parcel = wkt.loads(inputs[0].parcel_wkt)
    min_x, min_y, max_x, max_y = parcel.bounds  # pyright: ignore[reportAttributeAccessIssue]
    origin = (min_x, min_y)
    parcel_ring = [(x - min_x, y - min_y) for x, y in parcel.exterior.coords]  # pyright: ignore[reportAttributeAccessIssue]

    columns = 4
    rows = -(-len(runs) // columns)
    fig = plt.figure(figsize=(5 * columns, 4.6 * rows))
    for index, (run, call) in enumerate(zip(runs, inputs, strict=True), start=1):
        result = compute_buildable_envelope(call)
        got = (result.footprint_area_m2, result.effective_height_m, result.buildable_volume_m3)
        if got != (run.footprint_m2, run.height_m, run.volume_m3):
            raise ValueError(f"run {run.label}: tool gives {got}, report says {run}")

        ax = fig.add_subplot(rows, columns, index, projection="3d")
        ax.plot(*zip(*parcel_ring, strict=True), zs=0, color="black", linewidth=1.2)
        for polygon in _polygons(result.footprint_wkt):
            _prism(ax, polygon, result.effective_height_m, origin)
        ax.set_xlim(0, max_x - min_x)
        ax.set_ylim(0, max_y - min_y)
        ax.set_zlim(0, Z_MAX)
        ax.set_box_aspect((max_x - min_x, max_y - min_y, Z_MAX))
        ax.view_init(elev=24, azim=-62)
        ax.tick_params(labelsize=7)
        ax.set_title(
            textwrap.fill(run.label, 38)
            + f"\n{run.footprint_m2:,.0f} m² x {run.height_m:.0f} m = {run.volume_m3:,.0f} m³",
            fontsize=8,
        )
    fig.suptitle(
        "Envelope runs of the study (inputs from the S4 trace). Simplified whole-parcel "
        "bound: one prism, not two buildings; no inter-building distance, per-edge "
        "setbacks or green-space limits",
        fontsize=10,
    )
    fig.tight_layout()
    IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_PATH, dpi=130, bbox_inches="tight")
    print(f"wrote {IMAGE_PATH} ({len(runs)} runs)")


if __name__ == "__main__":
    plot()
