"""Compute a simplified buildable envelope from parcel and overlay geometry."""

import json
from typing import Any, Literal

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, Field, ValidationError
from shapely import wkt
from shapely.errors import GEOSException

from chat.tools import Tool, ToolContext
from chat.utils import tool_message


class OverlayCarveout(BaseModel):
    geometry_wkt: str = Field(
        ...,
        min_length=1,
        description=(
            "WKT polygon of the overlay/ER area to subtract, in the same CRS "
            "as the parcel."
        ),
    )
    reference: str = Field(
        ...,
        min_length=1,
        description=(
            "Which Phase 1 finding this carve-out comes from, e.g. "
            "'ER voirie_creation_amelioration'."
        ),
    )


class EnvelopeCalcInput(BaseModel):
    parcel_wkt: str = Field(
        ...,
        min_length=1,
        description=(
            "WKT polygon or multipolygon of the parcel boundary (Lambert-93 "
            "or another projected CRS in meters)."
        ),
    )
    setback_frontage_m: float = Field(..., ge=0)
    setback_side_views_m: float = Field(..., ge=0)
    setback_side_blank_m: float | None = Field(default=None, ge=0)
    overlay_carveouts: list[OverlayCarveout] = Field(default_factory=list)
    height_max_m: float = Field(..., gt=0)
    height_max_levels: int | None = Field(default=None, gt=0)
    floor_height_assumption_m: float = Field(
        default=3.0,
        gt=0,
        description=(
            "Used only if the levels cap is binding and needs conversion to meters."
        ),
    )
    simplification_note: str = Field(
        default=(
            "Uniform inward buffer applied using the maximum stated setback "
            "value; frontage-specific edge detection not performed in this "
            "timebox."
        ),
        description=(
            "Must be surfaced verbatim in the final report's assumptions section."
        ),
    )


class EnvelopeCalcOutput(BaseModel):
    footprint_wkt: str
    footprint_area_m2: float
    effective_height_m: float
    height_binding_constraint: Literal["metric", "levels"]
    buildable_volume_m3: float
    calculation_trace: list[str]
    simplification_note: str


def _polygon(wkt_text: str, label: str) -> Any:
    try:
        geometry = wkt.loads(wkt_text)
    except GEOSException as e:
        raise ValueError(f"invalid {label} WKT: {e}") from e
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"{label} must be a Polygon or MultiPolygon WKT")
    if geometry.is_empty:
        raise ValueError(f"{label} geometry is empty")
    return geometry


def compute_buildable_envelope(inp: EnvelopeCalcInput) -> EnvelopeCalcOutput:
    """Erode the parcel by the largest setback, then subtract overlays."""
    trace: list[str] = []

    parcel = _polygon(inp.parcel_wkt, "parcel")
    trace.append(f"Parcel area (raw): {parcel.area:.1f} m2")

    setbacks = [inp.setback_frontage_m, inp.setback_side_views_m]
    if inp.setback_side_blank_m is not None:
        setbacks.append(inp.setback_side_blank_m)
    max_setback = max(setbacks)
    trace.append(
        f"Applying uniform inward buffer of {max_setback} m (max of: {setbacks})"
    )

    buildable = parcel.buffer(-max_setback)
    if buildable.is_empty:
        raise ValueError(
            "Setback buffer exceeds parcel extent — no buildable area remains."
        )
    trace.append(f"Footprint area after setback erosion: {buildable.area:.1f} m2")

    for carveout in inp.overlay_carveouts:
        cut = _polygon(carveout.geometry_wkt, f"carve-out {carveout.reference!r}")
        before = buildable.area
        buildable = buildable.difference(cut)
        trace.append(
            f"Subtracted overlay [{carveout.reference}]: "
            f"{before:.1f} -> {buildable.area:.1f} m2"
        )

    if buildable.is_empty:
        raise ValueError("Overlay carve-outs remove all buildable area")

    footprint_area = buildable.area
    effective_height = inp.height_max_m
    binding: Literal["metric", "levels"] = "metric"
    if inp.height_max_levels is not None:
        levels_height = inp.height_max_levels * inp.floor_height_assumption_m
        trace.append(
            f"Level cap {inp.height_max_levels} x "
            f"{inp.floor_height_assumption_m}m/floor = {levels_height:.1f} m"
        )
        if levels_height < effective_height:
            effective_height = levels_height
            binding = "levels"
    trace.append(f"Binding height constraint: {binding} ({effective_height:.1f} m)")

    volume = footprint_area * effective_height
    trace.append(
        f"Buildable volume = {footprint_area:.1f} m2 x "
        f"{effective_height:.1f} m = {volume:.1f} m3"
    )

    return EnvelopeCalcOutput(
        footprint_wkt=buildable.wkt,
        footprint_area_m2=round(footprint_area, 1),
        effective_height_m=round(effective_height, 1),
        height_binding_constraint=binding,
        buildable_volume_m3=round(volume, 1),
        calculation_trace=trace,
        simplification_note=inp.simplification_note,
    )


class ComputeBuildableEnvelope(Tool):
    @property
    def name(self) -> str:
        return "compute_buildable_envelope"

    async def execute(
        self,
        ctx: ToolContext,
        tool_call: ChatCompletionMessageToolCall,
        args: dict[str, Any],
    ) -> list[ChatCompletionMessageParam]:
        try:
            result = compute_buildable_envelope(EnvelopeCalcInput.model_validate(args))
        except (ValidationError, ValueError) as e:
            return [tool_message(tool_call.id, json.dumps({"error": str(e)}))]
        return [
            tool_message(
                tool_call.id,
                result.model_dump_json(),
            )
        ]

    def into_chat_tool(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Compute a simplified buildable footprint and volume. "
                    "Applies one uniform inward buffer using the largest "
                    "stated setback, subtracts overlay/ER carve-outs, then "
                    "multiplies the remaining area by the binding height "
                    "(metric or level cap). Inputs must share one projected "
                    "CRS in meters."
                ),
                "parameters": EnvelopeCalcInput.model_json_schema(),
            },
        }
