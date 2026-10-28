import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai.types.chat import ChatCompletionMessageToolCall

from agent.tools import (
    ComputeBuildableEnvelope,
    EnvelopeCalcInput,
    OverlayCarveout,
    compute_buildable_envelope,
)
from chat.tools import ToolContext


def input_data() -> EnvelopeCalcInput:
    return EnvelopeCalcInput(
        parcel_wkt="POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
        setback_frontage_m=1.0,
        setback_side_views_m=1.0,
        setback_side_blank_m=2.0,
        overlay_carveouts=[
            OverlayCarveout(
                geometry_wkt="POLYGON ((4 4, 7 4, 7 7, 4 7, 4 4))",
                reference="ER voirie_creation_amelioration",
            )
        ],
        height_max_m=12.0,
        height_max_levels=3,
    )


def test_compute_buildable_envelope_uses_max_setback_and_level_cap() -> None:
    result = compute_buildable_envelope(input_data())

    assert result.footprint_area_m2 == pytest.approx(27.0)
    assert result.effective_height_m == pytest.approx(9.0)
    assert result.height_binding_constraint == "levels"
    assert result.buildable_volume_m3 == pytest.approx(243.0)
    assert any("Subtracted overlay" in line for line in result.calculation_trace)
    assert result.simplification_note.startswith("Uniform inward buffer")


def test_compute_buildable_envelope_accepts_multipolygon_parcels() -> None:
    result = compute_buildable_envelope(
        EnvelopeCalcInput(
            parcel_wkt=(
                "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), "
                "((6 0, 10 0, 10 4, 6 4, 6 0)))"
            ),
            setback_frontage_m=0.5,
            setback_side_views_m=0.5,
            height_max_m=10.0,
        )
    )

    assert result.footprint_area_m2 > 0
    assert result.height_binding_constraint == "metric"


async def test_compute_buildable_envelope_tool_returns_output() -> None:
    tool = ComputeBuildableEnvelope()
    call = cast(
        ChatCompletionMessageToolCall,
        SimpleNamespace(id="call_1"),
    )

    messages = await tool.execute(ToolContext(), call, input_data().model_dump())
    content = cast(dict[str, Any], messages[0])["content"]
    assert isinstance(content, str)
    payload: dict[str, Any] = json.loads(content)

    assert tool.name == "compute_buildable_envelope"
    assert payload["footprint_area_m2"] == pytest.approx(27.0)
    assert payload["height_binding_constraint"] == "levels"
    parameters = tool.into_chat_tool()["function"].get("parameters")
    assert isinstance(parameters, dict)
    properties = parameters.get("properties")
    assert isinstance(properties, dict)
    assert "parcel_wkt" in properties


async def test_compute_buildable_envelope_tool_returns_geometry_errors() -> None:
    tool = ComputeBuildableEnvelope()
    call = cast(
        ChatCompletionMessageToolCall,
        SimpleNamespace(id="call_1"),
    )
    args = input_data().model_dump()
    args["parcel_wkt"] = "not a polygon"

    messages = await tool.execute(ToolContext(), call, args)
    content = cast(dict[str, Any], messages[0])["content"]
    assert isinstance(content, str)
    payload: dict[str, Any] = json.loads(content)

    assert "invalid parcel WKT" in payload["error"]
