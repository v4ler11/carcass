import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from agent import pipeline
from chat.clients import Client


class FakeResult(BaseModel):
    answer: str


async def test_s1_saves_and_prints_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = FakeResult(answer="ok")
    calls: list[dict[str, Any]] = []

    async def fake_node(**kwargs: Any) -> tuple[Any, ...]:
        calls.append(kwargs)
        return result, None, None, None

    results_path = tmp_path / "out" / "s1" / "results.json"
    monkeypatch.setattr(pipeline, "s1_overlay_resolution_node", fake_node)
    monkeypatch.setattr(pipeline, "S1_RESULTS_PATH", results_path)

    results = await pipeline.run_pipeline(
        client=cast(Client, object()), model="m", stage="s1"
    )

    assert results == {"s1": result}
    assert len(calls) == 1
    assert json.loads(results_path.read_text()) == {"answer": "ok"}
    assert capsys.readouterr().out.strip() == result.model_dump_json(
        indent=2, by_alias=True
    )


async def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown stage 's9'"):
        await pipeline.run_pipeline(
            client=cast(Client, object()), model="m", stage="s9"
        )
