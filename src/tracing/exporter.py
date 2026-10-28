"""Disk export: append spans to daily JSONL, with large values in SQLite."""

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from tracing.db.content import (
    CONTENT_KEYS,
    MESSAGE_KEYS,
    REF_KEY,
    VALUE_KEYS,
    ContentStore,
    encode_messages,
    encode_value,
    resolve_content,
)


class JsonlSpanExporter(SpanExporter):
    """Append spans to `<directory>/YYYYMMDD.jsonl`, one JSON object per line.

    Large values (message content, multimodal text and image URLs, reasoning,
    reasoning signatures, tool call results, tool definitions) are replaced with
    a `{"$ref": sha256}` pointer and stored, zstd-compressed, in
    `<directory>/blobs_traces.db`. Content is written before the line: a crash
    can leave an unreferenced blob, never a dangling ref.

    The day comes from the span's start, not from the flush, so a batch
    exported after midnight still lands in the file of the day it happened.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.db_path = directory / "blobs_traces.db"
        directory.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        daily: dict[str, list[str]] = {}
        rows: list[dict[str, Any]] = []
        for span in spans:
            payload = json.loads(span.to_json(indent=None))
            attributes = payload["attributes"]
            for key in CONTENT_KEYS:
                raw = attributes.get(key)
                if not isinstance(raw, str):
                    continue
                try:
                    if key in MESSAGE_KEYS:
                        attributes[key], content_rows = encode_messages(raw)
                    elif key in VALUE_KEYS:
                        attributes[key], content_rows = encode_value(raw)
                    else:
                        continue
                except ValueError:
                    # Not JSON: keep the value and export the span anyway.
                    continue
                rows.extend(content_rows)

            day = datetime.fromtimestamp(
                (span.start_time or 0) / 1_000_000_000
            ).strftime("%Y%m%d")
            daily.setdefault(day, []).append(json.dumps(payload))

        try:
            if rows:
                with ContentStore(self.db_path) as store:
                    store.put_many(rows)
            for day, lines in daily.items():
                with (self.directory / f"{day}.jsonl").open(
                    "a", encoding="utf-8"
                ) as file:
                    file.write("\n".join(lines) + "\n")
        except OSError, sqlite3.Error:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Nothing to release: each batch opens and closes its store."""


def _has_content_ref(span: dict[str, Any]) -> bool:
    return any(
        isinstance(value := span["attributes"].get(key), str) and REF_KEY in value
        for key in CONTENT_KEYS
    )


def read_spans(directory: Path, *, resolve: bool = True) -> list[dict[str, Any]]:
    """Every span written under `directory`, oldest day first.

    Content referenced by `$ref` is resolved from `<directory>/blobs_traces.db`.
    A missing ref raises instead of returning a span whose transcript or tool
    list silently lost a value. `resolve=False` returns the JSONL payload as
    written: refs stay pointers and no blob is read.
    """
    spans: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as file:
            spans.extend(json.loads(line) for line in file if line.strip())

    if not resolve or not any(_has_content_ref(span) for span in spans):
        return spans

    with ContentStore(directory / "blobs_traces.db") as store:
        for span in spans:
            for key in CONTENT_KEYS:
                raw = span["attributes"].get(key)
                if not isinstance(raw, str) or REF_KEY not in raw:
                    continue
                span["attributes"][key] = resolve_content(
                    key, raw, store, span_name=span["name"]
                )
    return spans
