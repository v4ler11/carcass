import json
import sqlite3
from pathlib import Path

import pytest

from tracing.db.content import (
    RAW,
    ZSTD,
    ContentStore,
    decode,
    decode_messages,
    decode_value,
    encode,
    encode_messages,
    encode_value,
)
from tracing.exporter import read_spans


def test_round_trips_strings_and_multimodal_parts(tmp_path: Path) -> None:
    with ContentStore(tmp_path / "content.db") as store:
        values = ["hello", [{"type": "text", "text": "hi"}]]
        encoded = [encode(value) for value in values]
        store.put_many(
            [{"hash": e.digest, "codec": e.codec, "data": e.data} for e in encoded]
        )

        for value, e in zip(values, encoded, strict=True):
            row = store.get(e.digest)
            assert row is not None
            assert decode(e.digest.hex(), *row) == value


def test_duplicate_content_is_stored_once(tmp_path: Path) -> None:
    db = tmp_path / "content.db"
    encoded = encode("same")
    row = {"hash": encoded.digest, "codec": encoded.codec, "data": encoded.data}

    with ContentStore(db) as store:
        store.put_many([row, row, row])

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM content").fetchone()[0] == 1


def test_short_values_stay_raw_and_long_values_compress() -> None:
    assert encode("x").codec == RAW
    assert encode("x" * 10_000).codec == ZSTD


def test_missing_content_returns_none(tmp_path: Path) -> None:
    with ContentStore(tmp_path / "content.db") as store:
        assert store.get(b"\x00" * 32) is None


def test_encode_messages_only_hashes_message_content() -> None:
    raw = json.dumps(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        ]
    )

    rewritten, rows = encode_messages(raw)
    messages = json.loads(rewritten)

    assert set(messages[0]["content"]) == {"$ref"}
    assert messages[1]["content"] is None
    assert len(rows) == 1


def test_reasoning_fields_are_stored_out_of_line(tmp_path: Path) -> None:
    message = {
        "role": "assistant",
        "content": "answer",
        "reasoning": "thinking",
        "reasoning_details": [
            {
                "type": "reasoning.text",
                "text": "thinking",
                "signature": "signature",
                "format": "unknown",
                "index": 0,
            }
        ],
    }

    rewritten, rows = encode_messages(json.dumps([message]))
    stored = json.loads(rewritten)[0]

    assert set(stored["content"]) == {"$ref"}
    assert set(stored["reasoning"]) == {"$ref"}
    assert set(stored["reasoning_details"][0]["text"]) == {"$ref"}
    assert set(stored["reasoning_details"][0]["signature"]) == {"$ref"}

    with ContentStore(tmp_path / "content.db") as store:
        store.put_many(rows)
        assert json.loads(decode_messages(rewritten, store, span_name="chat m")) == [
            message
        ]

    # `reasoning` and the detail text are identical; the signature is distinct.
    with sqlite3.connect(tmp_path / "content.db") as connection:
        assert connection.execute("select count(*) from content").fetchone()[0] == 3


def test_signature_only_reasoning_detail_is_stored_out_of_line(
    tmp_path: Path,
) -> None:
    message = {
        "role": "assistant",
        "content": "answer",
        "reasoning_details": [{"type": "reasoning.text", "signature": "signature"}],
    }

    rewritten, rows = encode_messages(json.dumps([message]))
    stored = json.loads(rewritten)[0]

    assert set(stored["reasoning_details"][0]["signature"]) == {"$ref"}
    assert "text" not in stored["reasoning_details"][0]

    with ContentStore(tmp_path / "content.db") as store:
        store.put_many(rows)
        assert json.loads(decode_messages(rewritten, store, span_name="chat m")) == [
            message
        ]


def test_encode_value_round_trips_through_the_store(tmp_path: Path) -> None:
    value = [{"type": "function", "function": {"name": "lookup"}}]

    rewritten, rows = encode_value(json.dumps(value))
    assert set(json.loads(rewritten)) == {"$ref"}

    with ContentStore(tmp_path / "content.db") as store:
        store.put_many(rows)
        assert json.loads(decode_value(rewritten, store, span_name="chat m")) == value


def test_corrupt_content_fails_its_hash_check() -> None:
    encoded = encode("hello")

    with pytest.raises(ValueError, match="hash check"):
        decode(encoded.digest.hex(), RAW, b'"HELLO"')


def test_missing_ref_is_a_hard_error(tmp_path: Path) -> None:
    span = {
        "name": "chat test",
        "attributes": {
            "gen_ai.input.messages": json.dumps(
                [{"role": "user", "content": {"$ref": "00" * 32}}]
            )
        },
    }
    (tmp_path / "20260101.jsonl").write_text(json.dumps(span) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing content"):
        read_spans(tmp_path)
