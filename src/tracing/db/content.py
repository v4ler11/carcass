import hashlib
import json
from collections.abc import Sequence
from compression import zstd
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiosql

from tracing.attributes import Attr
from tracing.db.manager import DBManager

RAW = 0
ZSTD = 1
REF_KEY = "$ref"
MESSAGE_KEYS = (
    Attr.INPUT_MESSAGES,
    Attr.OUTPUT_MESSAGES,
    Attr.TOOL_CALL_RESULT,
)
VALUE_KEYS = (Attr.TOOL_DEFINITIONS,)
CONTENT_KEYS = MESSAGE_KEYS + VALUE_KEYS


class ContentStore(DBManager):
    Q: Any = aiosql.from_path(
        Path(__file__).parent / "sql", "sqlite3", mandatory_parameters=False
    )

    def _init_schema(self) -> None:
        with self.tx():
            self.Q.create_content_table(self.conn)

    def put_many(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.tx():
            self.Q.insert_content(self.conn, list(rows))

    def get(self, digest: bytes) -> tuple[int, bytes] | None:
        row = self.Q.get_content(self.conn, hash=digest)
        return (row["codec"], row["data"]) if row else None


@dataclass(frozen=True)
class EncodedContent:
    digest: bytes
    codec: int
    data: bytes


def canonical_bytes(value: Any) -> bytes:
    """The bytes hashed and stored: stable across processes and dict order."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def encode(value: Any) -> EncodedContent:
    raw = canonical_bytes(value)
    compressed = zstd.compress(raw)
    codec, data = (ZSTD, compressed) if len(compressed) < len(raw) else (RAW, raw)
    return EncodedContent(digest=hashlib.sha256(raw).digest(), codec=codec, data=data)


def decode(digest_hex: str, codec: int, data: bytes) -> Any:
    raw = zstd.decompress(data) if codec == ZSTD else data
    if hashlib.sha256(raw).hexdigest() != digest_hex:
        raise ValueError(f"content {digest_hex} failed its hash check")
    return json.loads(raw)


def _row(encoded: EncodedContent) -> dict[str, Any]:
    return {"hash": encoded.digest, "codec": encoded.codec, "data": encoded.data}


def _as_ref(value: Any, rows: list[dict[str, Any]]) -> dict[str, str]:
    encoded = encode(value)
    rows.append(_row(encoded))
    return {REF_KEY: encoded.digest.hex()}


def _is_ref_part_list(content: Any) -> bool:
    """True when every content part has a payload field we can ref separately."""
    if not isinstance(content, list) or not content:
        return False
    for part in content:
        if not isinstance(part, dict):
            return False
        kind = part.get("type")
        if kind == "text":
            if not isinstance(part.get("text"), str):
                return False
        elif kind == "image_url":
            image_url = part.get("image_url")
            if not isinstance(image_url, dict) or not isinstance(
                image_url.get("url"), str
            ):
                return False
        else:
            return False
    return True


def _encode_content(content: Any, rows: list[dict[str, Any]]) -> Any:
    """Ref text and image payloads, keeping the part structure readable."""
    if not _is_ref_part_list(content):
        return _as_ref(content, rows)
    for part in content:
        if part["type"] == "text":
            part["text"] = _as_ref(part["text"], rows)
        else:
            part["image_url"]["url"] = _as_ref(part["image_url"]["url"], rows)
    return content


def encode_messages(raw: str) -> tuple[str, list[dict[str, Any]]]:
    """Replace `content`, `reasoning`, and reasoning-detail text/signatures.

    Text and image payloads in multimodal content are ref'd individually; other
    content values are ref'd whole. Returns the rewritten JSON and the rows to
    store. A message without any of those fields (an assistant tool call) is
    left untouched. If nothing changed, the input string is returned as-is.
    """
    messages = json.loads(raw)
    if not isinstance(messages, list):
        return raw, []

    rows: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if content is not None:
            message["content"] = _encode_content(content, rows)

        reasoning = message.get("reasoning")
        if isinstance(reasoning, str):
            message["reasoning"] = _as_ref(reasoning, rows)

        details = message.get("reasoning_details")
        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                text = detail.get("text")
                if isinstance(text, str):
                    detail["text"] = _as_ref(text, rows)

                signature = detail.get("signature")
                if isinstance(signature, str):
                    detail["signature"] = _as_ref(signature, rows)

    if not rows:
        return raw, []
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":")), rows


def encode_value(raw: str) -> tuple[str, list[dict[str, Any]]]:
    """Replace a whole JSON attribute with a ref, storing the value once."""
    encoded = encode(json.loads(raw))
    return json.dumps({REF_KEY: encoded.digest.hex()}, ensure_ascii=False), [
        _row(encoded)
    ]


def decode_value(raw: str, store: ContentStore, *, span_name: str) -> str:
    """Resolve a whole-value ref back to its JSON."""
    ref = json.loads(raw)
    if not isinstance(ref, dict) or ref.keys() != {REF_KEY}:
        return raw
    digest_hex = ref[REF_KEY]
    row = store.get(bytes.fromhex(digest_hex))
    if row is None:
        raise ValueError(
            f"missing content {digest_hex} referenced by span {span_name!r}"
        )
    return json.dumps(
        decode(digest_hex, *row), ensure_ascii=False, separators=(",", ":")
    )


def _from_ref(value: Any, store: ContentStore, *, span_name: str) -> Any:
    if not isinstance(value, dict) or value.keys() != {REF_KEY}:
        return value
    digest_hex = value[REF_KEY]
    row = store.get(bytes.fromhex(digest_hex))
    if row is None:
        raise ValueError(
            f"missing content {digest_hex} referenced by span {span_name!r}"
        )
    return decode(digest_hex, *row)


def resolve_content(key: str, raw: str, store: ContentStore, *, span_name: str) -> str:
    """Resolve the refs of one content attribute; unknown keys pass through."""
    if key in MESSAGE_KEYS:
        return decode_messages(raw, store, span_name=span_name)
    if key in VALUE_KEYS:
        return decode_value(raw, store, span_name=span_name)
    return raw


def decode_messages(raw: str, store: ContentStore, *, span_name: str) -> str:
    """Resolve refs in content, reasoning, and reasoning-detail text/signatures."""
    if REF_KEY not in raw:
        return raw

    messages = json.loads(raw)
    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if isinstance(content, dict):
            message["content"] = _from_ref(content, store, span_name=span_name)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), dict):
                    part["text"] = _from_ref(part["text"], store, span_name=span_name)
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict) and isinstance(
                        image_url.get("url"), dict
                    ):
                        image_url["url"] = _from_ref(
                            image_url["url"], store, span_name=span_name
                        )

        reasoning = message.get("reasoning")
        if isinstance(reasoning, dict):
            message["reasoning"] = _from_ref(reasoning, store, span_name=span_name)

        details = message.get("reasoning_details")
        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                text = detail.get("text")
                if isinstance(text, dict):
                    detail["text"] = _from_ref(text, store, span_name=span_name)

                signature = detail.get("signature")
                if isinstance(signature, dict):
                    detail["signature"] = _from_ref(
                        signature, store, span_name=span_name
                    )

    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
