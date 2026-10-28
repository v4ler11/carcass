import json
from pathlib import Path
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
from opentelemetry.sdk.trace import TracerProvider

from core.server import create_app
from tests.stub_server import StubFactory, completion
from tracing import read_spans

MODEL = "m"


def _body(**extra: object) -> dict[str, object]:
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}


async def test_returns_the_upstream_completion_unchanged(stub: StubFactory) -> None:
    upstream = await stub(lambda body: completion(MODEL, {"role": "assistant", "content": "ok"}, prompt=3, completion_tokens=1))
    app = create_app(client=upstream)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post("/v1/chat/completions", json=_body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "ok"
    assert payload["usage"]["total_tokens"] == 4


async def test_streams_one_chunk_then_done(stub: StubFactory) -> None:
    upstream = await stub(lambda body: completion(MODEL, {"role": "assistant", "content": "ok"}, prompt=3, completion_tokens=1))
    app = create_app(client=upstream)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post("/v1/chat/completions", json=_body(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"

    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": "ok"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


async def test_request_identity_reaches_every_span(
    stub: StubFactory, traces_dir: Path, provider: TracerProvider
) -> None:
    conversation = f"conv-{uuid4().hex}"
    upstream = await stub(lambda body: completion(MODEL, {"role": "assistant", "content": "ok"}, prompt=1, completion_tokens=1))
    app = create_app(client=upstream)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post(
            "/v1/chat/completions",
            headers={"X-Conversation-Id": conversation},
            json=_body(user="user-1", metadata={"tenant": "acme"}),
        )

    assert response.status_code == 200
    provider.force_flush()
    spans = [
        span
        for span in read_spans(traces_dir)
        if span["attributes"].get("gen_ai.conversation.id") == conversation
    ]

    assert {span["name"] for span in spans} == {"invoke_agent", "chat"}
    for span in spans:
        assert span["attributes"]["session.id"] == conversation
        assert span["attributes"]["user.id"] == "user-1"
        assert span["attributes"]["langfuse.trace.metadata.tenant"] == "acme"


async def test_missing_model_falls_back_to_the_configured_default(stub: StubFactory) -> None:
    upstream = await stub(lambda body: completion(MODEL, {"role": "assistant", "content": "ok"}, prompt=1, completion_tokens=1))
    app = create_app(client=upstream, default_model=MODEL)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        response = await http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json()["model"] == MODEL
