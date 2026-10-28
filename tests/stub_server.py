from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

from aiohttp import web

from chat.clients import Client

CHAT_PATH = "/v1/chat/completions"
DECISIONS_PATH = "/v1/decisions"

Handler = Callable[[dict[str, Any]], dict[str, Any]]


class StubFactory(Protocol):
    def __call__(self, handler: Handler, *, path: str = CHAT_PATH) -> Awaitable[Client]: ...


@asynccontextmanager
async def stub_client(handler: Handler, *, path: str = CHAT_PATH) -> AsyncIterator[Client]:
    """A localhost endpoint at `path` that answers every POST from `handler`."""

    async def respond(request: web.Request) -> web.Response:
        return web.json_response(handler(await request.json()))

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_post(path, respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    addresses = runner.addresses
    assert addresses, "the stub server did not bind a port"
    client = Client(base_url=f"http://127.0.0.1:{addresses[0][1]}/v1", api_key="test")
    try:
        yield client
    finally:
        await client.aclose()
        await runner.cleanup()


def completion(model: str, message: dict[str, Any], *, prompt: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt + completion_tokens,
        },
    }
