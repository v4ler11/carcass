import os
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider

from chat.clients import Client
from tests.stub_server import CHAT_PATH, Handler, StubFactory, stub_client
from tracing import configure

os.environ["CARCASS_ENV"] = "test"


@pytest.fixture(scope="session")
def traces_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("traces")


@pytest.fixture(scope="session", autouse=True)
def provider(traces_dir: Path) -> Iterator[TracerProvider]:
    configured = configure(traces_dir)
    yield configured
    configured.shutdown()


@pytest.fixture
async def stub() -> AsyncIterator[StubFactory]:
    """A fresh localhost endpoint per call, torn down with the test."""
    async with AsyncExitStack() as stack:

        async def make(handler: Handler, *, path: str = CHAT_PATH) -> Client:
            return await stack.enter_async_context(stub_client(handler, path=path))

        yield make
