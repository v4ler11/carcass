from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self

from openai import AsyncOpenAI

from core.config import CHAT_MAX_RETRIES, CHAT_TIMEOUT_S


class Client:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        extra_body: Mapping[str, Any] | None = None,
        sdk: AsyncOpenAI | None = None,
        max_retries: int = CHAT_MAX_RETRIES,
        timeout_s: float = CHAT_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.extra_body: dict[str, Any] = dict(extra_body or {})
        self._sdk = sdk
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    @property
    def sdk(self) -> AsyncOpenAI:
        if self._sdk is None:
            self._sdk = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                max_retries=self.max_retries,
                timeout=self.timeout_s,
            )
        return self._sdk

    async def aclose(self) -> None:
        if self._sdk is not None:
            await self._sdk.close()
            self._sdk = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
