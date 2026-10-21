from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from secrag.core.errors import ProviderError, RateLimitError
from secrag.core.logging import get_logger
from secrag.observability.tracing import estimate_tokens, record_usage

log = get_logger(__name__)

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, *, model: str, timeout_s: float = 45.0, max_retries: int = 3) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> Completion: ...

    @abstractmethod
    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]: ...

    async def health(self) -> bool:
        try:
            await self.complete(
                [ChatMessage(role="user", content="ping")], max_tokens=8, temperature=0.0
            )
        except Exception:
            return False
        return True

    def _record(self, completion: Completion) -> Completion:
        record_usage(completion.model, completion.prompt_tokens, completion.completion_tokens)
        return completion

    @staticmethod
    def _estimate(messages: Sequence[ChatMessage], output: str) -> tuple[int, int]:
        prompt = sum(estimate_tokens(m.content) for m in messages)
        return prompt, estimate_tokens(output)


class HTTPProvider(LLMProvider):
    base_url: str = ""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 45.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(model=model, timeout_s=timeout_s, max_retries=max_retries)
        if not api_key:
            msg = f"{self.name} provider requires an API key"
            raise ProviderError(msg)
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_s, connect=10.0),
                limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            )
            self._client_loop = loop
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        self._client_loop = None

    def _retryer(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential_jitter(initial=0.5, max=8.0),
            retry=retry_if_exception_type((httpx.TransportError, RateLimitError)),
            reraise=True,
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        body = response.text[:500]
        if response.status_code == 429:
            msg = f"{self.name} rate limited"
            raise RateLimitError(msg, detail=body)
        if response.status_code >= 500:
            msg = f"{self.name} upstream error {response.status_code}"
            raise RateLimitError(msg, detail=body)
        msg = f"{self.name} request failed with {response.status_code}"
        raise ProviderError(msg, detail=body)
