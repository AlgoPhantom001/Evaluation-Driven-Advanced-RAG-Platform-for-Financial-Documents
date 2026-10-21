from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from secrag.core.errors import ProviderError
from secrag.core.logging import get_logger
from secrag.providers.base import ChatMessage, Completion, HTTPProvider

log = get_logger(__name__)


class GroqProvider(HTTPProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> Completion:
        payload = self._payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stream=False,
        )

        async for attempt in self._retryer():
            with attempt:
                response = await self.client.post(
                    "/chat/completions", json=payload, headers=self._headers()
                )
                self._raise_for_status(response)
                data = response.json()

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            usage = data.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            msg = "Groq returned an unexpected response shape"
            raise ProviderError(msg, detail=str(data)[:500]) from exc

        return self._record(
            Completion(
                text=text,
                provider=self.name,
                model=data.get("model", self.model),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                finish_reason=str(choice.get("finish_reason", "stop")),
            )
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        payload = self._payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            stream=True,
        )

        collected: list[str] = []
        prompt_tokens = completion_tokens = 0

        async with self.client.stream(
            "POST", "/chat/completions", json=payload, headers=self._headers()
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                self._raise_for_status(response)

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                blob = line.removeprefix("data: ").strip()
                if blob == "[DONE]":
                    break
                try:
                    event = json.loads(blob)
                except json.JSONDecodeError:
                    continue

                if usage := event.get("x_groq", {}).get("usage") or event.get("usage"):
                    prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
                    completion_tokens = int(usage.get("completion_tokens", completion_tokens))

                for choice in event.get("choices", []):
                    if piece := choice.get("delta", {}).get("content"):
                        collected.append(piece)
                        yield piece

        text = "".join(collected)
        if not prompt_tokens and not completion_tokens:
            prompt_tokens, completion_tokens = self._estimate(messages, text)
        self._record(
            Completion(
                text=text,
                provider=self.name,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
