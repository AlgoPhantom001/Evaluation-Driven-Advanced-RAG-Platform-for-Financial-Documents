from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from secrag.core.errors import ProviderError
from secrag.core.logging import get_logger
from secrag.providers.base import ChatMessage, Completion, HTTPProvider

log = get_logger(__name__)


class GeminiProvider(HTTPProvider):
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> dict[str, Any]:
        system_parts = [m.content for m in messages if m.role == "system"]
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in messages
            if m.role != "system"
        ]

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return payload

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts") or []
        return "".join(part.get("text", "") for part in parts)

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> Completion:
        payload = self._payload(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )

        async for attempt in self._retryer():
            with attempt:
                response = await self.client.post(
                    f"/models/{self.model}:generateContent",
                    json=payload,
                    headers=self._headers(),
                )
                self._raise_for_status(response)
                data = response.json()

        if not isinstance(data, dict):
            msg = "Gemini returned an unexpected response shape"
            raise ProviderError(msg, detail=str(data)[:500])

        usage = data.get("usageMetadata", {})
        return self._record(
            Completion(
                text=self._extract_text(data),
                provider=self.name,
                model=self.model,
                prompt_tokens=int(usage.get("promptTokenCount", 0)),
                completion_tokens=int(usage.get("candidatesTokenCount", 0)),
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
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=False
        )

        collected: list[str] = []
        prompt_tokens = completion_tokens = 0

        async with self.client.stream(
            "POST",
            f"/models/{self.model}:streamGenerateContent?alt=sse",
            json=payload,
            headers=self._headers(),
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                self._raise_for_status(response)

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line.removeprefix("data: ").strip())
                except json.JSONDecodeError:
                    continue

                if usage := event.get("usageMetadata"):
                    prompt_tokens = int(usage.get("promptTokenCount", prompt_tokens))
                    completion_tokens = int(usage.get("candidatesTokenCount", completion_tokens))

                if piece := self._extract_text(event):
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
