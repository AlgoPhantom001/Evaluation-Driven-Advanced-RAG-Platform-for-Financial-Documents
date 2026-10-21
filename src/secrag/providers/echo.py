from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator, Sequence

from secrag.providers.base import ChatMessage, Completion, LLMProvider

_CONTEXT_RE = re.compile(
    r"^\[(?P<marker>\d+)\]\s*(?P<label>.+?)\n(?P<body>.*?)(?=\n\[\d+\]\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _first_sentences(text: str, limit: int = 2) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    parts = [p for p in _SENTENCE_RE.split(cleaned) if p.strip()]
    return " ".join(parts[:limit]).strip()


class EchoProvider(LLMProvider):
    name = "echo"

    def __init__(self, *, model: str = "echo-1", max_context_used: int = 3, **_: object) -> None:
        super().__init__(model=model, timeout_s=1.0, max_retries=0)
        self.max_context_used = max_context_used

    def _synthesize(self, messages: Sequence[ChatMessage], *, json_mode: bool) -> str:
        prompt = "\n\n".join(m.content for m in messages if m.role == "user")

        if json_mode:
            return self._synthesize_json(prompt)

        blocks = _CONTEXT_RE.findall(prompt)
        if not blocks:
            digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
            return f"No context was supplied, so no grounded answer is available. (echo:{digest})"

        sentences: list[str] = []
        for marker, _label, body in blocks[: self.max_context_used]:
            if snippet := _first_sentences(body, limit=1):
                sentences.append(f"{snippet.rstrip('.')} [{marker}].")

        if not sentences:
            return "The retrieved context did not contain usable text. [1]"
        return " ".join(sentences)

    def _synthesize_json(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "supported" in lowered or "groundedness" in lowered:
            claims = re.findall(r"\[(\d+)\]", prompt)
            return json.dumps(
                {"supported": bool(claims), "score": 0.9 if claims else 0.1, "unsupported": []}
            )
        if "intent" in lowered or "route" in lowered:
            return json.dumps({"intent": "factoid", "confidence": 0.8})
        return json.dumps({"result": _first_sentences(prompt, 1)})

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> Completion:
        text = self._synthesize(messages, json_mode=json_mode)
        prompt_tokens, completion_tokens = self._estimate(messages, text)
        return self._record(
            Completion(
                text=text,
                provider=self.name,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        text = self._synthesize(messages, json_mode=False)
        prompt_tokens, completion_tokens = self._estimate(messages, text)
        for token in text.split(" "):
            yield token + " "
        self._record(
            Completion(
                text=text,
                provider=self.name,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    async def health(self) -> bool:
        return True
