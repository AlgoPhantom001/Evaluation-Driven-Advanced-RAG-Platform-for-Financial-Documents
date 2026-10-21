from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import Chunk, ScoredChunk
from secrag.observability.tracing import span
from secrag.retrieval.store import SearchFilter

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "we",
        "our",
        "us",
        "this",
        "these",
        "those",
        "which",
        "their",
        "they",
        "than",
        "then",
        "such",
        "may",
        "can",
        "could",
        "would",
        "should",
    }
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class BM25Index:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._bm25: Any = None
        self._chunks: list[Chunk] = []

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None and bool(self._chunks)

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: Iterable[Chunk]) -> int:
        from rank_bm25 import BM25Okapi

        with span("bm25_build"):
            self._chunks = list(chunks)
            if not self._chunks:
                self._bm25 = None
                return 0
            corpus = [tokenize(c.contextual_text()) for c in self._chunks]
            self._bm25 = BM25Okapi(corpus)
        log.info("bm25_built", documents=len(self._chunks))
        return len(self._chunks)

    def search(self, query: str, limit: int, flt: SearchFilter | None = None) -> list[ScoredChunk]:
        if not self.is_ready:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        with span("search_sparse_bm25", limit=limit):
            scores = self._bm25.get_scores(tokens)

            candidates = [
                (float(score), chunk)
                for score, chunk in zip(scores, self._chunks, strict=True)
                if score > 0.0 and (flt is None or flt.matches(chunk))
            ]
            candidates.sort(key=lambda pair: pair[0], reverse=True)
            top = candidates[:limit]

        return [
            ScoredChunk(
                chunk=chunk,
                score=score,
                stage="bm25",
                rank=rank,
                component_scores={"bm25": score},
            )
            for rank, (score, chunk) in enumerate(top, start=1)
        ]
