from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import QueryResponse
from secrag.observability.tracing import span
from secrag.retrieval.embedder import Embedder

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = get_logger(__name__)


@dataclass(slots=True)
class CacheEntry:
    question: str
    vector: NDArray[np.float32]
    response: QueryResponse
    created_at: float = field(default_factory=time.time)

    def is_expired(self, ttl_s: int) -> bool:
        return ttl_s > 0 and (time.time() - self.created_at) > ttl_s


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "lookups": self.lookups,
            "hit_rate": round(self.hit_rate, 4),
        }


class SemanticCache:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings)
        self.stats = CacheStats()
        self._partitions: dict[str, OrderedDict[str, CacheEntry]] = {}

    @property
    def enabled(self) -> bool:
        return self.settings.cache_enabled

    @property
    def size(self) -> int:
        return sum(len(p) for p in self._partitions.values())

    def clear(self) -> None:
        self._partitions.clear()
        self.stats = CacheStats()

    def get(self, question: str, partition: str = "") -> QueryResponse | None:
        if not self.enabled:
            return None

        bucket = self._partitions.get(partition)
        if not bucket:
            self.stats.misses += 1
            return None

        with span("cache_lookup", partition=partition or "default"):
            vector = self.embedder.embed_query(question)
            best_key, best_score = None, 0.0

            for key, entry in list(bucket.items()):
                if entry.is_expired(self.settings.cache_ttl_s):
                    del bucket[key]
                    self.stats.expirations += 1
                    continue
                score = float(np.dot(vector, entry.vector))
                if score > best_score:
                    best_key, best_score = key, score

            if best_key is not None and best_score >= self.settings.cache_similarity_threshold:
                entry = bucket[best_key]
                bucket.move_to_end(best_key)
                self.stats.hits += 1
                log.info(
                    "cache_hit",
                    similarity=round(best_score, 4),
                    original=entry.question[:60],
                )
                hit = entry.response.model_copy(deep=True)
                hit.cached = True
                return hit

        self.stats.misses += 1
        return None

    def put(self, question: str, response: QueryResponse, partition: str = "") -> None:
        if not self.enabled:
            return
        bucket = self._partitions.setdefault(partition, OrderedDict())
        vector = self.embedder.embed_query(question)

        stored = response.model_copy(deep=True)
        stored.cached = False
        bucket[question] = CacheEntry(question=question, vector=vector, response=stored)
        bucket.move_to_end(question)

        while len(bucket) > self.settings.cache_max_entries:
            bucket.popitem(last=False)
            self.stats.evictions += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "size": self.size,
            "partitions": len(self._partitions),
            "threshold": self.settings.cache_similarity_threshold,
            **self.stats.to_dict(),
        }


def partition_key(tickers: list[str], fiscal_years: list[int], sections: list[str]) -> str:
    return "|".join(
        (
            ",".join(sorted(t.upper() for t in tickers)),
            ",".join(str(y) for y in sorted(fiscal_years)),
            ",".join(sorted(sections)),
        )
    )
