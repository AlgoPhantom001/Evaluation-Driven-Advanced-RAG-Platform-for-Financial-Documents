from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import ScoredChunk
from secrag.observability.tracing import span
from secrag.retrieval.bm25 import BM25Index
from secrag.retrieval.embedder import Embedder
from secrag.retrieval.fusion import reciprocal_rank_fusion
from secrag.retrieval.rerank import (
    CrossEncoderReranker,
    LTRReranker,
    Reranker,
    build_reranker,
)
from secrag.retrieval.store import SearchFilter, VectorStore

log = get_logger(__name__)

ALL_ARMS = ("dense", "bm25", "splade")


@dataclass(slots=True)
class RetrievalResult:
    chunks: list[ScoredChunk]
    arms: dict[str, list[ScoredChunk]] = field(default_factory=dict)
    fused: list[ScoredChunk] = field(default_factory=list)
    reranker: str = "none"

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class HybridRetriever:
    def __init__(
        self,
        settings: Settings | None = None,
        store: VectorStore | None = None,
        bm25: BM25Index | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings)
        self.store = store or VectorStore(self.settings, self.embedder)
        self.bm25 = bm25 or BM25Index(self.settings)
        self._rerankers: dict[str, Reranker] = {}
        self._lock = threading.Lock()

    def ensure_ready(self, *, force: bool = False) -> int:
        if self.bm25.is_ready and not force:
            return self.bm25.size
        with self._lock:
            if self.bm25.is_ready and not force:
                return self.bm25.size
            return self.bm25.build(self.store.iter_chunks())

    def reranker(self, name: str) -> Reranker:
        if name not in self._rerankers:
            with self._lock:
                if name not in self._rerankers:
                    self._rerankers[name] = build_reranker(name, self.settings)
        return self._rerankers[name]

    @property
    def corpus_size(self) -> int:
        return self.store.count()

    def run_arms(
        self,
        query: str,
        flt: SearchFilter | None = None,
        arms: Sequence[str] = ALL_ARMS,
    ) -> dict[str, list[ScoredChunk]]:
        results: dict[str, list[ScoredChunk]] = {}

        if "dense" in arms:
            vector = self.embedder.embed_query(query)
            results["dense"] = self.store.search_dense(vector, self.settings.dense_top_k, flt)

        if "bm25" in arms:
            self.ensure_ready()
            results["bm25"] = self.bm25.search(query, self.settings.sparse_top_k, flt)

        if "splade" in arms and self.settings.enable_splade:
            sparse = self.embedder.embed_sparse_query(query)
            results["splade"] = self.store.search_sparse(sparse, self.settings.splade_top_k, flt)

        return {arm: hits for arm, hits in results.items() if hits}

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        flt: SearchFilter | None = None,
        arms: Sequence[str] = ALL_ARMS,
        reranker: str = "cross_encoder",
        weights: dict[str, float] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or self.settings.rerank_top_n

        with span("retrieve", query_chars=len(query), arms=list(arms), reranker=reranker):
            arm_results = self.run_arms(query, flt, arms)
            if not arm_results:
                log.info("retrieval_empty", query=query[:80])
                return RetrievalResult(chunks=[], arms={}, fused=[], reranker=reranker)

            if len(arm_results) == 1:
                fused = list(next(iter(arm_results.values())))[: self.settings.rerank_candidates]
            else:
                fused = reciprocal_rank_fusion(
                    arm_results,
                    k=self.settings.rrf_k,
                    weights=weights,
                    limit=self.settings.rerank_candidates,
                )

            engine = self.reranker(reranker)
            ranked = engine.rerank(query, fused, top_k)

        log.info(
            "retrieved",
            arms={arm: len(hits) for arm, hits in arm_results.items()},
            fused=len(fused),
            returned=len(ranked),
            reranker=engine.name,
        )
        return RetrievalResult(chunks=ranked, arms=arm_results, fused=fused, reranker=engine.name)

    def warmup(self) -> None:
        self.embedder.warmup()
        self.ensure_ready()

        for name in ("cross_encoder", "ltr", "xgb"):
            engine = self.reranker(name)
            if isinstance(engine, CrossEncoderReranker):
                engine.warmup()
            elif isinstance(engine, LTRReranker):
                _ = engine.booster
            elif engine.name == "xgb":
                _ = engine.model

        log.info(
            "retriever_warm",
            corpus=self.corpus_size,
            bm25=self.bm25.size,
            rerankers=sorted(self._rerankers),
        )
