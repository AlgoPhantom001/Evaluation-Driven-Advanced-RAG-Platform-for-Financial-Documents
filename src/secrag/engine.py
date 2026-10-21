from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from secrag.analytics.planner import execute_plan, plan_numeric
from secrag.cache import SemanticCache, partition_key
from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import (
    Answer,
    AnswerStatus,
    NumericResult,
    QueryIntent,
    QueryRequest,
    QueryResponse,
    RouteDecision,
    ScoredChunk,
)
from secrag.generation.synthesize import Generator
from secrag.ingest.xbrl import FactStore
from secrag.generation.chain import SecRagChain
from secrag.observability.tracing import Trace, span, start_trace
from secrag.retrieval.embedder import Embedder
from secrag.retrieval.pipeline import HybridRetriever
from secrag.retrieval.store import SearchFilter
from secrag.routing.router import QueryRouter

log = get_logger(__name__)


class QueryEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

        self.embedder = Embedder(self.settings)
        self.retriever = HybridRetriever(self.settings, embedder=self.embedder)
        self.generator = Generator(self.settings, embedder=self.embedder)
        self.router = QueryRouter(self.settings, embedder=self.embedder)
        self.cache = SemanticCache(self.settings, embedder=self.embedder)
        self.facts = FactStore.load(self.settings)

    @property
    def answer_chain(self) -> "SecRagChain":
        """LangChain Runnable over this engine's answer path."""
        from secrag.generation.chain import SecRagChain

        return SecRagChain(engine=self, settings=self.settings)

    def warmup(self) -> None:
        self.retriever.warmup()
        log.info(
            "engine_ready",
            corpus=self.retriever.corpus_size,
            facts=len(self.facts.df),
            router=self.router.is_available,
            tickers=self.facts.tickers(),
        )

    async def aclose(self) -> None:
        closer = getattr(self.generator.provider, "aclose", None)
        if closer is not None:
            await closer()
        self.retriever.store.close()

    def stats(self) -> dict[str, Any]:
        return {
            "corpus_chunks": self.retriever.corpus_size,
            "bm25_documents": self.retriever.bm25.size,
            "xbrl_rows": len(self.facts.df),
            "tickers": self.facts.tickers(),
            "router_available": self.router.is_available,
            "providers": getattr(self.generator.provider, "describe", lambda: [])(),
            "cache": self.cache.snapshot(),
            "settings": {
                "dense_model": self.settings.dense_model,
                "sparse_model": self.settings.sparse_model,
                "rerank_model": self.settings.rerank_model,
                "splade_enabled": self.settings.enable_splade,
                "min_groundedness": self.settings.min_groundedness,
            },
        }

    def _filter(self, request: QueryRequest) -> SearchFilter:
        return SearchFilter(
            tickers=request.companies,
            fiscal_years=request.fiscal_years,
            sections=request.sections,
        )

    def _partition(self, request: QueryRequest) -> str:
        return partition_key(
            request.companies, request.fiscal_years, [s.value for s in request.sections]
        )

    def _numeric(self, question: str, route: RouteDecision) -> list[NumericResult]:
        if route.intent not in (QueryIntent.NUMERIC, QueryIntent.COMPARATIVE):
            return []
        if self.facts.is_empty:
            return []
        with span("numeric_plan", intent=route.intent.value):
            plan = plan_numeric(question, self.facts)
            if plan is None:
                log.info("numeric_plan_unresolved", question=question[:80])
                return []
            return execute_plan(plan, self.facts)

    def _retrieve(
        self, request: QueryRequest, route: RouteDecision
    ) -> tuple[list[ScoredChunk], str]:
        top_k = request.top_k
        if route.intent is QueryIntent.COMPARATIVE:
            top_k = min(top_k * 2, 20)

        reranker = request.reranker if request.use_reranker else "none"
        result = self.retriever.retrieve(
            request.question, top_k=top_k, flt=self._filter(request), reranker=reranker
        )
        return result.chunks, result.reranker

    async def answer(self, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        partition = self._partition(request)

        with start_trace(question=request.question[:120]) as trace:
            if request.use_cache and (hit := self.cache.get(request.question, partition)):
                hit.trace_id = trace.trace_id
                hit.latency_ms = round((time.perf_counter() - started) * 1000, 2)
                return hit

            route = self.router.route(request.question)
            numeric = self._numeric(request.question, route)
            contexts, reranker = self._retrieve(request, route)

            if not contexts and not numeric:
                return self._empty_response(request, route, trace, started)

            generation = await self.generator.generate(request.question, contexts, numeric)

            response = QueryResponse(
                question=request.question,
                answer=generation.answer,
                route=route,
                contexts=generation.contexts,
                numeric_results=numeric,
                trace_id=trace.trace_id,
                cached=False,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            trace.metadata.update(
                {
                    "intent": route.intent.value,
                    "reranker": reranker,
                    "contexts": len(generation.contexts),
                    "numeric_results": len(numeric),
                    "groundedness": generation.answer.groundedness,
                    "status": generation.answer.status.value,
                }
            )

        if request.use_cache and response.answer.status is AnswerStatus.OK:
            self.cache.put(request.question, response, partition)

        log.info(
            "query_answered",
            intent=route.intent.value,
            status=response.answer.status.value,
            groundedness=response.answer.groundedness,
            latency_ms=response.latency_ms,
        )
        return response

    def _empty_response(
        self, request: QueryRequest, route: RouteDecision, trace: Trace, started: float
    ) -> QueryResponse:
        from secrag.generation.prompts import REFUSAL_NO_CONTEXT

        return QueryResponse(
            question=request.question,
            answer=Answer(
                text=REFUSAL_NO_CONTEXT,
                status=AnswerStatus.REFUSED_NO_CONTEXT,
                refusal_reason="Retrieval returned no passages and no figure could be resolved.",
            ),
            route=route,
            trace_id=trace.trace_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def stream(self, request: QueryRequest) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        partition = self._partition(request)

        with start_trace(question=request.question[:120]) as trace:
            if request.use_cache and (hit := self.cache.get(request.question, partition)):
                yield {
                    "event": "meta",
                    "data": {
                        "trace_id": trace.trace_id,
                        "route": hit.route.model_dump(mode="json") if hit.route else None,
                        "reranker": "cached",
                        "contexts": [_context_payload(c) for c in hit.contexts],
                        "numeric_results": [n.model_dump(mode="json") for n in hit.numeric_results],
                        "cached": True,
                    },
                }
                yield {"event": "token", "data": {"text": hit.answer.text}}
                yield {
                    "event": "done",
                    "data": {
                        "status": hit.answer.status.value,
                        "groundedness": hit.answer.groundedness,
                        "refusal_reason": hit.answer.refusal_reason,
                        "citations": [c.model_dump(mode="json") for c in hit.answer.citations],
                        "cached": True,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "trace": trace.to_dict(),
                    },
                }
                return

            route = self.router.route(request.question)
            numeric = self._numeric(request.question, route)
            contexts, reranker = self._retrieve(request, route)

            yield {
                "event": "meta",
                "data": {
                    "trace_id": trace.trace_id,
                    "route": route.model_dump(mode="json"),
                    "reranker": reranker,
                    "contexts": [_context_payload(c) for c in contexts],
                    "numeric_results": [n.model_dump(mode="json") for n in numeric],
                },
            }

            if not contexts and not numeric:
                from secrag.generation.prompts import REFUSAL_NO_CONTEXT

                yield {"event": "token", "data": {"text": REFUSAL_NO_CONTEXT}}
                yield {
                    "event": "done",
                    "data": {
                        "status": AnswerStatus.REFUSED_NO_CONTEXT.value,
                        "groundedness": 0.0,
                        "citations": [],
                        "cached": False,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                        "trace": trace.to_dict(),
                    },
                }
                return

            collected: list[str] = []
            async for piece in self.generator.stream(request.question, contexts, numeric):
                collected.append(piece)
                yield {"event": "token", "data": {"text": piece}}

            final = self.generator.finalise_streamed(
                "".join(collected), contexts, has_verified_figures=bool(numeric)
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)

            if request.use_cache and final.answer.status is AnswerStatus.OK:
                self.cache.put(
                    request.question,
                    QueryResponse(
                        question=request.question,
                        answer=final.answer,
                        route=route,
                        contexts=final.contexts,
                        numeric_results=list(numeric),
                        trace_id=trace.trace_id,
                        latency_ms=latency_ms,
                    ),
                    partition,
                )

            yield {
                "event": "done",
                "data": {
                    "status": final.answer.status.value,
                    "groundedness": final.answer.groundedness,
                    "refusal_reason": final.answer.refusal_reason,
                    "citations": [c.model_dump(mode="json") for c in final.answer.citations],
                    "cached": False,
                    "latency_ms": latency_ms,
                    "trace": trace.to_dict(),
                },
            }


def _context_payload(scored: ScoredChunk) -> dict[str, Any]:
    return {
        "chunk_id": scored.chunk.chunk_id,
        "label": scored.chunk.citation_label(),
        "text": scored.chunk.text,
        "kind": scored.chunk.kind.value,
        "section": scored.chunk.section.value,
        "company": scored.chunk.company,
        "ticker": scored.chunk.ticker,
        "fiscal_year": scored.chunk.fiscal_year,
        "source_url": scored.chunk.source_url,
        "score": round(scored.score, 6),
        "stage": scored.stage,
        "rank": scored.rank,
        "component_scores": {k: round(v, 6) for k, v in scored.component_scores.items()},
    }


def build_engine(settings: Settings | None = None) -> QueryEngine:
    return QueryEngine(settings)
