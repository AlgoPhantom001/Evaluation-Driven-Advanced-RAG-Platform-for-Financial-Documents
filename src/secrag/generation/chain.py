from __future__ import annotations

from typing import Any

from langchain_core.runnables import Runnable

from secrag.core.config import Settings, get_settings
from secrag.core.types import QueryRequest, QueryResponse, ScoredChunk


class SecRagChain(Runnable[dict[str, Any], dict[str, Any]]):
    """LangChain ``Runnable`` around the sec-rag answer path.

    Takes ``{"question": ..., "companies": [...], "fiscal_years": [...]}`` and
    returns the ``QueryResponse`` fields as a plain dict (``answer``,
    ``citations``, ``groundedness``, ``numeric_results``, ``route``, ``status``),
    so it drops into any LangChain pipeline / agent that consumes Runnables.
    The chain is thin: retrieval, caching, and verification all still happen
    inside the engine.
    """

    def __init__(self, engine: Any | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._engine = engine
        self._bound_engine: Any = None

    @property
    def engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self._bound_engine is None:
            from secrag.engine import build_engine

            self._bound_engine = build_engine(self.settings)
        return self._bound_engine

    def invoke(self, input: dict[str, Any], config: Any = None) -> dict[str, Any]:
        import asyncio

        async def run() -> dict[str, Any]:
            return await self.ainvoke(input, config)

        if _in_event_loop():
            raise RuntimeError("SecRagChain is async-only; use .ainvoke() instead")
        return asyncio.run(run())

    async def ainvoke(self, input: dict[str, Any], config: Any = None) -> dict[str, Any]:
        request = QueryRequest(
            question=input["question"],
            top_k=int(input.get("top_k", 6)),
            companies=[str(t).upper() for t in input.get("companies", [])],
            fiscal_years=list(input.get("fiscal_years", [])),
            reranker=input.get("reranker", "cross_encoder"),
            use_cache=bool(input.get("use_cache", True)),
        )
        response = await self.engine.answer(request)
        # JSON mode so callers can round-trip the result through
        # QueryResponse.model_validate(...) after a dict serialization hop.
        return _to_dict(response)

    def with_engine(self, engine: Any) -> SecRagChain:
        return SecRagChain(engine=engine, settings=self.settings)


def _in_event_loop() -> bool:
    import asyncio

    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _to_dict(response: QueryResponse) -> dict[str, Any]:
    answer = response.answer
    return {
        "question": response.question,
        "answer": {
            "text": answer.text,
            "status": answer.status.value,
            "citations": [c.model_dump(mode="json") for c in answer.citations],
            "groundedness": answer.groundedness,
            "refusal_reason": answer.refusal_reason,
        },
        "route": response.route.model_dump(mode="json") if response.route else None,
        "contexts": [_context_view(c) for c in response.contexts],
        "numeric_results": [n.model_dump(mode="json") for n in response.numeric_results],
        "trace_id": response.trace_id,
        "cached": response.cached,
        "latency_ms": response.latency_ms,
    }


def _context_view(scored: ScoredChunk) -> dict[str, Any]:
    return {
        "chunk_id": scored.chunk.chunk_id,
        "label": scored.chunk.citation_label(),
        "score": round(scored.score, 6),
        "rank": scored.rank,
        "source_url": scored.chunk.source_url,
    }


def build_chain(engine: Any | None = None, settings: Settings | None = None) -> SecRagChain:
    return SecRagChain(engine=engine, settings=settings)


__all__ = ["SecRagChain", "build_chain"]
