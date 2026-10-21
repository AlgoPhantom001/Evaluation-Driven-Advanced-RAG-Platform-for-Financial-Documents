from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from secrag import __version__
from secrag.core.config import get_settings
from secrag.core.errors import SecRagError
from secrag.core.logging import configure_logging, get_logger
from secrag.core.types import QueryRequest, QueryResponse
from secrag.engine import QueryEngine
from secrag.generation.chain import SecRagChain
from secrag.observability.metrics import REGISTRY

log = get_logger(__name__)

router = APIRouter()

UI_DIR = Path(__file__).resolve().parents[3] / "ui"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)

    engine = QueryEngine(settings)
    app.state.engine = engine
    app.state.chain: SecRagChain = engine.answer_chain
    app.state.ready = False

    async def warm() -> None:
        try:
            await asyncio.to_thread(engine.warmup)
            app.state.ready = True
            REGISTRY.gauge("secrag_corpus_chunks", engine.retriever.corpus_size)
            log.info("service_ready", version=__version__)
        except Exception as exc:
            log.warning("warmup_failed", error=str(exc))
            app.state.ready = True

    task = asyncio.create_task(warm())
    try:
        yield
    finally:
        with contextlib.suppress(TimeoutError, asyncio.CancelledError, Exception):
            await asyncio.wait_for(task, timeout=60.0)
        await engine.aclose()


def create_app() -> FastAPI:
    application = FastAPI(
        title="sec-rag-platform",
        version=__version__,
        description=(
            "Evaluation-driven retrieval-augmented generation over SEC filings. "
            "Hybrid retrieval, cross-encoder reranking, XBRL-verified figures, "
            "and groundedness verification on every answer."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    application.include_router(router)
    application.add_exception_handler(SecRagError, handle_secrag_error)
    return application


def engine_of(request: Request) -> QueryEngine:
    return request.app.state.engine


async def handle_secrag_error(_: Request, exc: SecRagError) -> JSONResponse:
    REGISTRY.increment("secrag_errors_total", code=exc.code)
    log.warning("request_failed", code=exc.code, message=exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


@router.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/ready", tags=["ops"])
async def ready(request: Request) -> JSONResponse:
    is_ready = bool(getattr(request.app.state, "ready", False))
    engine = engine_of(request)
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "ready": is_ready,
            "corpus_chunks": engine.retriever.corpus_size,
            "bm25_documents": engine.retriever.bm25.size,
        },
    )


@router.get("/v1/stats", tags=["ops"])
async def stats(request: Request) -> dict[str, Any]:
    return engine_of(request).stats()


@router.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
async def metrics() -> str:
    return REGISTRY.render_prometheus()


@router.get("/v1/metrics", tags=["ops"])
async def metrics_json() -> dict[str, object]:
    return REGISTRY.snapshot()


@router.post("/v1/query", response_model=QueryResponse, tags=["query"])
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    engine = engine_of(request)
    REGISTRY.increment("secrag_queries_total")

    # The LangChain Runnable is the API answer path; it delegates to the same
    # engine.answer() as the CLI, so behaviour and caching are unchanged.
    chain: SecRagChain = request.app.state.chain
    response = await chain.ainvoke(payload.model_dump())
    validated = QueryResponse.model_validate(response)

    REGISTRY.observe("secrag_query_latency_ms", validated.latency_ms)
    REGISTRY.increment("secrag_answers_total", status=validated.answer.status.value)
    if validated.cached:
        REGISTRY.increment("secrag_cache_hits_total")
    if validated.route:
        REGISTRY.increment("secrag_routes_total", intent=validated.route.intent.value)
    return validated


@router.post("/v1/query/stream", tags=["query"])
async def query_stream(request: Request, payload: QueryRequest) -> StreamingResponse:
    engine = engine_of(request)
    REGISTRY.increment("secrag_queries_total")
    REGISTRY.increment("secrag_stream_requests_total")

    async def event_source() -> AsyncIterator[str]:
        try:
            async for event in engine.stream(payload):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        except SecRagError as exc:
            yield f"event: error\ndata: {json.dumps(exc.to_dict())}\n\n"
        except Exception as exc:
            log.warning("stream_failed", error=str(exc))
            payload_error = {"code": "stream_error", "message": str(exc)}
            yield f"event: error\ndata: {json.dumps(payload_error)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    path = UI_DIR / "index.html"
    if not path.exists():
        return HTMLResponse(
            "<h1>sec-rag-platform</h1><p>API is running. See <a href='/docs'>/docs</a>.</p>"
        )
    return HTMLResponse(path.read_text(encoding="utf-8"))


app = create_app()
