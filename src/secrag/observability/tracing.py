from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "echo-1": (0.0, 0.0),
}
_DEFAULT_PRICE = (0.0, 0.0)


@dataclass(slots=True)
class Span:
    name: str
    start_ms: float
    end_ms: float | None = None
    depth: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return (self.end_ms if self.end_ms is not None else _now_ms()) - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "depth": self.depth,
            "attributes": self.attributes,
            "error": self.error,
        }


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.calls,
        }


@dataclass(slots=True)
class Trace:
    trace_id: str
    started_ms: float
    spans: list[Span] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    ended_ms: float | None = None

    @property
    def duration_ms(self) -> float:
        return (self.ended_ms if self.ended_ms is not None else _now_ms()) - self.started_ms

    def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage.add(prompt_tokens, completion_tokens)
        in_price, out_price = PRICE_PER_MTOK.get(model, _DEFAULT_PRICE)
        self.cost_usd += (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000

    def stage_timings(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for span in self.spans:
            if span.depth == 0:
                out[span.name] = round(out.get(span.name, 0.0) + span.duration_ms, 2)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "duration_ms": round(self.duration_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
            "stage_timings_ms": self.stage_timings(),
            "usage": self.usage.to_dict(),
            "estimated_cost_usd": round(self.cost_usd, 8),
            "metadata": self.metadata,
        }


_current_trace: ContextVar[Trace | None] = ContextVar("secrag_current_trace", default=None)
_current_depth: ContextVar[int] = ContextVar("secrag_current_depth", default=0)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def current_trace() -> Trace | None:
    return _current_trace.get()


def current_trace_id() -> str | None:
    trace = _current_trace.get()
    return trace.trace_id if trace else None


@contextmanager
def start_trace(trace_id: str | None = None, **metadata: Any) -> Iterator[Trace]:
    trace = Trace(
        trace_id=trace_id or uuid.uuid4().hex[:16],
        started_ms=_now_ms(),
        metadata=dict(metadata),
    )
    token: Token[Trace | None] = _current_trace.set(trace)
    depth_token: Token[int] = _current_depth.set(0)
    try:
        yield trace
    finally:
        trace.ended_ms = _now_ms()
        _current_trace.reset(token)
        _current_depth.reset(depth_token)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    trace = _current_trace.get()
    depth = _current_depth.get()
    current = Span(name=name, start_ms=_now_ms(), depth=depth, attributes=dict(attributes))
    depth_token: Token[int] = _current_depth.set(depth + 1)
    if trace is not None:
        trace.spans.append(current)
    try:
        yield current
    except Exception as exc:
        current.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        current.end_ms = _now_ms()
        _current_depth.reset(depth_token)


def record_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    if (trace := _current_trace.get()) is not None:
        trace.record_usage(model, prompt_tokens, completion_tokens)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
