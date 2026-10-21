from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.engine import QueryEngine, build_engine
from secrag.evaluation import metrics
from secrag.evaluation.goldens import load_goldens
from secrag.retrieval.store import SearchFilter

if TYPE_CHECKING:
    from rich.console import Console

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Configuration:
    name: str
    arms: tuple[str, ...]
    reranker: str
    description: str


CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration("dense only", ("dense",), "none", "BGE bi-encoder, no fusion"),
    Configuration("bm25 only", ("bm25",), "none", "Okapi BM25 lexical baseline"),
    Configuration("splade only", ("splade",), "none", "Learned sparse expansion"),
    Configuration("dense + bm25", ("dense", "bm25"), "none", "Two-arm RRF"),
    Configuration("dense + bm25 + splade", ("dense", "bm25", "splade"), "none", "Three-arm RRF"),
    Configuration("hybrid + LTR", ("dense", "bm25", "splade"), "ltr", "LambdaMART over features"),
    Configuration(
        "hybrid + XGB",
        ("dense", "bm25", "splade"),
        "xgb",
        "XGBoost rank:ndcg over the same features",
    ),
    Configuration(
        "hybrid + cross-encoder",
        ("dense", "bm25", "splade"),
        "cross_encoder",
        "MiniLM cross-encoder rerank",
    ),
)


def resolve_arms(arms: Sequence[str], *, enable_splade: bool) -> tuple[str, ...]:
    return tuple(a for a in arms if a != "splade" or enable_splade)


@dataclass(slots=True)
class ConfigurationResult:
    name: str
    description: str
    arms: list[str] = field(default_factory=list)
    reranker: str = "none"
    hit_rate: float = 0.0
    ndcg: float = 0.0
    mrr: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    n_queries: int = 0
    available: bool = True
    in_sample: bool = False
    note: str = ""


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


async def run_benchmark(
    *,
    settings: Settings | None = None,
    engine: QueryEngine | None = None,
    report_path: Path | None = None,
    console: Console | None = None,
) -> list[ConfigurationResult]:
    settings = settings or get_settings()
    engine = engine or build_engine(settings)
    engine.retriever.ensure_ready()

    cases = load_goldens(settings=settings)
    k = settings.eval_k
    results: list[ConfigurationResult] = []

    for config in CONFIGURATIONS:
        arms = resolve_arms(config.arms, enable_splade=settings.enable_splade)

        if not arms:
            results.append(
                ConfigurationResult(
                    name=config.name,
                    description=config.description,
                    arms=list(config.arms),
                    reranker=config.reranker,
                    available=False,
                    note="SPLADE disabled in this deployment",
                )
            )
            continue

        reduced = arms != config.arms

        if config.reranker == "ltr" and not engine.retriever.reranker("ltr").is_available:
            results.append(
                ConfigurationResult(
                    name=config.name,
                    description=config.description,
                    arms=list(config.arms),
                    reranker=config.reranker,
                    available=False,
                    note="LTR model not trained. Run: secrag train-ltr",
                )
            )
            continue

        if config.reranker == "xgb" and not engine.retriever.reranker("xgb").is_available:
            results.append(
                ConfigurationResult(
                    name=config.name,
                    description=config.description,
                    arms=list(config.arms),
                    reranker=config.reranker,
                    available=False,
                    note="XGBoost ranker not trained. Run: secrag train-xgb",
                )
            )
            continue

        accumulator = metrics.MetricAccumulator()
        latencies: list[float] = []

        for case in cases:
            flt = SearchFilter(tickers=case.companies, fiscal_years=case.fiscal_years)
            started = time.perf_counter()
            retrieved = engine.retriever.retrieve(
                case.question,
                top_k=k,
                flt=flt,
                arms=arms,
                reranker=config.reranker,
            )
            latencies.append((time.perf_counter() - started) * 1000.0)

            relevance = metrics.relevance_vector(
                retrieved.chunks, case.expected_sections, case.expected_terms
            )
            accumulator.add("hit_rate", metrics.hit_rate_at_k(relevance, k))
            accumulator.add("ndcg", metrics.ndcg_at_k(relevance, k))
            accumulator.add("mrr", metrics.reciprocal_rank(relevance))
            accumulator.add("precision", metrics.precision_at_k(relevance, k))
            # Recall over the retrieved top-k window (the denominator is the
            # number of relevant passages that made it into the window), so it
            # is well-defined without a corpus-wide relevance count.
            accumulator.add("recall", metrics.recall_at_k(relevance, k))

        results.append(
            ConfigurationResult(
                name=config.name,
                description=config.description,
                arms=list(arms),
                reranker=config.reranker,
                note="ran without the SPLADE arm" if reduced else "",
                in_sample=config.reranker in ("ltr", "xgb"),
                hit_rate=round(accumulator.mean("hit_rate"), 4),
                ndcg=round(accumulator.mean("ndcg"), 4),
                mrr=round(accumulator.mean("mrr"), 4),
                precision=round(accumulator.mean("precision"), 4),
                recall=round(accumulator.mean("recall"), 4),
                mean_latency_ms=round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                p95_latency_ms=round(_percentile(latencies, 95), 2),
                n_queries=len(cases),
            )
        )
        log.info("benchmark_config_done", config=config.name, ndcg=results[-1].ndcg)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_chunks": engine.retriever.corpus_size,
        "n_queries": len(cases),
        "eval_k": k,
        "models": {
            "dense": settings.dense_model,
            "sparse": settings.sparse_model,
            "reranker": settings.rerank_model,
        },
        "configurations": [asdict(r) for r in results],
        "markdown": to_markdown(results, k),
    }

    path = report_path or (settings.project_root / "evals" / "reports" / "benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if console is not None:
        _render(results, console, k)
        console.print(f"[dim]Report written to {path}[/dim]")

    return results


def to_markdown(results: Sequence[ConfigurationResult], k: int) -> str:
    lines = [
        f"| Configuration | nDCG@{k} | Hit@{k} | MRR | P@{k} | Recall@{k} | Mean ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    has_in_sample = False
    for result in results:
        if not result.available:
            lines.append(f"| {result.name} | _{result.note}_ | | | | | | |")
            continue
        label = result.name + (" \\*" if result.in_sample else "")
        if result.note == "ran without the SPLADE arm":
            label += " \\*\\*"
        has_in_sample = has_in_sample or result.in_sample
        lines.append(
            f"| {label} | {result.ndcg:.3f} | {result.hit_rate:.3f} | "
            f"{result.mrr:.3f} | {result.precision:.3f} | {result.recall:.3f} | "
            f"{result.mean_latency_ms:.0f} | {result.p95_latency_ms:.0f} |"
        )

    if any(r.note == "ran without the SPLADE arm" for r in results if r.available):
        lines.append("")
        lines.append(
            "\\*\\* Ran with the SPLADE arm disabled, so this row reflects dense plus BM25 only."
        )

    if has_in_sample:
        lines.append("")
        lines.append(
            "\\* Trained on these same queries, so this row is training-set "
            "performance and is optimistic. Its honest grouped cross-validation "
            "score is reported separately below."
        )
    return "\n".join(lines)


def _render(results: Sequence[ConfigurationResult], console: Console, k: int) -> None:
    from rich.table import Table

    table = Table(title=f"Retrieval ablation (n={results[0].n_queries if results else 0} queries)")
    table.add_column("Configuration")
    for column in (f"nDCG@{k}", f"Hit@{k}", "MRR", f"P@{k}", f"Recall@{k}", "mean ms", "p95 ms"):
        table.add_column(column, justify="right")

    best_ndcg = max((r.ndcg for r in results if r.available and not r.in_sample), default=0.0)
    for result in results:
        if not result.available:
            table.add_row(result.name, f"[dim]{result.note}[/dim]", "", "", "", "", "", "")
            continue
        highlight = "[bold green]" if result.ndcg == best_ndcg else ""
        suffix = " [yellow](in-sample)[/yellow]" if result.in_sample else ""
        if result.note == "ran without the SPLADE arm":
            suffix += " [dim](no splade)[/dim]"
        closing = "[/bold green]" if highlight else ""
        table.add_row(
            result.name + suffix,
            f"{highlight}{result.ndcg:.3f}{closing}",
            f"{result.hit_rate:.3f}",
            f"{result.mrr:.3f}",
            f"{result.precision:.3f}",
            f"{result.recall:.3f}",
            f"{result.mean_latency_ms:.0f}",
            f"{result.p95_latency_ms:.0f}",
        )
    console.print(table)


def _summary_payload(results: Sequence[ConfigurationResult]) -> dict[str, Any]:
    return {r.name: {"ndcg": r.ndcg, "mrr": r.mrr} for r in results if r.available}
