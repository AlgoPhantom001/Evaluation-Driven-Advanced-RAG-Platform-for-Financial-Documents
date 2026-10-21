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
from secrag.core.types import AnswerStatus, QueryRequest
from secrag.engine import QueryEngine, build_engine
from secrag.evaluation import metrics
from secrag.evaluation.goldens import GoldenCase, load_goldens
from secrag.observability.tracing import span

if TYPE_CHECKING:
    from rich.console import Console

log = get_logger(__name__)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "hit_rate@6": 0.80,
    "ndcg@6": 0.55,
    "mrr": 0.60,
    "citation_validity": 0.95,
    "groundedness": 0.40,
    "numeric_accuracy": 0.90,
    "routing_accuracy": 0.80,
    "refusal_correctness": 0.90,
}


def thresholds_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.project_root / "evals" / "thresholds.json"


def load_thresholds(settings: Settings | None = None) -> dict[str, float]:
    path = thresholds_path(settings)
    if not path.exists():
        return dict(DEFAULT_THRESHOLDS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides = {k: float(v) for k, v in payload.items() if not k.startswith("_")}
    return {**DEFAULT_THRESHOLDS, **overrides}


@dataclass(slots=True)
class CaseResult:
    id: str
    question: str
    intent: str
    predicted_intent: str
    status: str
    scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    n_contexts: int = 0
    error: str | None = None

    @property
    def routed_correctly(self) -> bool:
        return self.intent == self.predicted_intent


@dataclass(slots=True)
class EvaluationResult:
    cases: list[CaseResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    reranker: str = "cross_encoder"
    corpus_chunks: int = 0
    duration_s: float = 0.0
    provider: str = ""

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "reranker": self.reranker,
            "provider": self.provider,
            "corpus_chunks": self.corpus_chunks,
            "duration_s": round(self.duration_s, 2),
            "n_cases": len(self.cases),
            "summary": self.summary,
            "thresholds": self.thresholds,
            "passed": self.passed,
            "failures": self.failures,
            "cases": [asdict(c) for c in self.cases],
        }

    def render(self, console: Console) -> None:
        from rich.table import Table

        table = Table(title=f"Evaluation ({len(self.cases)} cases, reranker={self.reranker})")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Status", justify="center")

        for name in sorted(set(self.summary) | set(self.thresholds)):
            value = self.summary.get(name)
            threshold = self.thresholds.get(name)
            if value is None:
                continue
            if threshold is None:
                verdict = "[dim]-[/dim]"
            elif value >= threshold:
                verdict = "[green]PASS[/green]"
            else:
                verdict = "[red]FAIL[/red]"
            table.add_row(
                name,
                f"{value:.4f}",
                f"{threshold:.4f}" if threshold is not None else "-",
                verdict,
            )
        console.print(table)

        if failed := [c for c in self.cases if c.error]:
            console.print(f"[red]{len(failed)} cases errored[/red]")
            for case in failed[:5]:
                console.print(f"  [red]{case.id}[/red]: {case.error}")

        if self.failures:
            console.print("[bold red]Thresholds not met:[/bold red]")
            for failure in self.failures:
                console.print(f"  [red]{failure}[/red]")
        else:
            console.print("[bold green]All thresholds met[/bold green]")


async def evaluate_case(
    engine: QueryEngine, case: GoldenCase, *, reranker: str, k: int
) -> CaseResult:
    request = QueryRequest(
        question=case.question,
        top_k=max(k, 6),
        companies=case.companies,
        fiscal_years=case.fiscal_years,
        reranker=reranker,
        use_cache=False,
    )

    try:
        with span("eval_case", case=case.id):
            response = await engine.answer(request)
    except Exception as exc:
        log.warning("eval_case_failed", case=case.id, error=str(exc))
        return CaseResult(
            id=case.id,
            question=case.question,
            intent=case.intent,
            predicted_intent="",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )

    relevance = metrics.relevance_vector(
        response.contexts, case.expected_sections, case.expected_terms
    )
    answer = response.answer
    scores: dict[str, float] = {
        f"hit_rate@{k}": metrics.hit_rate_at_k(relevance, k),
        f"precision@{k}": metrics.precision_at_k(relevance, k),
        f"recall@{k}": metrics.recall_at_k(relevance, k),
        f"ndcg@{k}": metrics.ndcg_at_k(relevance, k),
        "mrr": metrics.reciprocal_rank(relevance),
        "citation_validity": metrics.citation_validity(answer.text, len(response.contexts)),
        "citation_density": metrics.citation_density(answer.text),
        "groundedness": answer.groundedness,
    }

    if case.must_include:
        scores["answer_coverage"] = metrics.answer_contains(answer.text, case.must_include)

    if case.is_numeric:
        computed = next((r.value for r in response.numeric_results if r.value is not None), None)
        accuracy = metrics.numeric_accuracy(computed, case.expected_value, case.tolerance_pct)
        if accuracy is not None:
            scores["numeric_accuracy"] = accuracy

    refused = answer.status is not AnswerStatus.OK
    scores["refusal_correctness"] = float(refused == case.expect_refusal)

    predicted = response.route.intent.value if response.route else ""
    scores["routing_accuracy"] = float(predicted == case.intent)

    return CaseResult(
        id=case.id,
        question=case.question,
        intent=case.intent,
        predicted_intent=predicted,
        status=answer.status.value,
        scores={k_: round(v, 4) for k_, v in scores.items()},
        latency_ms=response.latency_ms,
        n_contexts=len(response.contexts),
    )


async def run_evaluation(
    *,
    reranker: str = "cross_encoder",
    report_path: Path | None = None,
    settings: Settings | None = None,
    cases: Sequence[GoldenCase] | None = None,
    engine: QueryEngine | None = None,
) -> EvaluationResult:
    settings = settings or get_settings()
    started = time.perf_counter()

    golden_cases = list(cases) if cases is not None else load_goldens(settings=settings)
    engine = engine or build_engine(settings)
    engine.retriever.ensure_ready()

    k = settings.eval_k
    accumulator = metrics.MetricAccumulator()
    results: list[CaseResult] = []

    for case in golden_cases:
        case_result = await evaluate_case(engine, case, reranker=reranker, k=k)
        results.append(case_result)
        for name, value in case_result.scores.items():
            accumulator.add(name, value)

    summary = accumulator.summary()
    thresholds = load_thresholds(settings)

    failures = [
        f"{name}: {summary[name]:.4f} < {threshold:.4f}"
        for name, threshold in thresholds.items()
        if name in summary and summary[name] < threshold
    ]
    if errored := [r for r in results if r.error]:
        failures.append(f"{len(errored)} cases raised an exception")

    result = EvaluationResult(
        cases=results,
        summary=summary,
        thresholds=thresholds,
        failures=failures,
        reranker=reranker,
        corpus_chunks=engine.retriever.corpus_size,
        duration_s=time.perf_counter() - started,
        provider=",".join(getattr(engine.generator.provider, "describe", lambda: [])()),
    )

    path = report_path or (settings.project_root / "evals" / "reports" / "latest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    log.info(
        "evaluation_complete",
        cases=len(results),
        passed=result.passed,
        report=str(path),
        duration_s=round(result.duration_s, 2),
    )
    return result
