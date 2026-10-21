from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from secrag.core.config import get_settings
from secrag.core.logging import configure_logging
from secrag.corpus import CORPUS_TICKERS, CORPUS_YEARS

app = typer.Typer(
    name="secrag",
    help="Evaluation-driven RAG over SEC filings.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _setup(verbose: bool = False) -> None:
    settings = get_settings()
    configure_logging("DEBUG" if verbose else settings.log_level, json_output=settings.log_json)
    settings.ensure_dirs()


def _parse_tickers(value: str | None) -> list[str]:
    if not value:
        return list(CORPUS_TICKERS)
    return [t.strip().upper() for t in value.split(",") if t.strip()]


@app.command()
def ingest(
    tickers: Annotated[str | None, typer.Option(help="Comma separated, e.g. AAPL,MSFT")] = None,
    years: Annotated[int, typer.Option(help="Filings per company")] = CORPUS_YEARS,
    rebuild: Annotated[bool, typer.Option(help="Delete the index first")] = False,
    skip_xbrl: Annotated[bool, typer.Option(help="Skip XBRL fact ingestion")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _setup(verbose)
    from secrag.ingest.pipeline import ingest_tickers
    from secrag.ingest.xbrl import build_fact_store
    from secrag.retrieval.store import VectorStore

    settings = get_settings()
    symbols = _parse_tickers(tickers)
    console.print(f"[bold]Ingesting[/bold] {', '.join(symbols)} ({years} filings each)")

    store = VectorStore(settings)
    if rebuild:
        store.reset()
        store = VectorStore(settings)

    report = asyncio.run(ingest_tickers(symbols, years=years, store=store, settings=settings))

    table = Table(title="Ingestion", show_lines=False)
    for column in ("Filing", "Ticker", "FY", "Chunks", "Tables", "Status"):
        table.add_column(column)
    for entry in report.filings:
        table.add_row(
            entry.filing_id,
            entry.ticker,
            str(entry.fiscal_year),
            str(entry.chunks),
            str(entry.tables),
            "[green]ok[/green]" if entry.ok else f"[red]{entry.error}[/red]",
        )
    console.print(table)

    if not skip_xbrl:
        console.print("[bold]Fetching XBRL facts[/bold]")
        facts = asyncio.run(build_fact_store(symbols, settings))
        console.print(f"  {len(facts.df)} annual facts for {', '.join(facts.tickers())}")

    console.print(f"[bold green]{report.summary()}[/bold green]")
    console.print(f"Corpus now holds {store.count()} chunks")


@app.command()
def facts(
    tickers: Annotated[str | None, typer.Option(help="Comma separated, e.g. AAPL,MSFT")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _setup(verbose)
    from secrag.ingest.xbrl import build_fact_store

    settings = get_settings()
    symbols = _parse_tickers(tickers)
    store = asyncio.run(build_fact_store(symbols, settings))

    table = Table(title="XBRL facts")
    for column in ("Ticker", "Company", "Years", "Rows", "Latest revenue"):
        table.add_column(column)
    for ticker in store.tickers():
        years = store.years(ticker)
        latest = store.latest_year(ticker, "revenue")
        revenue = store.value_of(ticker, "revenue", latest) if latest else None
        rows = int((store.df["ticker"] == ticker).sum())
        company = str(store.df[store.df["ticker"] == ticker].iloc[0]["company"])[:28]
        table.add_row(
            ticker,
            company,
            f"{min(years)}-{max(years)}" if years else "-",
            str(rows),
            (
                f"FY{latest}: {revenue.value / 1e9:,.1f}B"
                if revenue and revenue.value is not None
                else "-"
            ),
        )
    console.print(table)
    console.print(f"[bold green]{len(store.df)} annual facts stored[/bold green]")


@app.command()
def stats(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    _setup(verbose)
    from secrag.engine import build_engine

    engine = build_engine()
    console.print_json(json.dumps(engine.stats(), indent=2, default=str))


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="The question to ask")],
    top_k: Annotated[int, typer.Option(help="Passages to retrieve")] = 6,
    companies: Annotated[str | None, typer.Option(help="Restrict to tickers")] = None,
    reranker: Annotated[str, typer.Option(help="cross_encoder, ltr, or none")] = "cross_encoder",
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _setup(verbose)
    from secrag.core.types import QueryRequest
    from secrag.engine import build_engine

    engine = build_engine()
    request = QueryRequest(
        question=question,
        top_k=top_k,
        companies=_parse_tickers(companies) if companies else [],
        reranker=reranker,
        use_cache=not no_cache,
    )
    response = asyncio.run(engine.answer(request))

    console.rule("[bold]Answer[/bold]")
    console.print(response.answer.text)

    if response.numeric_results:
        console.rule("[bold]Verified figures[/bold]")
        for result in response.numeric_results:
            value = f"{result.value:,.2f}" if result.value is not None else "N/A"
            console.print(f"  {result.label}: [cyan]{value}[/cyan] {result.unit}")
            console.print(f"     [dim]{result.formula}[/dim]")

    if response.answer.citations:
        console.rule("[bold]Sources[/bold]")
        for citation in response.answer.citations:
            console.print(
                f"  [{citation.marker}] {citation.label} "
                f"[dim](support {citation.support_score:.2f})[/dim]"
            )

    console.rule("[bold]Diagnostics[/bold]")
    console.print(
        f"  intent       {response.route.intent.value if response.route else 'n/a'} "
        f"(confidence {response.route.confidence if response.route else 0:.2f})"
    )
    console.print(f"  status       {response.answer.status.value}")
    console.print(f"  groundedness {response.answer.groundedness:.3f}")
    console.print(f"  contexts     {len(response.contexts)}")
    console.print(f"  cached       {response.cached}")
    console.print(f"  latency      {response.latency_ms:.0f} ms")


@app.command("train-router")
def train_router(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    _setup(verbose)
    from secrag.routing.router import QueryRouter

    report = QueryRouter().train()
    payload = report.to_dict()

    out = get_settings().project_root / "evals" / "reports" / "router_training.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    console.print_json(json.dumps(payload, indent=2))
    console.print(f"[dim]Report written to {out}[/dim]")


@app.command("train-ltr")
def train_ltr(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    _setup(verbose)
    from secrag.evaluation.ltr_training import train_ltr_model

    report = train_ltr_model()
    console.print_json(json.dumps(report, indent=2, default=str))


@app.command("train-xgb")
def train_xgb(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    _setup(verbose)
    from secrag.evaluation.xgb_training import train_xgb_ltr_model

    report = train_xgb_ltr_model()
    console.print_json(json.dumps(report, indent=2, default=str))


@app.command("eval")
def evaluate(
    report_path: Annotated[Path | None, typer.Option(help="Where to write the JSON report")] = None,
    reranker: Annotated[str, typer.Option()] = "cross_encoder",
    gate: Annotated[bool, typer.Option(help="Exit non-zero if thresholds are missed")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _setup(verbose)
    from secrag.evaluation.runner import run_evaluation

    result = asyncio.run(run_evaluation(reranker=reranker, report_path=report_path))
    result.render(console)
    if gate and not result.passed:
        console.print("[bold red]Quality gate FAILED[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    report_path: Annotated[Path | None, typer.Option()] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _setup(verbose)
    from secrag.evaluation.benchmark import run_benchmark

    asyncio.run(run_benchmark(report_path=report_path, console=console))


@app.command()
def serve(
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    reload: Annotated[bool, typer.Option()] = False,
) -> None:
    _setup()
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "secrag.api.app:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    app()
