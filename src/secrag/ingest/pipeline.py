from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from secrag.core.config import Settings, get_settings
from secrag.core.errors import EdgarError, IngestionError
from secrag.core.logging import get_logger
from secrag.core.types import Chunk, ChunkKind, Filing
from secrag.ingest.chunker import chunk_blocks
from secrag.ingest.edgar import EdgarClient, FilingRef
from secrag.ingest.parser import parse_filing
from secrag.observability.tracing import span

# On CPU-only deployments the SPLADE (prithivida/Splade_PP_en_v1) ONNX graph
# can request multi-GB activations when ONNX Runtime oversubscribes threads
# (observed as a 4 GB BFCArena allocation failure). Cap thread pools up front
# so the embedding pass stays within a single-core budget. Only set if the
# user has not already chosen a value.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ONNXRUNTIME_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
from secrag.retrieval.store import VectorStore

log = get_logger(__name__)


@dataclass(slots=True)
class FilingReport:
    filing_id: str
    ticker: str
    fiscal_year: int
    chunks: int = 0
    tables: int = 0
    sections: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class IngestReport:
    filings: list[FilingReport] = field(default_factory=list)
    total_chunks: int = 0

    @property
    def succeeded(self) -> list[FilingReport]:
        return [f for f in self.filings if f.ok]

    @property
    def failed(self) -> list[FilingReport]:
        return [f for f in self.filings if not f.ok]

    def summary(self) -> str:
        return (
            f"{len(self.succeeded)} filings indexed, {self.total_chunks} chunks, "
            f"{len(self.failed)} failed"
        )


def _to_filing(ref: FilingRef) -> Filing:
    filing_date: date | None = None
    if ref.filing_date:
        try:
            filing_date = date.fromisoformat(ref.filing_date)
        except ValueError:
            filing_date = None
    return Filing(
        filing_id=ref.filing_id,
        cik=ref.cik,
        ticker=ref.ticker,
        company=ref.company,
        form_type=ref.form,
        fiscal_year=ref.fiscal_year,
        filing_date=filing_date,
        source_url=ref.document_url,
    )


async def chunks_for_filing(
    client: EdgarClient, ref: FilingRef, settings: Settings
) -> tuple[Filing, list[Chunk]]:
    filing = _to_filing(ref)
    with span("ingest_filing", filing=filing.filing_id):
        html = await client.fetch_document(ref)
        blocks = parse_filing(html)
        chunks = chunk_blocks(blocks, filing, settings)
    if not chunks:
        msg = f"Filing {filing.filing_id} produced no chunks"
        raise IngestionError(msg)
    return filing, chunks


async def ingest_tickers(
    tickers: Sequence[str],
    *,
    years: int = 2,
    store: VectorStore | None = None,
    settings: Settings | None = None,
    form: str = "10-K",
) -> IngestReport:
    settings = settings or get_settings()
    settings.ensure_dirs()
    store = store or VectorStore(settings)
    report = IngestReport()

    async with EdgarClient(settings) as client:
        refs: list[FilingRef] = []
        for ticker in tickers:
            try:
                found = await client.find_filings(ticker, form=form, years=years)
                refs.extend(found)
                log.info("filings_found", ticker=ticker, count=len(found))
            except EdgarError as exc:
                report.filings.append(
                    FilingReport(
                        filing_id=f"?-{ticker}", ticker=ticker, fiscal_year=0, error=exc.message
                    )
                )
                log.warning("ticker_failed", ticker=ticker, error=exc.message)

        async def process(ref: FilingRef) -> tuple[FilingRef, list[Chunk] | None, str | None]:
            try:
                _, chunks = await chunks_for_filing(client, ref, settings)
            except (IngestionError, EdgarError) as exc:
                return ref, None, exc.message
            except Exception as exc:
                return ref, None, f"{type(exc).__name__}: {exc}"
            return ref, chunks, None

        outcomes = await asyncio.gather(*(process(ref) for ref in refs))

    for ref, chunks, error in outcomes:
        entry = FilingReport(
            filing_id=ref.filing_id,
            ticker=ref.ticker or "?",
            fiscal_year=ref.fiscal_year,
            error=error,
        )
        if chunks:
            store.upsert(chunks)
            entry.chunks = len(chunks)
            entry.tables = sum(1 for c in chunks if c.kind is ChunkKind.TABLE)
            for chunk in chunks:
                entry.sections[chunk.section.value] = entry.sections.get(chunk.section.value, 0) + 1
            report.total_chunks += len(chunks)
        report.filings.append(entry)

    log.info("ingest_complete", summary=report.summary())
    return report
