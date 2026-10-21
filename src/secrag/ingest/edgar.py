from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import httpx

from secrag.core.config import Settings, get_settings
from secrag.core.errors import EdgarError
from secrag.core.logging import get_logger

log = get_logger(__name__)

SEC_WWW = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"


@dataclass(frozen=True, slots=True)
class FilingRef:
    cik: str
    company: str
    ticker: str | None
    form: str
    fiscal_year: int
    filing_date: str
    report_date: str
    accession: str
    primary_document: str

    @property
    def accession_plain(self) -> str:
        return self.accession.replace("-", "")

    @property
    def document_url(self) -> str:
        return (
            f"{SEC_WWW}/Archives/edgar/data/{int(self.cik)}"
            f"/{self.accession_plain}/{self.primary_document}"
        )

    @property
    def filing_id(self) -> str:
        return f"{self.cik}-{self.form.replace('-', '')}-{self.fiscal_year}"


def _decode(payload: bytes) -> str:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


class RateLimiter:
    def __init__(self, per_second: float) -> None:
        self._min_interval = 1.0 / per_second
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last
            if (wait := self._min_interval - elapsed) > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class EdgarClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cache_dir: Path = self.settings.raw_dir / "edgar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = RateLimiter(self.settings.edgar_rate_limit_per_s)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": self.settings.edgar_user_agent,
                    "Accept-Encoding": "gzip, deflate",
                },
                timeout=httpx.Timeout(60.0, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    def _cache_path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        return self.cache_dir / safe

    async def _get(self, url: str, cache_key: str) -> str:
        path = self._cache_path(cache_key)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")

        await self._limiter.acquire()
        log.debug("edgar_fetch", url=url)
        try:
            response = await self.client.get(url)
        except httpx.HTTPError as exc:
            msg = f"Network error fetching {url}"
            raise EdgarError(msg, detail=str(exc)) from exc

        if response.status_code == 403:
            msg = (
                "EDGAR returned 403. This almost always means the User-Agent is missing "
                "a contact address. Set SECRAG_EDGAR_USER_AGENT."
            )
            raise EdgarError(msg)
        if response.status_code == 404:
            msg = f"EDGAR has no document at {url}"
            raise EdgarError(msg)
        if response.status_code >= 400:
            msg = f"EDGAR returned {response.status_code} for {url}"
            raise EdgarError(msg, detail=response.text[:300])

        text = _decode(response.content)
        path.write_text(text, encoding="utf-8")
        return text

    async def _get_json(self, url: str, cache_key: str) -> dict[str, Any]:
        raw = await self._get(url, cache_key)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._cache_path(cache_key).unlink(missing_ok=True)
            msg = f"EDGAR returned malformed JSON from {url}"
            raise EdgarError(msg) from exc
        if not isinstance(data, dict):
            msg = f"Expected a JSON object from {url}"
            raise EdgarError(msg)
        return data

    async def ticker_map(self) -> dict[str, tuple[str, str]]:
        data = await self._get_json(f"{SEC_WWW}/files/company_tickers.json", "company_tickers.json")
        out: dict[str, tuple[str, str]] = {}
        for entry in data.values():
            if not isinstance(entry, dict):
                continue
            ticker = str(entry.get("ticker", "")).upper()
            cik = str(entry.get("cik_str", "")).zfill(10)
            if ticker and cik.isdigit():
                out[ticker] = (cik, str(entry.get("title", ticker)))
        if not out:
            msg = "EDGAR ticker index was empty"
            raise EdgarError(msg)
        return out

    async def resolve_ticker(self, ticker: str) -> tuple[str, str]:
        mapping = await self.ticker_map()
        key = ticker.strip().upper()
        if key not in mapping:
            msg = f"Ticker {ticker!r} was not found in the EDGAR company index"
            raise EdgarError(msg)
        return mapping[key]

    async def find_filings(
        self, ticker: str, *, form: str = "10-K", years: int = 3
    ) -> list[FilingRef]:
        cik, company = await self.resolve_ticker(ticker)
        data = await self._get_json(
            f"{SEC_DATA}/submissions/CIK{cik}.json", f"submissions_CIK{cik}.json"
        )

        recent = data.get("filings", {}).get("recent", {})
        forms: list[str] = recent.get("form", [])
        accessions: list[str] = recent.get("accessionNumber", [])
        filing_dates: list[str] = recent.get("filingDate", [])
        report_dates: list[str] = recent.get("reportDate", [])
        documents: list[str] = recent.get("primaryDocument", [])

        results: list[FilingRef] = []
        for i, form_type in enumerate(forms):
            if form_type != form:
                continue
            report_date = report_dates[i] if i < len(report_dates) else ""
            filing_date = filing_dates[i] if i < len(filing_dates) else ""
            year_source = report_date or filing_date
            if not year_source:
                continue
            results.append(
                FilingRef(
                    cik=cik,
                    company=str(data.get("name", company)),
                    ticker=ticker.upper(),
                    form=form_type,
                    fiscal_year=int(year_source[:4]),
                    filing_date=filing_date,
                    report_date=report_date,
                    accession=accessions[i],
                    primary_document=documents[i],
                )
            )
            if len(results) >= years:
                break

        if not results:
            msg = f"No {form} filings found for {ticker}"
            raise EdgarError(msg)
        return results

    async def fetch_document(self, ref: FilingRef) -> str:
        return await self._get(ref.document_url, f"{ref.filing_id}_{ref.primary_document}")

    async def company_facts(self, cik: str) -> dict[str, Any]:
        cik = cik.zfill(10)
        return await self._get_json(
            f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik}.json", f"companyfacts_CIK{cik}.json"
        )
