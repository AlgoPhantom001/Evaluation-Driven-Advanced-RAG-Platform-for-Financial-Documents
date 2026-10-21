from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from secrag.core.logging import get_logger
from secrag.core.types import NumericResult
from secrag.ingest.xbrl import FactStore

log = get_logger(__name__)

Operation = Literal["value", "growth", "cagr", "ratio", "series"]

_YEAR_RE = re.compile(r"\b(?:fy\s?)?((?:19|20)\d{2})\b", re.IGNORECASE)

_METRIC_PHRASES: tuple[tuple[str, str], ...] = (
    ("operating cash flow", "operating_cash_flow"),
    ("cash from operations", "operating_cash_flow"),
    ("research and development", "rnd_expense"),
    ("r and d", "rnd_expense"),
    ("diluted earnings per share", "eps_diluted"),
    ("earnings per share", "eps_diluted"),
    ("diluted shares", "shares_diluted"),
    ("shares outstanding", "shares_diluted"),
    ("stockholders equity", "stockholders_equity"),
    ("shareholders equity", "stockholders_equity"),
    ("total liabilities", "total_liabilities"),
    ("total assets", "total_assets"),
    ("operating income", "operating_income"),
    ("gross profit", "gross_profit"),
    ("net income", "net_income"),
    ("net profit", "net_income"),
    ("revenue", "revenue"),
    ("sales", "revenue"),
    ("cash", "cash"),
    ("assets", "total_assets"),
    ("liabilities", "total_liabilities"),
    ("equity", "stockholders_equity"),
)

_RATIOS: tuple[tuple[str, str, str, str], ...] = (
    ("gross margin", "gross_profit", "revenue", "Gross margin"),
    ("operating margin", "operating_income", "revenue", "Operating margin"),
    ("net margin", "net_income", "revenue", "Net margin"),
    ("profit margin", "net_income", "revenue", "Net margin"),
    ("return on equity", "net_income", "stockholders_equity", "Return on equity"),
    ("roe", "net_income", "stockholders_equity", "Return on equity"),
    ("return on assets", "net_income", "total_assets", "Return on assets"),
    ("roa", "net_income", "total_assets", "Return on assets"),
)

_GROWTH_WORDS = (
    "grow",
    "grew",
    "increase",
    "decrease",
    "change",
    "changed",
    "rise",
    "rose",
    "fall",
    "fell",
    "decline",
    "up from",
    "down from",
)
_CAGR_WORDS = ("cagr", "compound annual", "compounded")
_SERIES_WORDS = ("trend", "over time", "each year", "history", "series", "year by year")


@dataclass(slots=True)
class NumericPlan:
    tickers: list[str]
    metric: str
    operation: Operation
    years: list[int] = field(default_factory=list)
    ratio_numerator: str = ""
    ratio_denominator: str = ""
    label: str = ""

    def describe(self) -> str:
        who = ", ".join(self.tickers)
        span_text = "-".join(str(y) for y in self.years) if self.years else "latest"
        return f"{self.operation}({self.metric}) for {who} over {span_text}"


def _find_tickers(question: str, store: FactStore) -> list[str]:
    known = store.tickers()
    if not known:
        return []
    found: list[str] = []
    upper = question.upper()

    for ticker in known:
        if re.search(rf"\b{re.escape(ticker)}\b", upper):
            found.append(ticker)
            continue
        rows = store.df[store.df["ticker"] == ticker]
        if rows.empty:
            continue
        company = str(rows.iloc[0]["company"])
        lead = re.split(r"[ ,.]", company.strip())[0]
        if len(lead) > 2 and re.search(rf"\b{re.escape(lead.upper())}\b", upper):
            found.append(ticker)

    return list(dict.fromkeys(found))


def _find_metric(lowered: str) -> str | None:
    for phrase, metric in _METRIC_PHRASES:
        if phrase in lowered:
            return metric
    return None


def _find_ratio(lowered: str) -> tuple[str, str, str] | None:
    for phrase, numerator, denominator, label in _RATIOS:
        if phrase in lowered:
            return numerator, denominator, label
    return None


def plan_numeric(question: str, store: FactStore) -> NumericPlan | None:
    if store.is_empty:
        return None

    lowered = question.lower()
    tickers = _find_tickers(question, store)
    if not tickers:
        return None

    years = sorted({int(y) for y in _YEAR_RE.findall(question)})
    available = set(store.years(tickers[0]))
    years = [y for y in years if y in available] or years

    if ratio := _find_ratio(lowered):
        numerator, denominator, label = ratio
        target_year = years[-1] if years else (store.latest_year(tickers[0], "revenue") or 0)
        return NumericPlan(
            tickers=tickers,
            metric=numerator,
            operation="ratio",
            years=[target_year],
            ratio_numerator=numerator,
            ratio_denominator=denominator,
            label=label,
        )

    metric = _find_metric(lowered)
    if metric is None:
        return None

    if any(word in lowered for word in _CAGR_WORDS) and len(years) >= 2:
        operation: Operation = "cagr"
    elif any(word in lowered for word in _GROWTH_WORDS) and len(years) >= 2:
        operation = "growth"
    elif any(word in lowered for word in _SERIES_WORDS):
        operation = "series"
    elif any(word in lowered for word in _GROWTH_WORDS) and len(years) == 1:
        operation = "growth"
        years = [years[0] - 1, years[0]]
    else:
        operation = "value"

    if not years:
        latest = store.latest_year(tickers[0], metric)
        if latest is None:
            return None
        years = [latest]

    return NumericPlan(tickers=tickers, metric=metric, operation=operation, years=years)


def execute_plan(plan: NumericPlan, store: FactStore) -> list[NumericResult]:
    results: list[NumericResult] = []

    for ticker in plan.tickers:
        match plan.operation:
            case "ratio":
                results.append(
                    store.ratio(
                        ticker,
                        plan.ratio_numerator,
                        plan.ratio_denominator,
                        plan.years[-1],
                        plan.label,
                    )
                )
            case "growth":
                results.append(store.growth(ticker, plan.metric, plan.years[0], plan.years[-1]))
            case "cagr":
                results.append(store.cagr(ticker, plan.metric, plan.years[0], plan.years[-1]))
            case "series":
                frame = store.series(ticker, plan.metric)
                for _, row in frame.iterrows():
                    results.append(store.value_of(ticker, plan.metric, int(row["fiscal_year"])))
            case _:
                for year in plan.years:
                    results.append(store.value_of(ticker, plan.metric, year))

    resolved = [r for r in results if r.value is not None]
    log.info(
        "numeric_plan_executed",
        plan=plan.describe(),
        resolved=len(resolved),
        total=len(results),
    )
    return resolved
