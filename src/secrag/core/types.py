from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class Mutable(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class FilingSection(StrEnum):
    BUSINESS = "item_1_business"
    RISK_FACTORS = "item_1a_risk_factors"
    LEGAL_PROCEEDINGS = "item_3_legal_proceedings"
    MDA = "item_7_mda"
    MARKET_RISK = "item_7a_market_risk"
    FINANCIAL_STATEMENTS = "item_8_financial_statements"
    CONTROLS = "item_9a_controls"
    OTHER = "other"


class ChunkKind(StrEnum):
    PROSE = "prose"
    TABLE = "table"


class Filing(Frozen):
    filing_id: str = Field(description="Stable id, formatted as {cik}-{form}-{fy}")
    cik: str = Field(description="Zero-padded 10 digit Central Index Key")
    ticker: str | None = None
    company: str
    form_type: str = Field(default="10-K")
    fiscal_year: int
    filing_date: date | None = None
    source_url: str

    @model_validator(mode="after")
    def _check_cik(self) -> Self:
        if not self.cik.isdigit():
            msg = f"cik must be numeric, got {self.cik!r}"
            raise ValueError(msg)
        return self


class Chunk(Frozen):
    chunk_id: str
    filing_id: str
    text: str
    kind: ChunkKind = ChunkKind.PROSE
    section: FilingSection = FilingSection.OTHER

    company: str
    ticker: str | None = None
    fiscal_year: int
    source_url: str

    ordinal: int = Field(default=0, description="Position within its section")
    char_start: int = 0
    char_end: int = 0
    token_estimate: int = 0

    def citation_label(self) -> str:
        who = self.ticker or self.company
        section = self.section.value.replace("_", " ").title()
        return f"{who} FY{self.fiscal_year} {section}"

    def contextual_text(self) -> str:
        kind = "Table" if self.kind is ChunkKind.TABLE else "Excerpt"
        return f"{self.citation_label()} | {kind}\n{self.text}"


class ScoredChunk(Frozen):
    chunk: Chunk
    score: float
    stage: str = Field(description="dense, sparse, splade, fused, cross_encoder, or ltr")
    rank: int = 0

    component_scores: dict[str, float] = Field(default_factory=dict)


class QueryIntent(StrEnum):
    FACTOID = "factoid"
    NUMERIC = "numeric"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"


class RouteDecision(Frozen):
    intent: QueryIntent
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    fell_back: bool = Field(
        default=False,
        description="True when confidence was below threshold and we defaulted to factoid",
    )


class Citation(Frozen):
    marker: int = Field(ge=1, description="The n in [n] as it appears in the answer")
    chunk_id: str
    label: str
    source_url: str
    quote: str = Field(default="", description="Supporting span from the source chunk")
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)


class AnswerStatus(StrEnum):
    OK = "ok"
    REFUSED_UNGROUNDED = "refused_ungrounded"
    REFUSED_NO_CONTEXT = "refused_no_context"
    REFUSED_GUARDRAIL = "refused_guardrail"


class Answer(Frozen):
    text: str
    status: AnswerStatus = AnswerStatus.OK
    citations: list[Citation] = Field(default_factory=list)
    groundedness: float = Field(default=0.0, ge=0.0, le=1.0)
    refusal_reason: str | None = None


class NumericResult(Frozen):
    label: str
    value: float | None
    unit: str = "USD"
    formula: str = ""
    inputs: dict[str, float] = Field(default_factory=dict)
    period: str = ""
    concept: str = ""
    source_url: str = ""


class QueryRequest(Mutable):
    question: Annotated[str, Field(min_length=3, max_length=2000)]
    top_k: Annotated[int, Field(ge=1, le=20)] = 6
    companies: list[str] = Field(
        default_factory=list, description="Optional ticker filter, for example ['AAPL']"
    )
    fiscal_years: list[int] = Field(default_factory=list)
    sections: list[FilingSection] = Field(default_factory=list)
    use_cache: bool = True
    use_reranker: bool = True
    reranker: str = Field(default="cross_encoder", description="cross_encoder, ltr, or none")


class QueryResponse(Mutable):
    question: str
    answer: Answer
    route: RouteDecision | None = None
    contexts: list[ScoredChunk] = Field(default_factory=list)
    numeric_results: list[NumericResult] = Field(default_factory=list)
    trace_id: str = ""
    cached: bool = False
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
