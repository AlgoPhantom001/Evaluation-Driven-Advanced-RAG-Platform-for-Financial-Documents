from __future__ import annotations

from collections.abc import Sequence

from secrag.core.types import NumericResult, ScoredChunk

SYSTEM_PROMPT = """You are a financial research assistant. You answer questions
strictly from the SEC filing excerpts you are given.

Rules you must follow:
1. Use ONLY the numbered context passages provided. Never use outside knowledge.
2. Cite every factual claim with the bracketed marker of its source, like [1] or [2].
   Place the marker at the end of the sentence it supports.
3. If the context does not contain the answer, say exactly what is missing.
   Do not guess, and do not fill gaps with general knowledge about the company.
4. When figures are supplied under "Verified figures", use those exact values.
   They were computed from the company's filed XBRL data and are authoritative.
   Never recompute or round them differently.
5. Quote specific numbers, dates, and defined terms verbatim from the context.
6. Be concise. Three to six sentences unless the question demands more.
7. Never invent a citation marker that does not appear in the context."""

REFUSAL_NO_CONTEXT = (
    "I could not find anything in the indexed filings that addresses this question. "
    "The corpus may not include the company, fiscal year, or topic you asked about."
)


def format_contexts(contexts: Sequence[ScoredChunk]) -> str:
    blocks: list[str] = []
    for i, scored in enumerate(contexts, start=1):
        chunk = scored.chunk
        blocks.append(f"[{i}] {chunk.citation_label()}\n{chunk.text}")
    return "\n\n".join(blocks)


def format_numeric(results: Sequence[NumericResult]) -> str:
    if not results:
        return ""
    lines = ["Verified figures, computed from filed XBRL data:"]
    for result in results:
        if result.value is None:
            continue
        value = f"{result.value:,.2f}".rstrip("0").rstrip(".")
        unit = "%" if result.unit == "percent" else f" {result.unit}"
        lines.append(f"  - {result.label}: {value}{unit}  ({result.formula})")
    return "\n".join(lines) if len(lines) > 1 else ""


def build_user_prompt(
    question: str,
    contexts: Sequence[ScoredChunk],
    numeric: Sequence[NumericResult] = (),
) -> str:
    parts = [format_contexts(contexts)]
    if numeric_block := format_numeric(numeric):
        parts.append(numeric_block)
    parts.append(f"Question: {question}")
    parts.append("Answer using only the context above, citing each claim with [n].")
    return "\n\n".join(parts)


DECOMPOSE_PROMPT = """Break this question into the smallest set of independent \
sub-questions needed to answer it. Each sub-question must be answerable on its own \
from a single company's filing.

Return JSON: {{"subquestions": ["...", "..."]}}
Return at most {max_parts}. If the question is already simple, return it unchanged.

Question: {question}"""
