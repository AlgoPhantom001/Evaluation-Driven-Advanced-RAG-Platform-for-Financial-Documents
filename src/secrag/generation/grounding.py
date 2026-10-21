from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from secrag.core.logging import get_logger
from secrag.core.types import Citation, ScoredChunk
from secrag.observability.tracing import span
from secrag.retrieval.bm25 import tokenize
from secrag.retrieval.embedder import Embedder

log = get_logger(__name__)

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")
_VERIFIED_RE = re.compile(r"\[\s*verified[^\]]*\]", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[]|\Z)")

_SUPPORT_FLOOR = 0.35


@dataclass(slots=True)
class GroundingReport:
    groundedness: float
    citations: list[Citation] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    sentence_scores: list[float] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)


def split_answer_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def split_claims(text: str) -> list[tuple[str, list[int]]]:
    spans: list[tuple[str, list[int]]] = []
    buffer: list[str] = []

    for sentence in split_answer_sentences(text):
        buffer.append(sentence)
        markers = [int(m) for m in _MARKER_RE.findall(sentence)]
        if markers:
            spans.append((" ".join(buffer), markers))
            buffer = []

    if buffer:
        spans.append((" ".join(buffer), []))
    return spans


def _best_quote(sentence: str, chunk_text: str) -> str:
    claim_tokens = set(tokenize(sentence))
    if not claim_tokens:
        return chunk_text[:240]

    best, best_score = "", 0.0
    for candidate in _SENTENCE_SPLIT_RE.split(chunk_text):
        candidate = candidate.strip()
        if len(candidate) < 25:
            continue
        candidate_tokens = set(tokenize(candidate))
        if not candidate_tokens:
            continue
        score = len(claim_tokens & candidate_tokens) / len(claim_tokens)
        if score > best_score:
            best, best_score = candidate, score

    return (best or chunk_text[:240])[:400]


def verify(
    answer_text: str,
    contexts: Sequence[ScoredChunk],
    embedder: Embedder,
    *,
    has_verified_figures: bool = False,
) -> GroundingReport:
    claims = split_claims(answer_text)
    if not claims or not contexts:
        return GroundingReport(groundedness=0.0)

    with span("verify_groundedness", claims=len(claims)):

        def valid(marker: int) -> bool:
            return 0 < marker <= len(contexts)

        cited_indexes = sorted({m - 1 for _, markers in claims for m in markers if valid(m)})
        if not cited_indexes:
            return GroundingReport(
                groundedness=0.0,
                unsupported=[text for text, _ in claims if _is_factual(text)],
                sentence_scores=[0.0] * len(claims),
            )

        chunk_texts = [contexts[i].chunk.text for i in cited_indexes]
        claim_texts = [text for text, _ in claims]
        matrix = embedder.embed_documents([*claim_texts, *chunk_texts])
        claim_vectors = matrix[: len(claims)]
        chunk_vectors = {idx: matrix[len(claims) + n] for n, idx in enumerate(cited_indexes)}

        scores: list[float] = []
        unsupported: list[str] = []
        citations: dict[int, Citation] = {}

        for position, (text, markers) in enumerate(claims):
            usable = [m for m in markers if valid(m)]
            if not usable:
                if has_verified_figures and _VERIFIED_RE.search(text):
                    scores.append(1.0)
                    continue
                scores.append(0.0)
                if _is_factual(text):
                    unsupported.append(text)
                continue

            best_support = 0.0
            for marker in usable:
                index = marker - 1
                similarity = float(np.dot(claim_vectors[position], chunk_vectors[index]))
                best_support = max(best_support, similarity)

                if marker not in citations:
                    scored = contexts[index]
                    citations[marker] = Citation(
                        marker=marker,
                        chunk_id=scored.chunk.chunk_id,
                        label=scored.chunk.citation_label(),
                        source_url=scored.chunk.source_url,
                        quote=_best_quote(text, scored.chunk.text),
                        support_score=round(max(0.0, min(1.0, similarity)), 4),
                    )

            clipped = max(0.0, min(1.0, best_support))
            scores.append(clipped)
            if clipped < _SUPPORT_FLOOR:
                unsupported.append(text)

        groundedness = float(np.mean(scores)) if scores else 0.0

    report = GroundingReport(
        groundedness=round(groundedness, 4),
        citations=[citations[k] for k in sorted(citations)],
        unsupported=unsupported,
        sentence_scores=[round(s, 4) for s in scores],
    )
    log.info(
        "groundedness_verified",
        score=report.groundedness,
        claims=len(claims),
        citations=len(report.citations),
        unsupported=len(report.unsupported),
    )
    return report


_HEDGE_PREFIXES = (
    "based on",
    "according to",
    "the context",
    "i could not",
    "this answer",
    "in summary",
    "overall",
    "note that",
    "however",
    "additionally",
)


def _is_factual(sentence: str) -> bool:
    lowered = sentence.lower().strip()
    if any(lowered.startswith(prefix) for prefix in _HEDGE_PREFIXES):
        return False
    return len(tokenize(sentence)) >= 6
