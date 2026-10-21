from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from secrag.core.types import ScoredChunk


def is_relevant(
    scored: ScoredChunk, expected_sections: Sequence[str], expected_terms: Sequence[str]
) -> bool:
    if expected_sections and scored.chunk.section.value not in expected_sections:
        return False
    if not expected_terms:
        return True
    lowered = scored.chunk.text.lower()
    return any(term.lower() in lowered for term in expected_terms)


def relevance_vector(
    results: Sequence[ScoredChunk],
    expected_sections: Sequence[str],
    expected_terms: Sequence[str],
) -> list[int]:
    return [1 if is_relevant(r, expected_sections, expected_terms) else 0 for r in results]


def hit_rate_at_k(relevance: Sequence[int], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def precision_at_k(relevance: Sequence[int], k: int) -> float:
    window = relevance[:k]
    return sum(window) / len(window) if window else 0.0


def recall_at_k(relevance: Sequence[int], k: int, total_relevant: int | None = None) -> float:
    denominator = total_relevant if total_relevant is not None else sum(relevance)
    return sum(relevance[:k]) / denominator if denominator else 0.0


def reciprocal_rank(relevance: Sequence[int]) -> float:
    for index, value in enumerate(relevance, start=1):
        if value:
            return 1.0 / index
    return 0.0


def dcg_at_k(relevance: Sequence[int], k: int) -> float:
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(relevance[:k], start=1))


def ndcg_at_k(relevance: Sequence[int], k: int) -> float:
    ideal = dcg_at_k(sorted(relevance, reverse=True), k)
    return dcg_at_k(relevance, k) / ideal if ideal > 0 else 0.0


_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


def citation_validity(answer_text: str, n_contexts: int) -> float:
    markers = [int(m) for m in _MARKER_RE.findall(answer_text)]
    if not markers:
        return 0.0
    valid = sum(1 for m in markers if 1 <= m <= n_contexts)
    return valid / len(markers)


def citation_density(answer_text: str) -> float:
    from secrag.generation.grounding import split_claims

    claims = split_claims(answer_text)
    if not claims:
        return 0.0
    cited = sum(1 for _, markers in claims if markers)
    return cited / len(claims)


def answer_contains(answer_text: str, required: Sequence[str]) -> float:
    if not required:
        return 1.0
    lowered = answer_text.lower()
    return sum(1 for term in required if term.lower() in lowered) / len(required)


def numeric_accuracy(
    actual: float | None, expected: float | None, tolerance_pct: float = 1.0
) -> float | None:
    if expected is None:
        return None
    if actual is None:
        return 0.0
    if expected == 0:
        return 1.0 if abs(actual) < 1e-9 else 0.0
    return 1.0 if abs(actual - expected) / abs(expected) * 100.0 <= tolerance_pct else 0.0


@dataclass(slots=True)
class MetricAccumulator:
    values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, value: float | None) -> None:
        if value is None:
            return
        self.values.setdefault(name, []).append(float(value))

    def mean(self, name: str) -> float:
        series = self.values.get(name, [])
        return sum(series) / len(series) if series else 0.0

    def count(self, name: str) -> int:
        return len(self.values.get(name, []))

    def summary(self) -> dict[str, float]:
        return {name: round(self.mean(name), 4) for name in sorted(self.values)}
