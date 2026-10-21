from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from secrag.core.types import ChunkKind, FilingSection, ScoredChunk
from secrag.retrieval.bm25 import tokenize

FEATURE_NAMES: tuple[str, ...] = (
    "rrf_score",
    "dense_score",
    "dense_rank_inv",
    "bm25_score",
    "bm25_rank_inv",
    "splade_score",
    "splade_rank_inv",
    "arm_count",
    "token_estimate",
    "is_table",
    "section_id",
    "query_tokens",
    "lexical_overlap",
    "lexical_coverage",
    "numeric_density",
    "year_recency",
)

_SECTION_IDS = {section: i for i, section in enumerate(FilingSection)}
_CURRENT_YEAR = 2025


def _rank_inv(components: dict[str, float], arm: str) -> float:
    rank = components.get(f"{arm}_rank")
    return 1.0 / rank if rank else 0.0


def extract_features(query: str, candidates: Sequence[ScoredChunk]) -> np.ndarray:
    query_tokens = tokenize(query)
    query_set = set(query_tokens)
    rows: list[list[float]] = []

    for candidate in candidates:
        chunk = candidate.chunk
        components = candidate.component_scores
        chunk_tokens = set(tokenize(chunk.text))

        overlap = len(query_set & chunk_tokens)
        digits = sum(ch.isdigit() for ch in chunk.text)

        rows.append(
            [
                float(candidate.score),
                float(components.get("dense", 0.0)),
                _rank_inv(components, "dense"),
                float(components.get("bm25", 0.0)),
                _rank_inv(components, "bm25"),
                float(components.get("splade", 0.0)),
                _rank_inv(components, "splade"),
                float(sum(1 for arm in ("dense", "bm25", "splade") if arm in components)),
                float(chunk.token_estimate),
                1.0 if chunk.kind is ChunkKind.TABLE else 0.0,
                float(_SECTION_IDS.get(chunk.section, 0)),
                float(len(query_tokens)),
                float(overlap),
                overlap / len(query_set) if query_set else 0.0,
                digits / max(len(chunk.text), 1),
                float(_CURRENT_YEAR - chunk.fiscal_year),
            ]
        )

    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)
