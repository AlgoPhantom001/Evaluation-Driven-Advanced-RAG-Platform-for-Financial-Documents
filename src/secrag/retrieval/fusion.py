from __future__ import annotations

from collections.abc import Mapping, Sequence

from secrag.core.types import ScoredChunk
from secrag.observability.tracing import span


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[ScoredChunk]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[ScoredChunk]:
    with span("rrf_fusion", arms=len(ranked_lists)):
        fused: dict[str, float] = {}
        best: dict[str, ScoredChunk] = {}
        components: dict[str, dict[str, float]] = {}

        for arm, results in ranked_lists.items():
            weight = (weights or {}).get(arm, 1.0)
            for rank, scored in enumerate(results, start=1):
                chunk_id = scored.chunk.chunk_id
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (k + rank)
                components.setdefault(chunk_id, {})[arm] = scored.score
                components[chunk_id][f"{arm}_rank"] = float(rank)
                best.setdefault(chunk_id, scored)

        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
        if limit is not None:
            ordered = ordered[:limit]

        return [
            ScoredChunk(
                chunk=best[chunk_id].chunk,
                score=score,
                stage="fused",
                rank=rank,
                component_scores=components.get(chunk_id, {}),
            )
            for rank, (chunk_id, score) in enumerate(ordered, start=1)
        ]
