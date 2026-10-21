from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import ScoredChunk
from secrag.observability.tracing import span
from secrag.retrieval.features import extract_features
from secrag.retrieval.rerank import Reranker

log = get_logger(__name__)


class XGBReranker(Reranker):
    """XGBoost learning-to-rank reranker.

    Trained by ``secrag train-xgb``; mirrors LTRReranker (LightGBM) so both
    gradient-boosted rankers can be benchmarked on the same feature matrix.
    """

    name = "xgb"

    def __init__(self, model_path: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_path = model_path or (self.settings.index_dir / "ltr_ranker_xgb.json")
        self._model: Any | None = None
        self._load_failed = False

    @property
    def model(self) -> Any | None:
        if self._model is None and not self._load_failed:
            if not self.model_path.exists():
                self._load_failed = True
                log.warning("xgb_model_missing", path=str(self.model_path))
                return None
            import xgboost as xgb

            booster = xgb.XGBRanker()
            booster.load_model(str(self.model_path))
            self._model = booster
            log.info("xgb_model_loaded", path=str(self.model_path))
        return self._model

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        model = self.model
        if model is None or not candidates:
            return candidates[:top_n]

        with span("rerank_xgb", candidates=len(candidates)):
            features = extract_features(query, candidates)
            # In XGBoost's Python API the rank objective's per-group output
            # is handled internally at fit time; predict takes the feature
            # matrix of one query-let and returns a score per row.
            scores = model.predict(features)

        ordered = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [
            ScoredChunk(
                chunk=candidate.chunk,
                score=float(score),
                stage="xgb",
                rank=rank,
                component_scores={**candidate.component_scores, "xgb": float(score)},
            )
            for rank, (candidate, score) in enumerate(ordered[:top_n], start=1)
        ]

    def warmup(self) -> None:
        _ = self.model
