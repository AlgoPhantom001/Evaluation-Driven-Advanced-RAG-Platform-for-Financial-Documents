from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import ScoredChunk
from secrag.observability.tracing import span
from secrag.retrieval.features import extract_features

log = get_logger(__name__)


class Reranker(ABC):
    name: str = "none"

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        pass

    @property
    def is_available(self) -> bool:
        return True


class NoOpReranker(Reranker):
    name = "none"

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        return candidates[:top_n]


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    with span("load_reranker", model=self.settings.rerank_model):
                        self._model = TextCrossEncoder(
                            model_name=self.settings.rerank_model,
                            cache_dir=str(self.settings.models_dir),
                        )
                    log.info("reranker_loaded", model=self.settings.rerank_model)
        return self._model

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []

        pool = candidates[: self.settings.rerank_candidates]
        with span("rerank_cross_encoder", candidates=len(pool)):
            documents = [c.chunk.contextual_text() for c in pool]
            scores = list(self.model.rerank(query, documents))

        ordered = sorted(zip(pool, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            ScoredChunk(
                chunk=candidate.chunk,
                score=float(score),
                stage="cross_encoder",
                rank=rank,
                component_scores={**candidate.component_scores, "cross_encoder": float(score)},
            )
            for rank, (candidate, score) in enumerate(ordered[:top_n], start=1)
        ]

    def warmup(self) -> None:
        _ = self.model


class LTRReranker(Reranker):
    name = "ltr"

    def __init__(self, model_path: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model_path = model_path or (self.settings.index_dir / "ltr_ranker.txt")
        self._booster: Any | None = None
        self._load_failed = False

    @property
    def booster(self) -> Any | None:
        if self._booster is None and not self._load_failed:
            if not self.model_path.exists():
                self._load_failed = True
                log.warning("ltr_model_missing", path=str(self.model_path))
                return None
            import lightgbm as lgb

            self._booster = lgb.Booster(model_file=str(self.model_path))
            log.info("ltr_model_loaded", path=str(self.model_path))
        return self._booster

    @property
    def is_available(self) -> bool:
        return self.booster is not None

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        booster = self.booster
        if booster is None or not candidates:
            return candidates[:top_n]

        with span("rerank_ltr", candidates=len(candidates)):
            features = extract_features(query, candidates)
            scores = booster.predict(features)

        ordered = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [
            ScoredChunk(
                chunk=candidate.chunk,
                score=float(score),
                stage="ltr",
                rank=rank,
                component_scores={**candidate.component_scores, "ltr": float(score)},
            )
            for rank, (candidate, score) in enumerate(ordered[:top_n], start=1)
        ]


def build_reranker(name: str, settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()
    match name.strip().lower():
        case "cross_encoder":
            return CrossEncoderReranker(settings)
        case "ltr":
            return LTRReranker(settings=settings)
        case "xgb":
            from secrag.retrieval.xgb_rerank import XGBReranker

            return XGBReranker(settings=settings)
        case _:
            return NoOpReranker()
