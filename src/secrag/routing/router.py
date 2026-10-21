from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import QueryIntent, RouteDecision
from secrag.observability.tracing import span
from secrag.retrieval.embedder import Embedder
from secrag.routing.dataset import TRAINING_EXAMPLES

log = get_logger(__name__)

_YEAR_RE = re.compile(r"\b(?:fy\s?)?(?:19|20)\d{2}\b|\bfy\s?\d{2}\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d")
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")

_COMPARATIVE_WORDS = frozenset(
    {
        "compare",
        "compared",
        "comparison",
        "versus",
        "vs",
        "contrast",
        "between",
        "rank",
        "ranking",
        "higher",
        "lower",
        "greater",
        "larger",
        "smaller",
        "better",
        "worse",
        "than",
        "either",
    }
)
_CHANGE_WORDS = frozenset(
    {
        "change",
        "changed",
        "changes",
        "evolve",
        "evolved",
        "evolution",
        "trend",
        "trends",
        "shift",
        "shifted",
        "since",
        "over",
        "trace",
        "track",
        "prior",
        "previous",
        "added",
        "removed",
        "new",
        "expanded",
        "developed",
    }
)
_METRIC_WORDS = frozenset(
    {
        "revenue",
        "revenues",
        "income",
        "margin",
        "margins",
        "assets",
        "liabilities",
        "cash",
        "equity",
        "eps",
        "earnings",
        "growth",
        "cagr",
        "profit",
        "sales",
        "expense",
        "spend",
        "outstanding",
        "shares",
        "flow",
    }
)
_QUANTITY_PHRASES = (
    "how much",
    "how many",
    "what was the",
    "what is the",
    "calculate",
    "compute",
    "give me the",
    "show the",
    "total ",
    "percentage",
    "figure",
)

LEXICAL_FEATURE_NAMES: tuple[str, ...] = (
    "has_year",
    "has_number",
    "comparative_hits",
    "change_hits",
    "metric_hits",
    "quantity_phrase",
    "candidate_tickers",
    "token_count",
)


@dataclass(slots=True)
class RouterReport:
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    n_examples: int = 0
    n_features: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cv_accuracy": round(self.accuracy, 4),
            "cv_macro_f1": round(self.macro_f1, 4),
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
            "confusion_matrix": self.confusion,
            "labels": self.labels,
            "n_examples": self.n_examples,
            "n_features": self.n_features,
        }


def lexical_features(query: str) -> list[float]:
    lowered = query.lower()
    words = set(re.findall(r"[a-z]+", lowered))
    return [
        1.0 if _YEAR_RE.search(query) else 0.0,
        1.0 if _NUMBER_RE.search(query) else 0.0,
        float(len(words & _COMPARATIVE_WORDS)),
        float(len(words & _CHANGE_WORDS)),
        float(len(words & _METRIC_WORDS)),
        1.0 if any(p in lowered for p in _QUANTITY_PHRASES) else 0.0,
        float(len(set(_TICKER_RE.findall(query)))),
        float(len(lowered.split())),
    ]


class QueryRouter:
    def __init__(self, settings: Settings | None = None, embedder: Embedder | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings)
        self.model_path: Path = self.settings.router_model_path or (
            self.settings.index_dir / "router.joblib"
        )
        self._pipeline: Any | None = None
        self._classes: list[str] = []
        self._load_failed = False

    def _matrix(self, queries: Sequence[str]) -> np.ndarray:
        embeddings = self.embedder.embed_documents(list(queries))
        lexical = np.asarray([lexical_features(q) for q in queries], dtype=np.float32)
        return np.hstack([embeddings, lexical]).astype(np.float32)

    def train(self, *, folds: int = 5, seed: int = 42) -> RouterReport:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, f1_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        queries = [q for q, _ in TRAINING_EXAMPLES]
        labels = np.asarray([intent.value for _, intent in TRAINING_EXAMPLES])

        with span("router_train", examples=len(queries)):
            features = self._matrix(queries)

            def make_pipeline() -> Pipeline:
                return Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=2000,
                                C=1.0,
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                )

            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            predicted = cross_val_predict(make_pipeline(), features, labels, cv=splitter)

            accuracy = float((predicted == labels).mean())
            macro_f1 = float(f1_score(labels, predicted, average="macro"))
            class_order = sorted(set(labels))
            per_class = f1_score(labels, predicted, average=None, labels=class_order)
            matrix = confusion_matrix(labels, predicted, labels=class_order)

            pipeline = make_pipeline()
            pipeline.fit(features, labels)

            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"pipeline": pipeline, "classes": list(pipeline.classes_)}, self.model_path)
            self._pipeline = pipeline
            self._classes = list(pipeline.classes_)
            self._load_failed = False

        report = RouterReport(
            accuracy=accuracy,
            macro_f1=macro_f1,
            per_class_f1=dict(zip(class_order, (float(f) for f in per_class), strict=True)),
            confusion=[[int(v) for v in row] for row in matrix],
            labels=class_order,
            n_examples=len(queries),
            n_features=features.shape[1],
        )
        log.info("router_trained", accuracy=round(accuracy, 4), macro_f1=round(macro_f1, 4))
        return report

    @property
    def pipeline(self) -> Any | None:
        if self._pipeline is None and not self._load_failed:
            if not self.model_path.exists():
                self._load_failed = True
                log.warning("router_model_missing", path=str(self.model_path))
                return None
            import joblib

            payload = joblib.load(self.model_path)
            self._pipeline = payload["pipeline"]
            self._classes = list(payload["classes"])
            log.info("router_model_loaded", path=str(self.model_path))
        return self._pipeline

    @property
    def is_available(self) -> bool:
        return self.pipeline is not None

    def route(self, query: str) -> RouteDecision:
        pipeline = self.pipeline
        if pipeline is None:
            return RouteDecision(
                intent=QueryIntent.FACTOID, confidence=0.0, probabilities={}, fell_back=True
            )

        with span("route_query"):
            features = self._matrix([query])
            probabilities = pipeline.predict_proba(features)[0]

        ranked = dict(zip(self._classes, (float(p) for p in probabilities), strict=True))
        best_label = max(ranked, key=lambda k: ranked[k])
        confidence = ranked[best_label]

        fell_back = confidence < self.settings.router_confidence_threshold
        intent = QueryIntent.FACTOID if fell_back else QueryIntent(best_label)

        log.info(
            "query_routed",
            intent=intent.value,
            confidence=round(confidence, 4),
            fell_back=fell_back,
        )
        return RouteDecision(
            intent=intent,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in ranked.items()},
            fell_back=fell_back,
        )
