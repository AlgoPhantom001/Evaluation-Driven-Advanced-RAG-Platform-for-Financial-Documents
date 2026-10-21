from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.core.types import ScoredChunk
from secrag.engine import QueryEngine, build_engine
from secrag.evaluation import metrics
from secrag.evaluation.goldens import GoldenCase, load_goldens
from secrag.retrieval.features import FEATURE_NAMES, extract_features
from secrag.retrieval.store import SearchFilter

log = get_logger(__name__)


@dataclass(slots=True)
class TrainingSample:
    query_id: str
    features: np.ndarray
    labels: np.ndarray

    @property
    def n_candidates(self) -> int:
        return len(self.labels)

    @property
    def has_signal(self) -> bool:
        return 0 < int(self.labels.sum()) < self.n_candidates


def collect_samples(
    engine: QueryEngine, cases: list[GoldenCase], *, settings: Settings
) -> list[TrainingSample]:
    samples: list[TrainingSample] = []

    for case in cases:
        flt = SearchFilter(tickers=case.companies, fiscal_years=case.fiscal_years)
        result = engine.retriever.retrieve(
            case.question,
            top_k=settings.rerank_candidates,
            flt=flt,
            reranker="none",
        )
        candidates: list[ScoredChunk] = result.fused or result.chunks
        if not candidates:
            continue

        labels = np.asarray(
            metrics.relevance_vector(candidates, case.expected_sections, case.expected_terms),
            dtype=np.int32,
        )
        sample = TrainingSample(
            query_id=case.id,
            features=extract_features(case.question, candidates),
            labels=labels,
        )
        if sample.has_signal:
            samples.append(sample)
        else:
            log.debug("ltr_sample_skipped", case=case.id, positives=int(labels.sum()))

    log.info("ltr_samples_collected", queries=len(samples))
    return samples


def _ndcg(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)
    return metrics.ndcg_at_k([int(v) for v in labels[order]], k)


def train_ltr_model(
    *,
    settings: Settings | None = None,
    engine: QueryEngine | None = None,
    n_folds: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    import lightgbm as lgb

    settings = settings or get_settings()
    engine = engine or build_engine(settings)
    engine.retriever.ensure_ready()

    cases = load_goldens(settings=settings)
    samples = collect_samples(engine, cases, settings=settings)

    if len(samples) < n_folds:
        msg = f"Only {len(samples)} usable queries, need at least {n_folds}"
        log.warning("ltr_training_skipped", reason=msg)
        return {"trained": False, "reason": msg, "n_queries": len(samples)}

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [settings.eval_k],
        "learning_rate": 0.08,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambdarank_truncation_level": 20,
        "verbosity": -1,
        "seed": seed,
    }

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    folds = np.array_split(order, n_folds)

    fold_scores: list[float] = []
    baseline_scores: list[float] = []

    for fold_index in range(n_folds):
        test_ids = set(folds[fold_index].tolist())
        train = [s for i, s in enumerate(samples) if i not in test_ids]
        test = [s for i, s in enumerate(samples) if i in test_ids]
        if not train or not test:
            continue

        dataset = lgb.Dataset(
            np.vstack([s.features for s in train]),
            label=np.concatenate([s.labels for s in train]),
            group=[s.n_candidates for s in train],
            feature_name=list(FEATURE_NAMES),
        )
        booster = lgb.train(params, dataset, num_boost_round=120)

        for sample in test:
            predicted = booster.predict(sample.features)
            fold_scores.append(_ndcg(np.asarray(predicted), sample.labels, settings.eval_k))
            baseline_scores.append(
                metrics.ndcg_at_k([int(v) for v in sample.labels], settings.eval_k)
            )

    full = lgb.Dataset(
        np.vstack([s.features for s in samples]),
        label=np.concatenate([s.labels for s in samples]),
        group=[s.n_candidates for s in samples],
        feature_name=list(FEATURE_NAMES),
    )
    final_booster = lgb.train(params, full, num_boost_round=120)

    model_path: Path = settings.index_dir / "ltr_ranker.txt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_booster.save_model(str(model_path))

    importance = dict(
        zip(
            FEATURE_NAMES,
            (int(v) for v in final_booster.feature_importance(importance_type="gain")),
            strict=True,
        )
    )

    report = {
        "trained": True,
        "n_queries": len(samples),
        "n_candidates": int(sum(s.n_candidates for s in samples)),
        "cv_ndcg": round(float(np.mean(fold_scores)), 4) if fold_scores else 0.0,
        "baseline_ndcg_fusion_order": (
            round(float(np.mean(baseline_scores)), 4) if baseline_scores else 0.0
        ),
        "lift_vs_fusion": (
            round(float(np.mean(fold_scores) - np.mean(baseline_scores)), 4) if fold_scores else 0.0
        ),
        "feature_importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
        "model_path": str(model_path),
    }

    report_path = settings.project_root / "evals" / "reports" / "ltr_training.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log.info("ltr_trained", cv_ndcg=report["cv_ndcg"], queries=len(samples))
    return report
