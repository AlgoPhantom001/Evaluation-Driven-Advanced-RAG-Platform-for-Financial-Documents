from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from secrag.core.config import Settings, get_settings
from secrag.core.logging import get_logger
from secrag.engine import QueryEngine, build_engine
from secrag.evaluation import metrics
from secrag.evaluation.goldens import load_goldens
from secrag.evaluation.ltr_training import collect_samples
from secrag.retrieval.features import FEATURE_NAMES

log = get_logger(__name__)

_BOOST_ROUNDS = 200


def _metrics_for_ranking(scores: np.ndarray, labels: np.ndarray, k: int) -> dict[str, float]:
    """Pointwise NDCG@k, MRR, Recall@k for one query-let of (score, label) pairs."""
    order = np.argsort(-scores, kind="stable")
    relevance = [int(v) for v in labels[order]]
    n_relevant = int(labels.sum())
    recall = metrics.recall_at_k(relevance, k, total_relevant=n_relevant) if n_relevant else 0.0
    return {
        "ndcg": metrics.ndcg_at_k(relevance, k),
        "mrr": metrics.reciprocal_rank(relevance),
        "recall": recall,
    }


def train_xgb_ltr_model(
    *,
    settings: Settings | None = None,
    engine: QueryEngine | None = None,
    n_folds: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Train an XGBoost rank:ndcg reranker, with grouped CV scored against RRF order.

    The same feature matrix and labels used for the LightGBM LTR pipeline are
    reused, so the two rankers are directly comparable. CV folds are grouped
    by query: no query's candidates appear in more than one fold.
    """
    import xgboost as xgb

    settings = settings or get_settings()
    engine = engine or build_engine(settings)
    engine.retriever.ensure_ready()

    cases = load_goldens(settings=settings)
    samples = collect_samples(engine, cases, settings=settings)

    if len(samples) < n_folds:
        msg = f"Only {len(samples)} usable queries, need at least {n_folds}"
        log.warning("xgb_training_skipped", reason=msg)
        return {"trained": False, "reason": msg, "n_queries": len(samples)}

    k = settings.eval_k

    def make_model() -> xgb.XGBRanker:
        return xgb.XGBRanker(
            objective="rank:ndcg",
            n_estimators=_BOOST_ROUNDS,
            max_depth=4,
            learning_rate=0.05,
            reg_lambda=1.0,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            tree_method="hist",
            n_jobs=0,
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    folds = np.array_split(order, n_folds)

    cv_metrics: list[dict[str, float]] = []
    baseline_metrics: list[dict[str, float]] = []

    for fold_index in range(n_folds):
        test_ids = set(folds[fold_index].tolist())
        train = [s for i, s in enumerate(samples) if i not in test_ids]
        test = [s for i, s in enumerate(samples) if i in test_ids]
        if not train or not test:
            continue

        model = make_model()
        model.fit(
            np.vstack([s.features for s in train]),
            np.concatenate([s.labels for s in train]),
            group=[s.n_candidates for s in train],
        )

        for sample in test:
            predicted = model.predict(sample.features)
            cv_metrics.append(_metrics_for_ranking(np.asarray(predicted), sample.labels, k))
            # Baseline: keep the fused RRF order (candidate 0 is the top fused
            # candidate), so scores are a strictly decreasing rank position.
            baseline_scores = np.arange(len(sample.labels), dtype=np.float32)[::-1]
            baseline_metrics.append(_metrics_for_ranking(baseline_scores, sample.labels, k))

    def mean_metric(rows: list[dict[str, float]], key: str) -> float:
        values = [row[key] for row in rows]
        return round(float(np.mean(values)), 4) if values else 0.0

    full = make_model()
    full.fit(
        np.vstack([s.features for s in samples]),
        np.concatenate([s.labels for s in samples]),
        group=[s.n_candidates for s in samples],
    )

    model_path: Path = settings.index_dir / "ltr_ranker_xgb.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    full.save_model(str(model_path))

    importance = {name: float(v) for name, v in zip(FEATURE_NAMES, full.feature_importances_)}

    report: dict[str, Any] = {
        "trained": True,
        "n_queries": len(samples),
        "n_candidates": int(sum(s.n_candidates for s in samples)),
        "cv_ndcg": mean_metric(cv_metrics, "ndcg"),
        "cv_mrr": mean_metric(cv_metrics, "mrr"),
        "cv_recall": mean_metric(cv_metrics, "recall"),
        "baseline_ndcg_fusion_order": mean_metric(baseline_metrics, "ndcg"),
        "baseline_mrr_fusion_order": mean_metric(baseline_metrics, "mrr"),
        "baseline_recall_fusion_order": mean_metric(baseline_metrics, "recall"),
        "lift_vs_fusion": round(
            mean_metric(cv_metrics, "ndcg") - mean_metric(baseline_metrics, "ndcg"), 4
        ),
        "feature_importance": dict(sorted(importance.items(), key=lambda kv: -kv[1])),
        "model_path": str(model_path),
    }

    report_path = settings.project_root / "evals" / "reports" / "xgb_training.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    log.info(
        "xgb_trained",
        cv_ndcg=report["cv_ndcg"],
        cv_mrr=report["cv_mrr"],
        cv_recall=report["cv_recall"],
        queries=len(samples),
    )
    return report
