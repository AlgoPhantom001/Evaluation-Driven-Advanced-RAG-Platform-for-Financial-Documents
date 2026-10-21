from __future__ import annotations

import pytest

from secrag.evaluation import metrics
from secrag.retrieval.fusion import reciprocal_rank_fusion
from tests.conftest import make_scored


def test_rrf_promotes_documents_found_by_several_arms() -> None:
    dense = [make_scored("a", 0.9, rank=1), make_scored("b", 0.8, rank=2)]
    bm25 = [make_scored("c", 12.0, "bm25", 1), make_scored("b", 9.0, "bm25", 2)]

    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, k=60)

    assert fused[0].chunk.chunk_id == "b"


def test_rrf_preserves_component_scores_for_downstream_features() -> None:
    dense = [make_scored("a", 0.9, rank=1)]
    bm25 = [
        make_scored("x", 20.0, "bm25", 1),
        make_scored("y", 15.0, "bm25", 2),
        make_scored("a", 11.0, "bm25", 3),
    ]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": bm25}, k=60)

    winner = next(c for c in fused if c.chunk.chunk_id == "a")
    assert winner.component_scores["dense"] == pytest.approx(0.9)
    assert winner.component_scores["bm25"] == pytest.approx(11.0)
    assert winner.component_scores["bm25_rank"] == 3.0
    assert winner.component_scores["dense_rank"] == 1.0


def test_rrf_is_score_scale_invariant() -> None:
    small = [make_scored("a", 0.01, "bm25", 1), make_scored("b", 0.005, "bm25", 2)]
    large = [make_scored("a", 1000.0, "bm25", 1), make_scored("b", 500.0, "bm25", 2)]
    dense = [make_scored("b", 0.9, rank=1)]

    order_small = [c.chunk.chunk_id for c in reciprocal_rank_fusion({"s": small, "d": dense})]
    order_large = [c.chunk.chunk_id for c in reciprocal_rank_fusion({"s": large, "d": dense})]
    assert order_small == order_large


def test_rrf_respects_limit() -> None:
    arm = [make_scored(f"c{i}", 1.0 / (i + 1), rank=i + 1) for i in range(10)]
    assert len(reciprocal_rank_fusion({"dense": arm}, limit=3)) == 3


@pytest.mark.parametrize(
    ("relevance", "expected"),
    [([1, 0, 0], 1.0), ([0, 1, 0], 0.5), ([0, 0, 1], 1 / 3), ([0, 0, 0], 0.0)],
)
def test_reciprocal_rank(relevance: list[int], expected: float) -> None:
    assert metrics.reciprocal_rank(relevance) == pytest.approx(expected)


def test_ndcg_rewards_higher_placement() -> None:
    assert metrics.ndcg_at_k([1, 0, 0, 0], 4) > metrics.ndcg_at_k([0, 0, 0, 1], 4)
    assert metrics.ndcg_at_k([1, 1, 0, 0], 4) == pytest.approx(1.0)
    assert metrics.ndcg_at_k([0, 0, 0, 0], 4) == 0.0


def test_citation_validity_catches_invented_markers() -> None:
    assert metrics.citation_validity("Revenue rose [1]. Margins fell [2].", 3) == 1.0
    assert metrics.citation_validity("Revenue rose [7].", 3) == 0.0
    assert metrics.citation_validity("Revenue rose [1] and fell [9].", 3) == pytest.approx(0.5)
    assert metrics.citation_validity("No markers at all.", 3) == 0.0


def test_citation_density() -> None:
    assert metrics.citation_density("A [1]. B [2].") == pytest.approx(1.0)
    assert metrics.citation_density("A [1]. B.") == pytest.approx(0.5)


def test_numeric_accuracy_respects_tolerance() -> None:
    assert metrics.numeric_accuracy(100.0, 100.0) == 1.0
    assert metrics.numeric_accuracy(100.5, 100.0, tolerance_pct=1.0) == 1.0
    assert metrics.numeric_accuracy(105.0, 100.0, tolerance_pct=1.0) == 0.0
    assert metrics.numeric_accuracy(None, 100.0) == 0.0
    assert metrics.numeric_accuracy(100.0, None) is None


def test_accumulator_skips_none() -> None:
    acc = metrics.MetricAccumulator()
    acc.add("x", 1.0)
    acc.add("x", None)
    acc.add("x", 0.0)
    assert acc.count("x") == 2
    assert acc.mean("x") == pytest.approx(0.5)


def test_disabled_splade_is_dropped_not_fatal() -> None:
    from secrag.evaluation.benchmark import resolve_arms

    assert resolve_arms(("dense", "bm25", "splade"), enable_splade=False) == ("dense", "bm25")
    assert resolve_arms(("dense", "bm25", "splade"), enable_splade=True) == (
        "dense",
        "bm25",
        "splade",
    )


def test_splade_only_config_is_empty_when_disabled() -> None:
    from secrag.evaluation.benchmark import resolve_arms

    assert resolve_arms(("splade",), enable_splade=False) == ()
    assert resolve_arms(("splade",), enable_splade=True) == ("splade",)


def test_configs_without_splade_are_untouched() -> None:
    from secrag.evaluation.benchmark import resolve_arms

    assert resolve_arms(("dense",), enable_splade=False) == ("dense",)
    assert resolve_arms(("dense", "bm25"), enable_splade=False) == ("dense", "bm25")
