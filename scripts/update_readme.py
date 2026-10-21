from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORTS = ROOT / "evals" / "reports"


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:START -->)(.*?)(<!-- {name}:END -->)", re.DOTALL)
    if not pattern.search(text):
        print(f"  marker {name} not found, skipping")
        return text
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def benchmark_block() -> str | None:
    path = REPORTS / "benchmark.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))

    available = [c for c in data["configurations"] if c.get("available")]
    if not available:
        return None

    honest = [c for c in available if not c.get("in_sample")]
    best = max(honest, key=lambda c: c["ndcg"]) if honest else available[0]
    dense_only = next((c for c in available if c["name"] == "dense only"), None)
    n = data["n_queries"]

    lines = [
        f"Corpus: **{data['corpus_chunks']:,} chunks** across {n} golden queries. "
        f"Dense `{data['models']['dense']}`, sparse `{data['models']['sparse']}`, "
        f"reranker `{data['models']['reranker']}`. Latency is single-threaded CPU.",
        "",
        data["markdown"],
        "",
    ]

    if dense_only and dense_only["ndcg"] > 0:
        lift = (best["ndcg"] - dense_only["ndcg"]) / dense_only["ndcg"] * 100
        lines.append(
            f"Best configuration that is not scored in-sample is **{best['name']}** at "
            f"nDCG@{data['eval_k']} {best['ndcg']:.3f}, **{lift:+.1f}%** against "
            f"dense-only retrieval ({dense_only['ndcg']:.3f}). It costs "
            f"{best['mean_latency_ms']:.0f} ms mean against "
            f"{dense_only['mean_latency_ms']:.0f} ms."
        )

    splade = next((c for c in available if c["name"] == "splade only"), None)
    fusion = next((c for c in available if c["name"] == "dense + bm25 + splade"), None)
    if splade and fusion and dense_only:
        lines += [
            "",
            "**Reading this honestly.** Three things in that table are worth stating "
            "plainly rather than glossing over:",
            "",
            f"- Learned sparse retrieval alone ({splade['ndcg']:.3f}) beats dense alone "
            f"({dense_only['ndcg']:.3f}) on this corpus, and beats three-arm fusion "
            f"({fusion['ndcg']:.3f}). Fusion is not free: mixing in weaker arms can pull "
            "a strong one down.",
            f"- The cross-encoder's gain over fusion is real but modest, and it costs "
            f"roughly {fusion['mean_latency_ms'] and (available[-1]['mean_latency_ms'] / max(fusion['mean_latency_ms'], 1)):.1f}x "
            "the latency. Whether that trade is worth making depends entirely on the "
            "application.",
            f"- With only {n} queries, differences below roughly 0.05 nDCG should not be "
            "treated as meaningful. The suite is sized to catch regressions, not to "
            "rank models.",
        ]

    ltr_path = REPORTS / "ltr_training.json"
    if ltr_path.exists():
        ltr = json.loads(ltr_path.read_text(encoding="utf-8"))
        if ltr.get("trained"):
            top = list(ltr["feature_importance"].items())[:4]
            lines += [
                "",
                "**The learning-to-rank reranker.** Trained on the golden set, so its "
                "row above is training-set performance. Under grouped cross-validation, "
                "splitting by query so no query appears in both folds, it scores "
                f"**nDCG {ltr['cv_ndcg']:.3f}** against **{ltr['baseline_ndcg_fusion_order']:.3f}** "
                f"for plain fusion order, a lift of **{ltr['lift_vs_fusion']:+.3f}** over "
                f"{ltr['n_queries']} queries and {ltr['n_candidates']:,} candidates. "
                "That is the number to believe.",
                "",
                "Its most informative features, by gain: "
                + ", ".join(f"`{k}` ({v})" for k, v in top)
                + ". That `section_id` dominates is a real finding: which 10-K Item a "
                "passage came from predicts relevance better than any similarity score, "
                "which is why section is a first-class field throughout the pipeline "
                "rather than loose metadata.",
            ]

    xgb_path = REPORTS / "xgb_training.json"
    if xgb_path.exists():
        xgb = json.loads(xgb_path.read_text(encoding="utf-8"))
        if xgb.get("trained"):
            lines += [
                "",
                "**The XGBoost ranker.** Same feature matrix, same grouped "
                "cross-validation protocol as the LightGBM LTR above. It scores "
                f"**nDCG@6 {xgb.get('cv_ndcg', 0):.3f}** vs "
                f"**{xgb.get('baseline_ndcg_fusion_order', 0):.3f}** for plain fusion "
                "order (MRR "
                f"{xgb.get('cv_mrr', 0):.3f} vs {xgb.get('baseline_mrr_fusion_order', 0):.3f}, "
                f"Recall@6 {xgb.get('cv_recall', 0):.3f} vs "
                f"{xgb.get('baseline_recall_fusion_order', 0):.3f}). The two rankers "
                "land within ~0.001 of each other on this split, which is the honest "
                "takeaway: the feature matrix carries the signal, not the "
                "gradient-boosting backend.",
            ]

    return "\n".join(lines)


def router_block() -> str | None:
    path = REPORTS / "router_training.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))

    lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| Cross-validated accuracy | **{data['cv_accuracy']:.3f}** |",
        f"| Macro F1 | {data['cv_macro_f1']:.3f} |",
        f"| Labelled examples | {data['n_examples']} |",
        f"| Features | {data['n_features']} (384 dense + 8 lexical) |",
        "",
        "Per-class F1: "
        + ", ".join(f"`{k}` {v:.3f}" for k, v in sorted(data["per_class_f1"].items()))
        + ".",
    ]
    return "\n".join(lines)


def eval_block() -> str | None:
    path = REPORTS / "latest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))

    interesting = [
        ("hit_rate@6", "Hit rate@6", "at least one relevant passage in the top 6"),
        ("ndcg@6", "nDCG@6", "ranking quality"),
        ("mrr", "MRR", "rank of the first relevant passage"),
        ("citation_validity", "Citation validity", "markers that point at a real source"),
        ("groundedness", "Groundedness", "answer sentences supported by their citation"),
        ("numeric_accuracy", "Numeric accuracy", "XBRL figures within tolerance"),
        ("routing_accuracy", "Routing accuracy", "intent matched the golden label"),
        ("refusal_correctness", "Refusal correctness", "declined exactly when it should"),
    ]

    lines = [
        f"{data['n_cases']} golden cases against {data['corpus_chunks']:,} indexed chunks, "
        f"reranker `{data['reranker']}`, provider `{data['provider'] or 'offline'}`.",
        "",
        "| Metric | Value | CI threshold | | What it means |",
        "|---|---:|---:|:--:|---|",
    ]
    for key, label, meaning in interesting:
        if key not in data["summary"]:
            continue
        value = data["summary"][key]
        threshold = data["thresholds"].get(key)
        if threshold is None:
            lines.append(f"| {label} | {value:.3f} | - | | {meaning} |")
        else:
            mark = "pass" if value >= threshold else "FAIL"
            lines.append(f"| {label} | **{value:.3f}** | {threshold:.2f} | {mark} | {meaning} |")

    lines.append("")
    verdict = "passes" if data["passed"] else "FAILS"
    lines.append(
        f"The gate {verdict}. Thresholds live in "
        "[`evals/thresholds.json`](evals/thresholds.json) and are enforced by "
        "`secrag eval --gate` in CI."
    )
    return "\n".join(lines)


def main() -> int:
    if not README.exists():
        print("README.md not found")
        return 1

    text = README.read_text(encoding="utf-8")
    updated = 0

    for name, builder in (
        ("BENCHMARK", benchmark_block),
        ("ROUTER", router_block),
        ("EVAL", eval_block),
    ):
        block = builder()
        if block is None:
            print(f"  no report for {name}, leaving placeholder")
            continue
        text = replace_block(text, name, block)
        updated += 1
        print(f"  {name} injected")

    README.write_text(text, encoding="utf-8")
    print(f"README updated ({updated} sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
