from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
REPORTS = ROOT / "evals" / "reports"

failures: list[str] = []
checks = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(f"{name}: {detail}")


print("Evaluation metrics in README match evals/reports/latest.json")
eval_path = REPORTS / "latest.json"
if eval_path.exists():
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    for key, label in [
        ("hit_rate@6", "Hit rate@6"),
        ("ndcg@6", "nDCG@6"),
        ("mrr", "MRR"),
        ("groundedness", "Groundedness"),
        ("numeric_accuracy", "Numeric accuracy"),
        ("routing_accuracy", "Routing accuracy"),
    ]:
        value = data["summary"].get(key)
        if value is None:
            continue
        check(f"{label} = {value:.3f}", f"**{value:.3f}**" in README, "not found in README")
    check("gate passed", data["passed"], f"failures: {data['failures']}")
else:
    check("evaluation report exists", False, "run: secrag eval")

print("\nBenchmark table in README matches evals/reports/benchmark.json")
bench_path = REPORTS / "benchmark.json"
if bench_path.exists():
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    check(
        f"corpus size {bench['corpus_chunks']:,}",
        f"{bench['corpus_chunks']:,}" in README,
        "corpus size not stated in README",
    )
    for config in bench["configurations"]:
        if not config.get("available"):
            continue
        check(
            f"{config['name']} nDCG {config['ndcg']:.3f}",
            f"{config['ndcg']:.3f}" in README,
            "row missing from README",
        )
    check(
        "in-sample result is labelled",
        all(
            not c.get("in_sample") or "training-set performance" in README
            for c in bench["configurations"]
        ),
        "an in-sample score is reported without a caveat",
    )
else:
    check("benchmark report exists", False, "run: secrag benchmark")

print("\nRouter metrics in README match evals/reports/router_training.json")
router_path = REPORTS / "router_training.json"
if router_path.exists():
    router = json.loads(router_path.read_text(encoding="utf-8"))
    check(
        f"router accuracy {router['cv_accuracy']:.3f}",
        f"{router['cv_accuracy']:.3f}" in README,
        "not found in README",
    )
else:
    check("router report exists", False, "run: secrag train-router")

print("\nNothing secret is committed")
tracked = subprocess.run(
    ["git", "ls-files"], capture_output=True, text=True, cwd=ROOT, check=False
).stdout.splitlines()

check(".env is not tracked", ".env" not in tracked, ".env is in the index")
check("no index data tracked", not any(t.startswith("data/index/qdrant") for t in tracked))
check("no model weights tracked", not any(t.startswith("data/models/") for t in tracked))
check("no raw filings tracked", not any(t.startswith("data/raw/edgar") for t in tracked))

key_pattern = re.compile(r"\b(gsk_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_\-]{30,})")
leaked = []
for rel in tracked:
    path = ROOT / rel
    if not path.is_file() or path.suffix in {".png", ".jpg", ".parquet", ".joblib"}:
        continue
    try:
        if match := key_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            leaked.append(f"{rel}: {match.group(0)[:12]}...")
    except OSError:
        continue
check("no API keys in tracked files", not leaked, "; ".join(leaked))

EM_DASH = chr(0x2014)

print("\nThe em dash character does not appear in any tracked file")
em_dash_files = [
    rel
    for rel in tracked
    if (ROOT / rel).is_file()
    and (ROOT / rel).suffix not in {".png", ".jpg", ".parquet", ".joblib"}
    and EM_DASH in (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
]
check("no em dash present", not em_dash_files, ", ".join(em_dash_files[:5]))

print(f"\n{checks - len(failures)}/{checks} checks passed")
if failures:
    print("\nFailures:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("All README claims are backed by the reports in evals/reports/.")
