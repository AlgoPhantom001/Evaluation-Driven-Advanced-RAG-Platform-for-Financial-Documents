from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "app.py",
    "requirements.txt",
    "pyproject.toml",
    "LICENSE",
    "src",
    "ui",
    "evals/goldens",
    "evals/thresholds.json",
]

INCLUDE_INDEX = [
    "data/index/qdrant",
    "data/index/facts.parquet",
    "data/index/router.joblib",
    "data/index/ltr_ranker.txt",
]

FRONTMATTER = """---
title: SEC RAG Platform
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
# Spaces default to Python 3.10. This project requires 3.11 or newer for
# StrEnum and datetime.UTC, so pip install refuses before it starts and the
# build fails with a cache-miss message that says nothing about the cause.
python_version: "3.12"
pinned: false
license: mit
short_description: RAG over SEC filings with XBRL-verified figures
---

"""


def run(cmd: list[str], cwd: Path, quiet: bool = False) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"\ncommand failed: {' '.join(cmd[:3])} ...", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        raise SystemExit(1)
    if not quiet and result.stdout.strip():
        print("  " + result.stdout.strip().splitlines()[-1])


FRONTMATTER_LIMITS = {"short_description": 60, "title": 100}


def check_frontmatter() -> None:
    problems = []
    for line in FRONTMATTER.splitlines():
        if ":" not in line or line.strip() in {"---", ""}:
            continue
        key, _, value = line.partition(":")
        limit = FRONTMATTER_LIMITS.get(key.strip())
        if limit and len(value.strip()) > limit:
            problems.append(f"  {key.strip()} is {len(value.strip())} characters, limit is {limit}")
    if problems:
        print("Space metadata is invalid:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(1)


def check_index() -> None:
    missing = [p for p in INCLUDE_INDEX if not (ROOT / p).exists()]
    if missing:
        print("The prebuilt index is incomplete. Missing:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("\nRun these first:", file=sys.stderr)
        print("  secrag ingest --rebuild", file=sys.stderr)
        print("  secrag train-router", file=sys.stderr)
        print("  secrag train-ltr", file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the Hugging Face Space")
    parser.add_argument("--username", required=True, help="Hugging Face username")
    parser.add_argument("--space", default="sec-rag-platform", help="Space name")
    args = parser.parse_args()

    check_frontmatter()
    check_index()

    token = (os.environ.get("HF_TOKEN") or "").strip()
    if token:
        print("Using HF_TOKEN from the environment.")
    else:
        token = getpass.getpass("Hugging Face write token (input hidden): ").strip()
    if not token.startswith("hf_"):
        print("That does not look like a Hugging Face token. They start with 'hf_'.")
        return 1

    url = f"https://huggingface.co/spaces/{args.username}/{args.space}"
    print(f"\nPublishing to {url}")

    with tempfile.TemporaryDirectory(prefix="secrag-space-") as tmp:
        staging = Path(tmp) / "space"
        staging.mkdir()

        print("\nStaging files")
        total = 0
        for rel in [*INCLUDE, *INCLUDE_INDEX]:
            source = ROOT / rel
            if not source.exists():
                continue
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
                )
            else:
                shutil.copy2(source, target)
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) or (
                target.stat().st_size if target.is_file() else 0
            )
            total += size
            print(f"  {rel:<32} {size / 1e6:>7.1f} MB")
        print(f"  {'total':<32} {total / 1e6:>7.1f} MB")

        readme = ROOT / "README.md"
        (staging / "README.md").write_text(
            FRONTMATTER + readme.read_text(encoding="utf-8"), encoding="utf-8"
        )

        (staging / ".env").write_text(
            "SECRAG_ENABLE_SPLADE=false\nSECRAG_LOG_JSON=true\n", encoding="utf-8"
        )
        (staging / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")

        print("\nPushing")
        run(["git", "init", "-q", "-b", "main"], staging, quiet=True)
        run(["git", "config", "user.name", args.username], staging, quiet=True)
        run(
            ["git", "config", "user.email", f"{args.username}@users.noreply.huggingface.co"],
            staging,
            quiet=True,
        )

        run(["git", "lfs", "install", "--local"], staging, quiet=True)
        (staging / ".gitattributes").write_text(
            "*.sqlite filter=lfs diff=lfs merge=lfs -text\n"
            "*.parquet filter=lfs diff=lfs merge=lfs -text\n",
            encoding="utf-8",
        )
        run(["git", "add", ".gitattributes"], staging, quiet=True)
        run(["git", "add", "-A"], staging, quiet=True)
        run(["git", "commit", "-q", "-m", "deploy space"], staging, quiet=True)

        tracked = subprocess.run(
            ["git", "lfs", "ls-files"],
            cwd=staging,
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if "storage.sqlite" not in tracked:
            print("\nThe index was not converted to LFS; the push would be rejected.")
            return 1
        print(f"  LFS tracking {len(tracked.strip().splitlines())} file(s)")

        remote = (
            f"https://{args.username}:{token}@huggingface.co/spaces/{args.username}/{args.space}"
        )
        run(["git", "push", "--force", remote, "main"], staging, quiet=True)

    print(f"\nDone. The Space is building at:\n  {url}")
    print("\nFirst build takes 5 to 10 minutes while dependencies install.")
    print("Add SECRAG_GROQ_API_KEY under Settings -> Variables and secrets,")
    print("otherwise the Space will run on the offline provider.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
