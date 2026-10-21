from __future__ import annotations

import hashlib
import re
from datetime import date

import numpy as np
import pytest

from secrag.core.config import Settings
from secrag.core.types import Chunk, ChunkKind, Filing, FilingSection, ScoredChunk


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        llm_providers="echo",
        enable_splade=False,
        cache_max_entries=8,
        _env_file=None,
    )


@pytest.fixture
def filing() -> Filing:
    return Filing(
        filing_id="0000320193-10K-2024",
        cik="0000320193",
        ticker="AAPL",
        company="Apple Inc.",
        fiscal_year=2024,
        filing_date=date(2024, 11, 1),
        source_url="https://sec.gov/example.htm",
    )


def make_chunk(
    chunk_id: str = "c1",
    text: str = "The Company relies on single source suppliers for certain components.",
    section: FilingSection = FilingSection.RISK_FACTORS,
    ticker: str = "AAPL",
    fiscal_year: int = 2024,
    kind: ChunkKind = ChunkKind.PROSE,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        filing_id=f"{ticker}-10K-{fiscal_year}",
        text=text,
        kind=kind,
        section=section,
        company=f"{ticker} Inc.",
        ticker=ticker,
        fiscal_year=fiscal_year,
        source_url="https://sec.gov/example.htm",
        token_estimate=max(1, len(text) // 4),
    )


def make_scored(
    chunk_id: str = "c1", score: float = 0.9, stage: str = "dense", rank: int = 1, **kwargs
) -> ScoredChunk:
    return ScoredChunk(
        chunk=make_chunk(chunk_id=chunk_id, **kwargs),
        score=score,
        stage=stage,
        rank=rank,
        component_scores={stage: score, f"{stage}_rank": float(rank)},
    )


@pytest.fixture
def scored_chunks() -> list[ScoredChunk]:
    return [
        make_scored("c1", 0.92, rank=1, text="Supply chain disruption could reduce revenue."),
        make_scored("c2", 0.81, rank=2, text="Competition from larger firms is intense."),
        make_scored(
            "c3",
            0.70,
            rank=3,
            text="Currency movements may affect reported results.",
            section=FilingSection.MDA,
        ),
    ]


class FakeEmbedder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]
        for token in tokens:
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vec[index] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_documents(self, texts) -> np.ndarray:
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
