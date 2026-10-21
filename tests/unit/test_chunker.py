from __future__ import annotations

from secrag.core.config import Settings
from secrag.core.types import ChunkKind, FilingSection
from secrag.ingest.chunker import _split_table, chunk_blocks, estimate_tokens, split_sentences
from secrag.ingest.parser import Block


def test_sentences_survive_financial_abbreviations() -> None:
    text = "Apple Inc. reported growth. U.S. GAAP requires disclosure. Revenue rose."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Apple Inc. reported growth."


def test_split_table_repeats_header_in_every_part() -> None:
    header = "Year | Revenue | Income"
    rows = [f"20{i:02d} | {i * 1000} | {i * 100}" for i in range(30)]
    rendered = "\n".join([header, *rows])

    parts = _split_table(rendered, max_tokens=40)

    assert len(parts) > 1, "table should have been split"
    for part in parts:
        assert part.startswith(header), "a table fragment without its header is meaningless"


def test_small_table_is_not_split() -> None:
    rendered = "Year | Revenue\n2024 | 100\n2023 | 90"
    assert _split_table(rendered, max_tokens=500) == [rendered]


def test_chunks_are_deterministic(filing, settings: Settings) -> None:
    blocks = [
        Block(ChunkKind.PROSE, "Risk disclosure. " * 40, 0, section=FilingSection.RISK_FACTORS)
    ]
    first = chunk_blocks(blocks, filing, settings)
    second = chunk_blocks(blocks, filing, settings)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_consecutive_same_section_prose_is_merged(filing, settings: Settings) -> None:
    blocks = [
        Block(
            ChunkKind.PROSE,
            "First paragraph about risk. " * 5,
            0,
            section=FilingSection.RISK_FACTORS,
        ),
        Block(
            ChunkKind.PROSE,
            "Second paragraph about risk. " * 5,
            1,
            section=FilingSection.RISK_FACTORS,
        ),
    ]
    chunks = chunk_blocks(blocks, filing, settings)
    assert len(chunks) == 1
    assert "First paragraph" in chunks[0].text
    assert "Second paragraph" in chunks[0].text


def test_different_sections_are_never_merged(filing, settings: Settings) -> None:
    blocks = [
        Block(
            ChunkKind.PROSE, "Business description here. " * 6, 0, section=FilingSection.BUSINESS
        ),
        Block(ChunkKind.PROSE, "Risk disclosure here. " * 6, 1, section=FilingSection.RISK_FACTORS),
    ]
    chunks = chunk_blocks(blocks, filing, settings)
    assert {c.section for c in chunks} == {FilingSection.BUSINESS, FilingSection.RISK_FACTORS}


def test_tiny_fragments_are_dropped(filing, settings: Settings) -> None:
    blocks = [Block(ChunkKind.PROSE, "Page 12", 0, section=FilingSection.OTHER)]
    assert chunk_blocks(blocks, filing, settings) == []


def test_contextual_text_carries_provenance(filing, settings: Settings) -> None:
    blocks = [
        Block(
            ChunkKind.PROSE, "Supply risk exists here. " * 8, 0, section=FilingSection.RISK_FACTORS
        )
    ]
    chunk = chunk_blocks(blocks, filing, settings)[0]
    contextual = chunk.contextual_text()
    assert "AAPL" in contextual
    assert "FY2024" in contextual
    assert chunk.text in contextual


def test_token_estimate_is_monotonic() -> None:
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 100)
