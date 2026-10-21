from __future__ import annotations

import pytest

from secrag.core.errors import IngestionError
from secrag.core.types import ChunkKind, FilingSection
from secrag.ingest.parser import _is_data_table, extract_blocks, parse_filing

FILING_HTML = """
<html><body>
<p>TABLE OF CONTENTS</p>
<table>
  <tr><td>Item 1.</td><td>Business</td><td>1</td></tr>
  <tr><td>Item 1A.</td><td>Risk Factors</td><td>5</td></tr>
  <tr><td>Item 7.</td><td>Management's Discussion</td><td>20</td></tr>
</table>
<h2>Item 1. Business</h2>
<p>We design consumer devices and sell them worldwide through many channels.</p>
<h2>Item 1A. Risk Factors</h2>
<p>Supply chain disruption could materially reduce our revenue and margins.</p>
<table>
  <tr><th>Year</th><th>Revenue</th><th>Income</th></tr>
  <tr><td>2024</td><td>391,035</td><td>93,736</td></tr>
  <tr><td>2023</td><td>383,285</td><td>96,995</td></tr>
</table>
<h2>Item 7. Management's Discussion and Analysis</h2>
<p>Revenue increased two percent driven by services growth during the year.</p>
</body></html>
"""


def test_sections_are_assigned_past_the_table_of_contents() -> None:
    blocks = parse_filing(FILING_HTML)
    sections = {b.section for b in blocks}
    assert FilingSection.RISK_FACTORS in sections
    assert FilingSection.MDA in sections

    risk_text = " ".join(b.text for b in blocks if b.section is FilingSection.RISK_FACTORS)
    assert "Supply chain disruption" in risk_text


def test_table_of_contents_is_not_indexed_as_a_data_table() -> None:
    blocks = extract_blocks(FILING_HTML)
    tables = [b.text for b in blocks if b.kind is ChunkKind.TABLE]
    assert not any("Business" in t and "Risk Factors" in t for t in tables)


def test_financial_table_is_preserved_with_structure() -> None:
    blocks = extract_blocks(FILING_HTML)
    tables = [b.text for b in blocks if b.kind is ChunkKind.TABLE]
    assert len(tables) == 1
    assert "391,035" in tables[0]
    assert "|" in tables[0]


@pytest.mark.parametrize(
    ("rendered", "expected"),
    [
        ("Year | Revenue\n2024 | 391,035\n2023 | 383,285", True),
        ("California | 94-2404110\n(State) | (IRS No.)\nOne Apple Park Way | 95014", False),
        ("Item 1. | Business | 1\nItem 1A. | Risk Factors | 5\nItem 7. | MD&A | 20", False),
        ("single cell", False),
    ],
)
def test_data_table_heuristic(rendered: str, expected: bool) -> None:
    assert _is_data_table(rendered) is expected


def test_empty_document_raises() -> None:
    with pytest.raises(IngestionError):
        extract_blocks("   ")


def test_private_use_glyphs_are_stripped() -> None:
    html = (
        "<html><body><p>Products launched in the quarter: \uf0b7 iPhone 16; "
        "\uf0a7 Apple Watch Series 10; and \uf0b7 AirPods. "
        "These contributed to net sales growth during the period.</p></body></html>"
    )
    blocks = extract_blocks(html)
    combined = " ".join(b.text for b in blocks)

    assert "\uf0b7" not in combined
    assert "\uf0a7" not in combined
    assert "iPhone 16" in combined
    assert "Apple Watch Series 10" in combined
