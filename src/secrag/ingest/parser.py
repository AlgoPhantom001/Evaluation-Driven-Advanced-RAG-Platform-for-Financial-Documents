from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from selectolax.parser import HTMLParser, Node

from secrag.core.errors import IngestionError
from secrag.core.logging import get_logger
from secrag.core.types import ChunkKind, FilingSection

log = get_logger(__name__)

_SKIP_TAGS = frozenset({"script", "style", "head", "noscript", "svg", "iframe"})
_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}
)

_DASH = "\\-\u2010\u2011\u2012\u2013\u2014"
_SEP = rf"[\s.:;,{_DASH}]*"

_ITEM_PATTERNS: tuple[tuple[FilingSection, str], ...] = (
    (FilingSection.RISK_FACTORS, rf"item{_SEP}1a{_SEP}risk\s+factors"),
    (FilingSection.BUSINESS, rf"item{_SEP}1{_SEP}business"),
    (FilingSection.LEGAL_PROCEEDINGS, rf"item{_SEP}3{_SEP}legal\s+proceedings"),
    (FilingSection.MDA, rf"item{_SEP}7{_SEP}management.{{0,3}}s\s+discussion"),
    (FilingSection.MARKET_RISK, rf"item{_SEP}7a{_SEP}quantitative\s+and\s+qualitative"),
    (FilingSection.FINANCIAL_STATEMENTS, rf"item{_SEP}8{_SEP}financial\s+statements"),
    (FilingSection.CONTROLS, rf"item{_SEP}9a{_SEP}controls\s+and\s+procedures"),
)
_COMPILED = tuple((section, re.compile(pat, re.IGNORECASE)) for section, pat in _ITEM_PATTERNS)

_WS_RE = re.compile(r"[ \t\u00a0\u2007\u202f]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


@dataclass(slots=True)
class Block:
    kind: ChunkKind
    text: str
    order: int
    start: int = 0
    end: int = 0
    section: FilingSection = FilingSection.OTHER


_PUA_RE = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd]")


def _clean(text: str) -> str:
    text = _PUA_RE.sub(" ", text.replace("\xad", ""))
    return _NEWLINES_RE.sub("\n\n", _WS_RE.sub(" ", text)).strip()


def _render_table(node: Node) -> str:
    rows: list[str] = []
    for row in node.css("tr"):
        cells = [_clean(cell.text(separator=" ")) for cell in row.css("td, th")]
        cells = [c for c in cells if c not in {"", "$", "%", ")", "("}]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


_NUMERIC_CELL_RE = re.compile(r"^[\s$(){}\[\]%+*,.\-\d]*\d[\s$(){}\[\]%+*,.\-\d]*$")
_TOC_ROW_RE = re.compile(r"^\s*item\s+\d+[a-z]?\s*[.:]?\s*\|", re.IGNORECASE)


def _is_data_table(rendered: str) -> bool:
    lines = [ln for ln in rendered.split("\n") if ln.strip()]
    if len(lines) < 2 or sum(1 for ln in lines if "|" in ln) < 2:
        return False

    cells = [cell.strip() for line in lines for cell in line.split("|") if cell.strip()]
    if len(cells) < 4:
        return False

    if sum(1 for line in lines if _TOC_ROW_RE.match(line)) >= 3:
        return False

    numeric = sum(1 for cell in cells if _NUMERIC_CELL_RE.match(cell))
    return numeric / len(cells) >= 0.4


def _walk(node: Node | None) -> Iterator[tuple[str, str | Node]]:
    while node is not None:
        tag = node.tag
        if tag == "-text":
            if (raw := node.text_content) and raw.strip():
                yield "text", raw
        elif tag in _SKIP_TAGS:
            pass
        elif tag == "table":
            yield "table", node
        else:
            if tag in _BLOCK_TAGS:
                yield "break", ""
            yield from _walk(node.child)
            if tag in _BLOCK_TAGS:
                yield "break", ""
        node = node.next


def extract_blocks(html: str) -> list[Block]:
    if not html or not html.strip():
        msg = "Filing document was empty"
        raise IngestionError(msg)

    tree = HTMLParser(html)
    root = tree.body or tree.root
    if root is None:
        msg = "Filing document had no parseable body"
        raise IngestionError(msg)

    blocks: list[Block] = []
    buffer: list[str] = []
    cursor = 0

    def flush() -> None:
        nonlocal cursor
        if not (text := _clean(" ".join(buffer))):
            buffer.clear()
            return
        blocks.append(Block(ChunkKind.PROSE, text, len(blocks), cursor, cursor + len(text)))
        cursor += len(text) + 1
        buffer.clear()

    for kind, payload in _walk(root.child):
        if kind == "text":
            buffer.append(str(payload))
        elif kind == "break":
            flush()
        elif kind == "table":
            rendered = _render_table(payload)
            if _is_data_table(rendered):
                flush()
                blocks.append(
                    Block(ChunkKind.TABLE, rendered, len(blocks), cursor, cursor + len(rendered))
                )
                cursor += len(rendered) + 1
            else:
                buffer.append(rendered.replace("|", " "))
    flush()

    if not blocks:
        msg = "No readable content was extracted from the filing"
        raise IngestionError(msg)
    return blocks


def assign_sections(blocks: list[Block]) -> list[Block]:
    if not blocks:
        return blocks

    full = "\n".join(b.text for b in blocks)
    lowered = full.lower()

    hits: list[tuple[int, FilingSection]] = []
    for section, pattern in _COMPILED:
        hits.extend((m.start(), section) for m in pattern.finditer(lowered))

    if not hits:
        log.warning("no_item_headings_found", blocks=len(blocks))
        return blocks

    hits.sort()
    best: dict[FilingSection, tuple[int, int]] = {}
    for i, (pos, section) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(lowered)
        length = end - pos
        if section not in best or length > best[section][1]:
            best[section] = (pos, length)

    boundaries = sorted((pos, section) for section, (pos, _) in best.items())

    offsets: list[int] = []
    running = 0
    for block in blocks:
        offsets.append(running)
        running += len(block.text) + 1

    for block, offset in zip(blocks, offsets, strict=True):
        current = FilingSection.OTHER
        for pos, section in boundaries:
            if offset >= pos:
                current = section
            else:
                break
        block.section = current

    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.section.value] = counts.get(block.section.value, 0) + 1
    log.info("sections_assigned", counts=counts)
    return blocks


def parse_filing(html: str) -> list[Block]:
    return assign_sections(extract_blocks(html))
