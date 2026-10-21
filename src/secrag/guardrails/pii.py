from __future__ import annotations

import re
from dataclasses import dataclass, field

from secrag.core.logging import get_logger

log = get_logger(__name__)

_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("ssn", re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), "[REDACTED_SSN]"),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d{4}[ \-]){3}\d{4}\b"),
        "[REDACTED_CARD]",
    ),
    (
        "us_phone",
        re.compile(r"\b(?:\+1[ \-.])?\(?\d{3}\)?[ \-.]\d{3}[ \-.]\d{4}\b"),
        "[REDACTED_PHONE]",
    ),
)


@dataclass(slots=True)
class RedactionReport:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.counts)


def redact(text: str) -> RedactionReport:
    counts: dict[str, int] = {}
    cleaned = text
    for name, pattern, replacement in _RULES:
        cleaned, hits = pattern.subn(replacement, cleaned)
        if hits:
            counts[name] = hits
    if counts:
        log.info("pii_redacted", counts=counts)
    return RedactionReport(text=cleaned, counts=counts)
