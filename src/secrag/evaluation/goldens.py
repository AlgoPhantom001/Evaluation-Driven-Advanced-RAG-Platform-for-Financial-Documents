from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from secrag.core.config import Settings, get_settings
from secrag.core.errors import EvaluationError
from secrag.core.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class GoldenCase:
    id: str
    question: str
    intent: str
    companies: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)
    expected_sections: list[str] = field(default_factory=list)
    expected_terms: list[str] = field(default_factory=list)
    must_include: list[str] = field(default_factory=list)
    expected_value: float | None = None
    tolerance_pct: float = 1.0
    expect_refusal: bool = False
    notes: str = ""

    @property
    def is_numeric(self) -> bool:
        return self.expected_value is not None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GoldenCase:
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            intent=str(payload.get("intent", "factoid")),
            companies=[str(c).upper() for c in payload.get("companies", [])],
            fiscal_years=[int(y) for y in payload.get("fiscal_years", [])],
            expected_sections=[str(s) for s in payload.get("expected_sections", [])],
            expected_terms=[str(t) for t in payload.get("expected_terms", [])],
            must_include=[str(t) for t in payload.get("must_include", [])],
            expected_value=(
                float(payload["expected_value"])
                if payload.get("expected_value") is not None
                else None
            ),
            tolerance_pct=float(payload.get("tolerance_pct", 1.0)),
            expect_refusal=bool(payload.get("expect_refusal", False)),
            notes=str(payload.get("notes", "")),
        )


def golden_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.project_root / "evals" / "goldens" / "golden_set.json"


def load_goldens(path: Path | None = None, settings: Settings | None = None) -> list[GoldenCase]:
    path = path or golden_path(settings)
    if not path.exists():
        msg = f"Golden set not found at {path}"
        raise EvaluationError(msg)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Golden set at {path} is not valid JSON"
        raise EvaluationError(msg, detail=str(exc)) from exc

    cases_raw = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(cases_raw, list):
        msg = "Golden set must contain a list of cases"
        raise EvaluationError(msg)

    cases = [GoldenCase.from_dict(item) for item in cases_raw]

    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            msg = f"Duplicate golden case id: {case.id}"
            raise EvaluationError(msg)
        seen.add(case.id)

    log.info("goldens_loaded", count=len(cases), path=str(path))
    return cases


def filter_cases(cases: Sequence[GoldenCase], intents: Sequence[str] = ()) -> list[GoldenCase]:
    if not intents:
        return list(cases)
    wanted = {i.lower() for i in intents}
    return [c for c in cases if c.intent.lower() in wanted]
