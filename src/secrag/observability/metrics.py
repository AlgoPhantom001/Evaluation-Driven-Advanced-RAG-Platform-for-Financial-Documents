from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field

DEFAULT_BUCKETS: tuple[float, ...] = (
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
)


@dataclass(slots=True)
class Histogram:
    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    counts: dict[float, int] = field(default_factory=dict)
    total: float = 0.0
    n: int = 0

    def observe(self, value: float) -> None:
        self.n += 1
        self.total += value
        for bound in self.buckets:
            if value <= bound:
                self.counts[bound] = self.counts.get(bound, 0) + 1

    def quantile(self, q: float) -> float:
        if self.n == 0:
            return 0.0
        target = q * self.n
        cumulative = 0
        for bound in self.buckets:
            cumulative = self.counts.get(bound, 0)
            if cumulative >= target:
                return bound
        return self.buckets[-1]


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, Histogram] = {}

    @staticmethod
    def _key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((labels or {}).items()))

    def increment(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[(name, self._key(labels))] += value

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms.setdefault(name, Histogram()).observe(value)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "counters": {
                    (f"{n}{{{','.join(f'{k}={v}' for k, v in labels)}}}" if labels else n): value
                    for (n, labels), value in self._counters.items()
                },
                "gauges": dict(self._gauges),
                "histograms": {
                    name: {
                        "count": h.n,
                        "sum": round(h.total, 2),
                        "mean": round(h.total / h.n, 2) if h.n else 0.0,
                        "p50": h.quantile(0.5),
                        "p95": h.quantile(0.95),
                        "p99": h.quantile(0.99),
                    }
                    for name, h in self._histograms.items()
                },
            }

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                rendered = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{rendered} {value}")

            for name, value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")

            for name, histogram in sorted(self._histograms.items()):
                lines.append(f"# TYPE {name} histogram")
                cumulative = 0
                for bound in histogram.buckets:
                    cumulative = max(cumulative, histogram.counts.get(bound, 0))
                    lines.append(f'{name}_bucket{{le="{bound}"}} {cumulative}')
                lines.append(f'{name}_bucket{{le="+Inf"}} {histogram.n}')
                lines.append(f"{name}_sum {histogram.total}")
                lines.append(f"{name}_count {histogram.n}")

        return "\n".join(lines) + "\n"


REGISTRY = MetricsRegistry()
