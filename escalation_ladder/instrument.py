"""Measured cost instrumentation shared by every rung.

The book's cost claims are measured, not asserted. Every rung wraps its model calls in
`measured`, so `scripts/measure_costs.py` can regenerate the cross-rung comparison table
from real numbers rather than from a table someone typed by hand.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class Measurement:
    """The cost of one instrumented call."""

    label: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model_calls: int = 0


@dataclass
class CostLedger:
    """Accumulates measurements across one rung's handling of one incident."""

    measurements: list[Measurement] = field(default_factory=list)

    def record(self, measurement: Measurement) -> None:
        self.measurements.append(measurement)

    @property
    def total_input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.measurements)

    @property
    def total_output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.measurements)

    @property
    def total_latency_ms(self) -> float:
        return sum(m.latency_ms for m in self.measurements)

    @property
    def model_calls(self) -> int:
        return sum(m.model_calls for m in self.measurements)


def measured(label: str, ledger: CostLedger) -> Callable[[F], F]:
    """Record latency, and token usage when the return value carries an Anthropic `usage`.

    A call with no `usage` attribute counts as zero model calls, which is what makes the
    Level 0 baseline row honest: deterministic code costs latency and nothing else.

    The `try/finally` matters. A rung that raises partway through still contributes its
    latency to the ledger; without it, a rung that crashes on an expensive call would look
    free in the cost table.
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            result: Any = None
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                usage = getattr(result, "usage", None)
                ledger.record(
                    Measurement(
                        label=label,
                        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                        latency_ms=elapsed_ms,
                        model_calls=1 if usage is not None else 0,
                    )
                )

        return wrapper  # type: ignore[return-value]

    return decorator
