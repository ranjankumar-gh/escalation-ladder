from dataclasses import dataclass

from escalation_ladder.instrument import CostLedger, Measurement, measured


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeMessage:
    usage: FakeUsage


def test_measured_records_latency_for_a_plain_function():
    ledger = CostLedger()

    @measured("plain", ledger)
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5
    assert len(ledger.measurements) == 1
    assert ledger.measurements[0].label == "plain"
    assert ledger.measurements[0].latency_ms >= 0.0


def test_measured_counts_no_model_call_without_usage():
    ledger = CostLedger()

    @measured("plain", ledger)
    def noop() -> None:
        return None

    noop()
    assert ledger.model_calls == 0
    assert ledger.total_input_tokens == 0


def test_measured_extracts_token_usage_when_present():
    ledger = CostLedger()

    @measured("llm", ledger)
    def call_model() -> FakeMessage:
        return FakeMessage(usage=FakeUsage(input_tokens=120, output_tokens=45))

    call_model()
    assert ledger.total_input_tokens == 120
    assert ledger.total_output_tokens == 45
    assert ledger.model_calls == 1


def test_ledger_totals_accumulate_across_calls():
    ledger = CostLedger()

    @measured("llm", ledger)
    def call_model() -> FakeMessage:
        return FakeMessage(usage=FakeUsage(input_tokens=10, output_tokens=5))

    for _ in range(3):
        call_model()

    assert ledger.model_calls == 3
    assert ledger.total_input_tokens == 30
    assert ledger.total_output_tokens == 15
    assert ledger.total_latency_ms >= 0.0


def test_measured_preserves_function_metadata():
    ledger = CostLedger()

    @measured("plain", ledger)
    def documented() -> None:
        """Docstring survives."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Docstring survives."


def test_measured_records_even_when_the_wrapped_call_raises():
    ledger = CostLedger()

    @measured("boom", ledger)
    def explode() -> None:
        raise ValueError("boom")

    try:
        explode()
    except ValueError:
        pass

    assert len(ledger.measurements) == 1
    assert ledger.measurements[0].label == "boom"
