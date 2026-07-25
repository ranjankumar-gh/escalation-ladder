import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import measure_costs  # noqa: E402

from escalation_ladder.instrument import CostLedger, Measurement  # noqa: E402


def test_percentile_nearest_rank():
    assert measure_costs.percentile([1, 2, 3, 4, 5], 50) == 3
    assert measure_costs.percentile([1, 2, 3, 4, 5], 100) == 5
    assert measure_costs.percentile([5, 1, 3], 50) == 3      # sorts first


def test_percentile_of_an_empty_series_is_zero():
    assert measure_costs.percentile([], 50) == 0.0
    assert measure_costs.percentile([], 99) == 0.0


def test_percentile_p99_of_a_hundred_values_is_the_second_highest():
    # Nearest-rank: the 99th percentile is the value at rank ceil(0.99 * 100) = 99, i.e.
    # index 98. Exactly 99 of the 100 values are <= 98, so 98 is correct and 99 is the
    # 100th percentile, not the 99th.
    assert measure_costs.percentile(list(range(100)), 99) == 98
    assert measure_costs.percentile(list(range(100)), 100) == 99


def test_percentile_p99_of_a_small_series_is_the_max():
    # With the 6-incident fixture corpus, p99 collapses to the worst case -- which is the
    # honest reading of a tail percentile over a handful of samples.
    assert measure_costs.percentile([1.0, 2.0, 3.0, 4.0, 5.0, 99.0], 99) == 99.0


def test_render_markdown_table_emits_headers_with_no_rows():
    table = measure_costs.render_markdown_table([])
    lines = table.splitlines()
    assert lines[0].startswith("| Rung |")
    assert set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}
    assert len(lines) == 2          # header + separator only


def test_render_markdown_table_emits_one_line_per_row():
    rows = [
        {"Rung": "Level 0: Deterministic Code", "Model calls": 0, "Input tokens": 0,
         "Output tokens": 0, "p50 latency (ms)": 1.2, "p99 latency (ms)": 3.4},
    ]
    lines = measure_costs.render_markdown_table(rows).splitlines()
    assert len(lines) == 3
    assert "Level 0: Deterministic Code" in lines[2]


def test_summarise_ledgers_averages_tokens_and_percentiles_latency():
    ledgers = []
    for latency in (10.0, 20.0, 30.0):
        ledger = CostLedger()
        ledger.record(Measurement("call", input_tokens=100, output_tokens=50,
                                  latency_ms=latency, model_calls=1))
        ledgers.append(ledger)

    row = measure_costs.summarise_ledgers("Level 1: Prompt Application", ledgers)
    assert row["Rung"] == "Level 1: Prompt Application"
    assert row["Model calls"] == 1
    assert row["Input tokens"] == 100
    assert row["p99 latency (ms)"] == 30.0


def test_summarise_ledgers_of_no_ledgers_is_a_zero_row():
    row = measure_costs.summarise_ledgers("Level 9: Nothing", [])
    assert row["Rung"] == "Level 9: Nothing"
    assert row["Model calls"] == 0
    assert set(row) == set(measure_costs.COLUMNS)


def test_main_succeeds_with_no_rungs_registered(tmp_path):
    out = tmp_path / "table.md"
    exit_code = measure_costs.main(["--out", str(out)])
    assert exit_code == 0
    written = out.read_text(encoding="utf-8")
    assert "| Rung |" in written
