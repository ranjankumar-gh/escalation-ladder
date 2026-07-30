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


def test_summarise_ledgers_of_no_ledgers_is_not_measured():
    """Amended in Chapter 10, and the old assertion WAS the defect.

    This used to assert a row of ZEROS, which seats a rung that measured nothing
    beside Level 0, the one rung that genuinely is free. Chapter 10 is the first
    to reach the branch - two of its three roles have no recordings, so every
    incident raises and no ledger survives - and a Level 7 row reading "0 tokens"
    would have advertised the most expensive rung in the book as the cheapest.
    """
    row = measure_costs.summarise_ledgers("Level 9: Nothing", [])
    assert row["Rung"] == "Level 9: Nothing"
    assert row["Model calls"] == "not measured"
    assert set(row) == set(measure_costs.COLUMNS)


def test_main_succeeds_with_no_rungs_registered(tmp_path):
    out = tmp_path / "table.md"
    exit_code = measure_costs.main(["--out", str(out)])
    assert exit_code == 0
    written = out.read_text(encoding="utf-8")
    assert "| Rung |" in written


def test_a_rung_whose_every_call_failed_reads_as_not_measured():
    """Added with Chapter 4.

    A rung that attempted calls and was billed for none of them measured nothing.
    Averaging that to zeros would seat it in the table looking exactly like
    Level 0, which genuinely is free - so a reader with no API key would read
    "Level 1: 0 tokens" as a result rather than as a missing key.
    """
    failed = [
        CostLedger(measurements=[Measurement("classify.ask", 0, 0, 3.1, 1)])
        for _ in range(6)
    ]
    row = measure_costs.summarise_ledgers("Level 1: Prompt Application", failed)
    assert row["Rung"] == "Level 1: Prompt Application"
    assert row["Input tokens"] == "not measured"
    assert row["p50 latency (ms)"] == "not measured"


def test_level_zero_is_still_reported_as_genuinely_free():
    """The counterpart: no model calls at all is a measurement, not a failure."""
    free = [
        CostLedger(measurements=[Measurement("rules.route", 0, 0, 0.0025, 0)])
        for _ in range(6)
    ]
    row = measure_costs.summarise_ledgers("Level 0: Deterministic Code", free)
    assert row["Model calls"] == 0
    assert row["Input tokens"] == 0
    assert row["p50 latency (ms)"] != "not measured"
