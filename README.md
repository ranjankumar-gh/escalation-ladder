# escalation-ladder

Companion code for **The 7 GenAI Architectures** by Ranjan Kumar.

One incident-triage implementation per rung of the Escalation Ladder, accreting one increment
per chapter. Each chapter's increment is tagged with its slug, so:

``` bash
git checkout ch06-llm-workflow
```

gives you the codebase exactly as it stands at the end of Chapter 6.

## Setup

``` bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
python -m pip install -e ".[dev]"
python -m pytest
```

Requires Python 3.12 or newer.

## Design rules

- **Modules are named by capability, never by level.** `retrieve.py`, not `level2.py`.
  Chapter 11 composes these paths and Chapter 16 falls back to them; both need the cheap
  paths to still exist. Nothing here is ever rewritten from scratch by a later chapter.
- **Fixtures are deterministic.** No network, no clock reads, no `random`. The simulated
  incident environment produces identical results on every machine and every run, which is
  what makes the book's measured numbers reproducible years after publication.
- **Cost is measured, not asserted.** Every rung wraps its model calls in
  `escalation_ladder.instrument.measured`, and `scripts/measure_costs.py` regenerates the
  cross-rung comparison table used in Chapter 2 and Appendix B.

## Layout

```
escalation_ladder/
  instrument.py      cost measurement shared by every rung
  rungs.py           rung registry; each level module registers itself
  fixtures/          the simulated incident environment
scripts/
  measure_costs.py   regenerate the cross-rung cost table
```

Rung modules (`rules.py`, `classify.py`, `retrieve.py`, `workflow.py`, `tools.py`,
`reasoning.py`, `agent.py`, `crew.py`) appear as their chapters land.
