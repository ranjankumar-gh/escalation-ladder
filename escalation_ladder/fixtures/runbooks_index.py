"""Locate the runbook corpus. Chapter 5 embeds these; earlier chapters only list them."""
from __future__ import annotations

from pathlib import Path

RUNBOOK_DIR = Path(__file__).parent / "runbooks"


def runbook_paths() -> list[Path]:
    """Every runbook markdown file, in stable alphabetical order."""
    return sorted(RUNBOOK_DIR.glob("*.md"))
