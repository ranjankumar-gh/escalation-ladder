"""Chapter 15 - what a climb costs, measured on this repository.

The Addition Test has two clauses and this module implements both.

The STATIC clause asks whether a rung module imports a rung above it. That edge
costs nothing to add and is the one thing that makes Chapter 16's descent
impossible: you cannot delete Level 6 if Level 2 imports it. It is checked
against the source tree, by reading it rather than by importing it, so a module
that will not import is still checked.

The HISTORICAL clause asks what the last climb actually edited below itself. It
reads this repository's own tags, one per chapter, and splits every changed line
under `escalation_ladder/` by the ROLE of the file it landed in. Four roles can
receive a below-edit and only one of them counts against the test: a change in a
seam is what a seam is for, a change in a fixture is the simulated environment
growing, a change in a consumer is a tool being improved, and a change in a rung
module is the rung below being rewritten to accommodate the rung above.

Registers no rung. Appears in no cross-rung cost table. Standalone in the same
way `diagnose.py` is standalone, and for the same reason: an instrument that
measures the ladder must not be on it.
"""
from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from escalation_ladder.rungs import RUNG_MODULES

PACKAGE = "escalation_ladder"
REPO = Path(__file__).resolve().parents[1]

# The ladder order is read from the registry rather than restated here, so a
# rung added to `RUNG_MODULES` is ranked by this module without an edit. A
# second hand-maintained ordering is a second thing to get wrong.
RUNG_LEVEL: dict[str, int] = {
    module.split(".")[-1]: level for level, module in enumerate(RUNG_MODULES)
}

# A seam is a module every rung is allowed to import. Widening one is the
# expected cost of a climb; these four absorbed 75 percent of this book's.
SEAM_MODULES: frozenset[str] = frozenset(
    {"llm", "instrument", "orchestration", "rungs"}
)

Kind = Literal["new", "seam", "fixture", "consumer", "rung"]

# Everything except "new". A new file is not a below-edit at all, which is the
# whole point of the measurement.
BELOW: tuple[Kind, ...] = ("seam", "fixture", "consumer", "rung")


def module_of(path: str) -> str:
    """The top-level name a file belongs to.

    `escalation_ladder/fixtures/metrics.py` is `fixtures`, and so is
    `tests/test_fixtures.py` - a rung's tests get the rung's role, which is the
    only way the same question can be asked of the test tree.
    """
    parts = Path(path).parts
    if not parts:
        return ""
    if parts[0] == PACKAGE:
        return parts[1].removesuffix(".py") if len(parts) > 1 else ""
    if parts[0] == "tests" and len(parts) > 1:
        return parts[-1].removesuffix(".py").removeprefix("test_")
    return ""


def role(path: str) -> Kind:
    """Where a changed line landed, ignoring whether the file is new.

    Mechanical on purpose. The classifier gives you a number and the number is
    small enough that you then read the surviving diffs by hand - which is the
    intended workflow, not a shortcoming of it.
    """
    module = module_of(path)
    if module == "fixtures":
        return "fixture"
    if module in SEAM_MODULES:
        return "seam"
    if module in RUNG_LEVEL:
        return "rung"
    return "consumer"


@dataclass(frozen=True)
class Change:
    """One file's contribution to one climb."""

    path: str
    added: int
    removed: int
    kind: Kind

    @property
    def lines(self) -> int:
        """Added plus removed. A moved line is counted twice and should be."""
        return self.added + self.removed


@dataclass(frozen=True)
class Climb:
    """What one chapter's increment cost, split by where it landed.

    `measured` is false when the tags are not in the clone, and `decided_by`
    says which one was missing. It refuses rather than reporting zeros, because
    a climb with no edits below and a climb that was never measured are the two
    readings this whole chapter is about telling apart.
    """

    tag: str
    previous: str
    changes: tuple[Change, ...]
    measured: bool = True
    decided_by: str = "measured"

    @property
    def added(self) -> int:
        """Lines in files that did not exist before. The cost of the rung
        itself."""
        return sum(c.added for c in self.changes if c.kind == "new")

    def below(self, kind: Kind | None = None) -> int:
        """Lines changed in files that already existed, optionally by role."""
        kinds = (kind,) if kind else BELOW
        return sum(c.lines for c in self.changes if c.kind in kinds)

    @property
    def rung_lines(self) -> int:
        """The only number the Addition Test scores. A pass is zero."""
        return self.below("rung")


def _numstat(
    previous: str, tag: str, *, added: bool, scope: str
) -> list[tuple[int, int, str]]:
    filt = "A" if added else "M"
    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "diff",
            "--numstat",
            f"--diff-filter={filt}",
            f"{previous}..{tag}",
            "--",
            scope,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise LookupError(result.stderr.strip().splitlines()[-1:] or ["git failed"])
    rows: list[tuple[int, int, str]] = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or "-" in fields[:2]:
            continue  # A binary file has no line count; nothing here is binary.
        rows.append((int(fields[0]), int(fields[1]), fields[2]))
    return rows


def edits_below(previous: str, tag: str, *, scope: str = PACKAGE) -> Climb:
    """Measure one climb from the repository's own history.

    `scope` is a pathspec. The default is the package, because that is what the
    Addition Test scores - but point it at `tests` and you get the number this
    chapter found more interesting than its own headline.
    """
    try:
        new = _numstat(previous, tag, added=True, scope=scope)
        modified = _numstat(previous, tag, added=False, scope=scope)
    except LookupError as exc:
        return Climb(tag, previous, (), measured=False, decided_by=str(exc))
    changes = [Change(path, a, r, "new") for a, r, path in new]
    changes.extend(Change(path, a, r, role(path)) for a, r, path in modified)
    changes.sort(key=lambda c: (-c.lines, c.path))
    return Climb(tag, previous, tuple(changes))


def chapter_tags() -> tuple[str, ...]:
    """Every `chNN-` tag in the clone, in chapter order."""
    result = subprocess.run(
        ["git", "-C", str(REPO), "tag", "--list", "ch*"],
        capture_output=True,
        text=True,
    )
    return tuple(sorted(t for t in result.stdout.split() if t[:2] == "ch"))


def climbs(
    tags: tuple[str, ...] | None = None, *, scope: str = PACKAGE
) -> tuple[Climb, ...]:
    """Every consecutive pair of chapter tags, measured."""
    ordered = tags if tags is not None else chapter_tags()
    return tuple(
        edits_below(previous, tag, scope=scope)
        for previous, tag in zip(ordered, ordered[1:])
    )


def imports_of(module: str) -> tuple[str, ...]:
    """Top-level package modules imported by one module, read from source.

    Parsed rather than imported. Importing to find out what something imports
    runs the module, which needs its dependencies installed and its side effects
    to be acceptable - neither of which a design check should require.
    """
    source = (REPO / PACKAGE / f"{module}.py").read_text(encoding="utf-8")
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] == PACKAGE and len(parts) > 1 and parts[1] not in found:
                found.append(parts[1])
    return tuple(found)


def upward_imports() -> tuple[tuple[str, str], ...]:
    """Every place a rung module imports a rung above it.

    Empty is the invariant. An upward edge does not hurt the climb that adds it
    - it hurts the descent, years later, run by somebody who did not add it.
    """
    edges: list[tuple[str, str]] = []
    for module, level in RUNG_LEVEL.items():
        for imported in imports_of(module):
            if RUNG_LEVEL.get(imported, -1) > level:
                edges.append((module, imported))
    return tuple(edges)


@dataclass(frozen=True)
class AdditionResult:
    """Both clauses of the Addition Test, and which one decided the verdict."""

    passed: bool
    upward: tuple[tuple[str, str], ...]
    rung_lines: int
    decided_by: str


def addition_test(climb: Climb | None = None) -> AdditionResult:
    """Run the static clause always, and the historical clause when given one.

    The static clause is answerable before the next rung exists, which is the
    point: this is a design check applied to level N, not a post-mortem on
    level N+1.
    """
    upward = upward_imports()
    if upward:
        edges = ", ".join(f"{a} imports {b}" for a, b in upward)
        return AdditionResult(False, upward, 0, f"import points up the ladder: {edges}")
    if climb is None:
        return AdditionResult(True, (), 0, "no rung imports a rung above it")
    if not climb.measured:
        return AdditionResult(False, (), 0, f"not measured: {climb.decided_by}")
    if climb.rung_lines:
        touched = sorted({c.path for c in climb.changes if c.kind == "rung"})
        return AdditionResult(
            False,
            (),
            climb.rung_lines,
            f"{climb.rung_lines} lines changed in {', '.join(touched)}",
        )
    return AdditionResult(True, (), 0, f"{climb.tag} edited no rung module")
