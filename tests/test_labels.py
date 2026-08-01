"""The golden set is only ground truth if the ground is still there.

Chapter 13 argues that a label which cannot be executed rots silently: the
fixtures drift, the probe stops pointing at anything, and the tool-choice score
quietly measures nothing. Every probe in `labels.py` is therefore a claim about
the fixture corpus, and this file runs all of them.
"""
from __future__ import annotations

import pytest

from escalation_ladder.fixtures.incidents import load_incidents
from escalation_ladder.fixtures.labels import GOLDEN
from escalation_ladder.fixtures.retrieval_labels import LABELS
from escalation_ladder.llm import ToolCall
from escalation_ladder.tools import Toolbox

INCIDENTS = {incident.incident_id: incident for incident in load_incidents()}


def test_every_incident_has_a_label():
    assert set(GOLDEN) == set(INCIDENTS)


def test_service_and_severity_are_not_restated():
    """One source of truth for the two fields Chapter 5 already labeled."""
    for incident_id, golden in GOLDEN.items():
        assert golden.service == LABELS[incident_id].service
        assert golden.severity == LABELS[incident_id].severity
        assert golden.answers == LABELS[incident_id].answers


@pytest.mark.parametrize("incident_id", sorted(GOLDEN))
def test_every_probe_still_finds_its_evidence(incident_id: str):
    """Execute the label. A probe pointing at nothing is a silent zero."""
    golden = GOLDEN[incident_id]
    box = Toolbox(incident=INCIDENTS[incident_id])
    for probe in golden.probes:
        result = box.execute(
            ToolCall(call_id="probe", name=probe.tool, arguments=dict(probe.arguments))
        )
        assert not result.is_error, f"{incident_id}: {probe.tool} errored: {result.content}"
        assert probe.contains in result.content, (
            f"{incident_id}: {probe.tool}{probe.arguments} no longer contains "
            f"{probe.contains!r}"
        )


def test_the_unreachable_incident_has_no_probes():
    """INC-1043's answer is in a runbook. A probe here would be a false label.

    Chapter 9 spent a chapter establishing that this incident cannot be settled
    from the tools; inventing a probe for it would make the tool-choice surface
    score a rung for failing to do something impossible.
    """
    unreachable = [g for g in GOLDEN.values() if not g.reachable]
    assert [g.incident_id for g in unreachable] == ["INC-1043"]
    assert unreachable[0].probes == ()


def test_probes_are_argument_level():
    """A probe with only a tool name would be satisfied by any investigation."""
    for golden in GOLDEN.values():
        for probe in golden.probes:
            assert probe.arguments, f"{golden.incident_id}: {probe.tool} has no arguments"
