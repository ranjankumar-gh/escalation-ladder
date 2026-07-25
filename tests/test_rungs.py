import escalation_ladder.rungs as rungs
from escalation_ladder.fixtures.incidents import SEED_INCIDENTS
from escalation_ladder.instrument import CostLedger


def test_register_rung_adds_to_the_registry():
    original = dict(rungs.RUNGS)
    try:
        @rungs.register_rung("Level 9: Test")
        def fake(incident):
            return CostLedger()

        assert "Level 9: Test" in rungs.RUNGS
        assert rungs.RUNGS["Level 9: Test"] is fake
    finally:
        rungs.RUNGS.clear()
        rungs.RUNGS.update(original)


def test_register_rung_returns_the_function_unchanged():
    original = dict(rungs.RUNGS)
    try:
        def fake(incident):
            return CostLedger()

        decorated = rungs.register_rung("Level 9: Test")(fake)
        assert decorated is fake
        assert decorated(SEED_INCIDENTS[0]).model_calls == 0
    finally:
        rungs.RUNGS.clear()
        rungs.RUNGS.update(original)


def test_rung_modules_are_listed_lowest_rung_first():
    assert rungs.RUNG_MODULES[0].endswith(".rules")
    assert rungs.RUNG_MODULES[-1].endswith(".crew")
    assert len(rungs.RUNG_MODULES) == 8


def test_rung_modules_are_named_by_capability_not_by_level():
    # A module called level3.py would break the accretion story the book depends on.
    for module in rungs.RUNG_MODULES:
        leaf = module.rsplit(".", 1)[-1]
        assert "level" not in leaf, f"{leaf} is named by level, not capability"


def test_load_all_tolerates_rung_modules_that_do_not_exist_yet():
    # No rung modules have been written yet, so this must return cleanly rather than raise
    # ModuleNotFoundError. It stays valid as modules land: the registry just gets fuller.
    registry = rungs.load_all()
    assert isinstance(registry, dict)
    assert registry is rungs.RUNGS


def test_load_all_reraises_a_missing_third_party_dependency(monkeypatch):
    """A broken `import anthropic` inside a rung module must fail loudly.

    Swallowing it would silently drop that rung from the published cost table.
    """
    import importlib

    def fake_import(name):
        # Simulate escalation_ladder.rules existing but failing on its own dependency.
        if name == "escalation_ladder.rules":
            raise ModuleNotFoundError("No module named 'anthropic'", name="anthropic")
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    try:
        rungs.load_all()
    except ModuleNotFoundError as exc:
        assert exc.name == "anthropic"
    else:
        raise AssertionError("load_all swallowed a missing third-party dependency")
