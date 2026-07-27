"""Chapter 4 - the SDK seam.

These tests pin the properties the chapter argues for structurally rather than
the SDK's behaviour:

  * `Completion` carries no exceptions in its contract, so every vendor failure
    arrives as a `failed` string and nothing above `llm.py` catches an SDK
    exception.
  * A refusal is a successful HTTP response, so `stop_reason` is checked before
    the content is trusted.
  * `RecordedCompleter` is keyed on the exact prompt, so editing a prompt
    invalidates every recording rather than quietly passing.

One test asserts a fact about the SDK itself - that unsupported JSON Schema
constraints are demoted into a field `description` rather than enforced - because
the chapter makes that claim and tells the reader to check it in a test rather
than assume it. If a future SDK release changes it, this is where the book finds
out.
"""

import httpx
import pytest
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

import anthropic
from anthropic.resources.messages.messages import transform_schema
from pydantic import TypeAdapter

from escalation_ladder.llm import (
    Completer,
    Completion,
    RecordedCompleter,
    Usage,
    AnthropicCompleter,
    prompt_key,
)


class Tiny(BaseModel):
    """Minimal schema, so these tests do not depend on Chapter 4's real one."""

    label: Literal["a", "b"]
    note: str


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class _FakeMessage:
    """The shape `messages.parse` returns, reduced to what llm.py reads."""

    def __init__(self, parsed, stop_reason="end_turn", tokens=(11, 7)):
        self.parsed_output = parsed
        self.stop_reason = stop_reason
        self.usage = Usage(input_tokens=tokens[0], output_tokens=tokens[1])


class _FakeMessages:
    def __init__(self, outcome):
        self._outcome = outcome
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _FakeClient:
    def __init__(self, outcome):
        self.messages = _FakeMessages(outcome)


def _completer(outcome) -> tuple[AnthropicCompleter, _FakeClient]:
    client = _FakeClient(outcome)
    return AnthropicCompleter(client=client), client


# --------------------------------------------------------------------------
# Completion is the whole contract
# --------------------------------------------------------------------------


def test_ok_requires_both_a_value_and_no_failure():
    assert Completion(Tiny(label="a", note="n"), Usage()).ok
    assert not Completion(None, Usage()).ok
    assert not Completion(Tiny(label="a", note="n"), Usage(), "boom").ok


def test_both_implementations_match_the_protocol_signature():
    """`Completer` is a static contract, so check the shape, not isinstance.

    Chapters 5 through 10 all call `parse` through this seam. An implementation
    that drifts - drops a keyword, renames one, makes one positional - breaks
    every rung above it, and a type checker is not run in CI.
    """
    import inspect

    expected = inspect.signature(Completer.parse)
    for impl in (AnthropicCompleter, RecordedCompleter):
        actual = inspect.signature(impl.parse)
        assert list(actual.parameters) == list(expected.parameters), impl.__name__
        for name, param in expected.parameters.items():
            if name == "self":
                continue
            assert param.kind == inspect.Parameter.KEYWORD_ONLY, name
            assert actual.parameters[name].kind == param.kind, (impl.__name__, name)
            assert actual.parameters[name].default == param.default, (
                impl.__name__,
                name,
            )


# --------------------------------------------------------------------------
# Every vendor failure becomes a string, never an exception
# --------------------------------------------------------------------------


def test_api_errors_are_translated_rather_than_raised():
    exc = anthropic.APIConnectionError(message="down", request=_request())
    completer, _ = _completer(exc)
    result = completer.parse(system="s", user="u", schema=Tiny)
    assert result.parsed is None
    assert not result.ok
    assert "api error" in result.failed
    assert "APIConnectionError" in result.failed


def test_rate_limits_are_translated_rather_than_raised():
    response = httpx.Response(429, request=_request())
    exc = anthropic.RateLimitError("slow down", response=response, body=None)
    completer, _ = _completer(exc)
    result = completer.parse(system="s", user="u", schema=Tiny)
    assert "RateLimitError" in result.failed


def test_client_side_validation_failure_names_the_field():
    try:
        Tiny.model_validate({"label": "zzz", "note": "n"})
    except ValidationError as exc:
        raised = exc
    completer, _ = _completer(raised)
    result = completer.parse(system="s", user="u", schema=Tiny)
    assert result.failed.startswith("schema violated:")
    assert "label" in result.failed
    # The tokens were spent and the ledger cannot see them. The chapter says so;
    # this pins it, so nobody later "fixes" it into a silent zero-cost path.
    assert result.usage == Usage(0, 0)


# --------------------------------------------------------------------------
# A refusal is a successful response, not an exception
# --------------------------------------------------------------------------


def test_missing_credentials_do_not_escape_the_seam():
    """The commonest configuration of all: a fresh clone with no key set.

    The SDK signals unresolvable credentials with a bare TypeError, at request
    time rather than at construction. Without translation it escapes as a
    traceback from inside llm.py - the exact failure the seam exists to contain,
    and the one that makes "it degrades to Level 0" false in practice.
    """
    completer, _ = _completer(TypeError("Could not resolve authentication method."))
    result = completer.parse(system="s", user="u", schema=Tiny)
    assert result.parsed is None
    assert not result.ok
    assert "no credentials" in result.failed
    assert "ANTHROPIC_API_KEY" in result.failed


def test_a_bad_schema_still_raises_rather_than_posing_as_a_config_problem():
    """The other TypeError the SDK raises must not be mistaken for a missing key.

    `messages.parse` raises TypeError both for unresolvable credentials and for an
    `output_format` it cannot build a schema from. Collapsing the second into the
    first would send a programmer chasing their environment, so it is rejected up
    front and stays an exception - raising is reserved for programmer error.
    """
    completer, client = _completer(_FakeMessage(None))
    with pytest.raises(TypeError, match="BaseModel subclass"):
        completer.parse(system="s", user="u", schema=dict)
    # Rejected before any call was attempted.
    assert client.messages.calls == []


def test_a_refusal_is_reported_and_still_carries_its_bill():
    message = _FakeMessage(Tiny(label="a", note="n"), stop_reason="refusal")
    completer, _ = _completer(message)
    result = completer.parse(system="s", user="u", schema=Tiny)
    assert result.parsed is None
    assert "declined" in result.failed
    # Billed even though there is no usable answer.
    assert result.usage.input_tokens == 11


def test_a_response_with_no_parsable_content_is_not_an_answer():
    completer, _ = _completer(_FakeMessage(None))
    result = completer.parse(system="s", user="u", schema=Tiny)
    assert result.parsed is None
    assert "no parsable content" in result.failed


def test_a_good_response_carries_the_parsed_value_and_the_usage():
    completer, _ = _completer(_FakeMessage(Tiny(label="b", note="hi")))
    result = completer.parse(system="s", user="u", schema=Tiny)
    assert result.ok
    assert result.parsed.label == "b"
    assert result.usage == Usage(11, 7)


# --------------------------------------------------------------------------
# effort and the schema compose rather than conflict
# --------------------------------------------------------------------------


def test_effort_and_schema_are_both_sent():
    """The chapter's claim: output_format merges INTO output_config.

    If a future SDK replaced the dict instead of merging, effort would silently
    vanish and every call would run at the default. That is a cost regression
    with no error, so it is pinned here.
    """
    completer, client = _completer(_FakeMessage(Tiny(label="a", note="n")))
    completer.parse(system="s", user="u", schema=Tiny, effort="low")
    sent = client.messages.calls[0]
    assert sent["output_format"] is Tiny
    assert sent["output_config"] == {"effort": "low"}
    assert sent["messages"] == [{"role": "user", "content": "u"}]
    assert sent["system"] == "s"


def test_unsupported_constraints_are_demoted_not_enforced():
    """The Format Illusion, at the mechanical level.

    `maxLength` and numeric bounds are not part of the enforced grammar. The SDK
    moves them into the field description, where they are a request to the model
    rather than a constraint on its tokens - and Pydantic then raises locally.
    """

    class Draft(BaseModel):
        severity: Literal["SEV1", "SEV2", "SEV3"]
        summary: str = Field(max_length=200)
        confidence: float = Field(ge=0.0, le=1.0)

    sent = transform_schema(TypeAdapter(Draft).json_schema())
    props = sent["properties"]

    # The enum survives as an enum: this one is real.
    assert props["severity"]["enum"] == ["SEV1", "SEV2", "SEV3"]

    # These do not survive as constraints at all.
    assert "maxLength" not in props["summary"]
    assert "maxLength" in props["summary"]["description"]
    assert "maximum" not in props["confidence"]
    assert "maximum" in props["confidence"]["description"]

    # And the client-side half still raises, after you have paid for the tokens.
    with pytest.raises(ValidationError):
        Draft.model_validate(
            {"severity": "SEV1", "summary": "x" * 201, "confidence": 0.5}
        )


# --------------------------------------------------------------------------
# The recorded fake is keyed on the prompt, on purpose
# --------------------------------------------------------------------------


def test_prompt_key_is_stable_and_prompt_sensitive():
    a = prompt_key(system="sys", user="usr")
    assert a == prompt_key(system="sys", user="usr")
    assert a != prompt_key(system="sys ", user="usr")
    assert a != prompt_key(system="sys", user="usr ")
    # The separator matters: without it, ("ab","c") and ("a","bc") would collide.
    assert prompt_key(system="ab", user="c") != prompt_key(system="a", user="bc")
    assert len(a) == 12


def test_recorded_completer_replays_a_hit():
    key = prompt_key(system="s", user="u")
    fake = RecordedCompleter(
        recordings={key: '{"label": "a", "note": "recorded"}'},
        usage=Usage(100, 20),
    )
    result = fake.parse(system="s", user="u", schema=Tiny)
    assert result.ok
    assert result.parsed.note == "recorded"
    assert result.usage == Usage(100, 20)


def test_editing_the_prompt_misses_every_recording():
    key = prompt_key(system="s", user="u")
    fake = RecordedCompleter(recordings={key: '{"label": "a", "note": "n"}'})
    result = fake.parse(system="s EDITED", user="u", schema=Tiny)
    assert result.parsed is None
    assert "no recording" in result.failed


def test_a_bad_recording_fails_the_way_production_would():
    key = prompt_key(system="s", user="u")
    fake = RecordedCompleter(recordings={key: '{"label": "not-a-member"}'})
    result = fake.parse(system="s", user="u", schema=Tiny)
    assert result.parsed is None
    assert result.failed.startswith("schema violated:")


# --------------------------------------------------------------------------
# invoke - one round of tools (Chapter 7)
#
# The vendor is stubbed rather than recorded here, because what is being
# asserted is the SHAPE of the exchange - two requests, tools on both, results
# fed back in one user turn - and a recording cannot fail when that shape
# changes. These are the tests that would go red if a future edit turned this
# seam into a loop.
# --------------------------------------------------------------------------
from dataclasses import dataclass as _dataclass  # noqa: E402
from typing import Any as _Any  # noqa: E402

from escalation_ladder.llm import ToolResult, ToolSpec as _ToolSpec  # noqa: E402


@_dataclass
class _Block:
    type: str
    id: str = ""
    name: str = ""
    input: dict | None = None


@_dataclass
class _Usage:
    input_tokens: int = 1000
    output_tokens: int = 100


@_dataclass
class _Message:
    content: list
    stop_reason: str
    usage: _Usage
    parsed_output: _Any = None


class _StubMessages:
    def __init__(self, script):
        self.script = script
        self.seen = []

    def create(self, **kwargs):
        self.seen.append(("create", kwargs))
        return self.script.pop(0)

    def parse(self, **kwargs):
        self.seen.append(("parse", kwargs))
        return self.script.pop(0)


class _StubClient:
    def __init__(self, script):
        self.messages = _StubMessages(script)


class _Answer(BaseModel):
    verdict: str


_SPEC = _ToolSpec(
    name="read_thing",
    description="Read a thing.",
    parameters={
        "type": "object",
        "properties": {"which": {"type": "string"}},
        "required": ["which"],
        "additionalProperties": False,
    },
    consequence="read",
)

_TOOL_BLOCK = _Block("tool_use", "toolu_1", "read_thing", {"which": "a"})


def _invoke(script, execute=None):
    completer = AnthropicCompleter(client=_StubClient(list(script)))
    run = completer.invoke(
        system="s",
        user="u",
        tools=(_SPEC,),
        execute=execute or (lambda call: ToolResult(call.call_id, "42")),
        schema=_Answer,
    )
    return run, completer._client.messages.seen


def test_invoke_makes_exactly_two_requests_and_no_third():
    """One round means one round. A third request here would be Level 5."""
    run, seen = _invoke([
        _Message([_TOOL_BLOCK], "tool_use", _Usage()),
        _Message([_Block("text")], "end_turn", _Usage(500, 50), _Answer(verdict="ok")),
    ])
    assert run.ok
    assert [kind for kind, _ in seen] == ["create", "parse"]


def test_invoke_sends_every_result_in_one_user_turn():
    run, seen = _invoke([
        _Message(
            [_TOOL_BLOCK, _Block("tool_use", "toolu_2", "read_thing", {"which": "b"})],
            "tool_use",
            _Usage(),
        ),
        _Message([_Block("text")], "end_turn", _Usage(500, 50), _Answer(verdict="ok")),
    ])
    _, second = seen[1]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert len(second["messages"][2]["content"]) == 2
    assert len(run.results) == 2


def test_invoke_leaves_the_tools_on_the_second_request():
    """Removing them would force an answer and hide the Failure Receipt."""
    _, seen = _invoke([
        _Message([_TOOL_BLOCK], "tool_use", _Usage()),
        _Message([_Block("text")], "end_turn", _Usage(500, 50), _Answer(verdict="ok")),
    ])
    assert seen[1][1]["tools"]


def test_invoke_reports_a_second_round_instead_of_running_it():
    run, seen = _invoke([
        _Message([_TOOL_BLOCK], "tool_use", _Usage()),
        _Message([_TOOL_BLOCK], "tool_use", _Usage(500, 50)),
    ])
    assert run.wanted_more
    assert not run.ok
    assert len(seen) == 2


def test_invoke_bills_both_calls():
    run, _ = _invoke([
        _Message([_TOOL_BLOCK], "tool_use", _Usage(1000, 100)),
        _Message([_Block("text")], "end_turn", _Usage(500, 50), _Answer(verdict="ok")),
    ])
    assert run.usage.input_tokens == 1500
    assert run.usage.output_tokens == 150


def test_invoke_refuses_when_no_tool_was_requested():
    run, seen = _invoke([_Message([_Block("text")], "end_turn", _Usage())])
    assert run.failed == "the model asked for no tools"
    assert len(seen) == 1


def test_invoke_translates_a_refusal_at_either_call():
    first, _ = _invoke([_Message([], "refusal", _Usage())])
    assert first.failed == "the model declined the request"
    second, _ = _invoke([
        _Message([_TOOL_BLOCK], "tool_use", _Usage()),
        _Message([], "refusal", _Usage(500, 50)),
    ])
    assert second.failed == "the model declined the request"
    assert second.usage.input_tokens == 1500


def test_invoke_translates_an_unparsable_second_response():
    run, _ = _invoke([
        _Message([_TOOL_BLOCK], "tool_use", _Usage()),
        _Message([_Block("text")], "end_turn", _Usage(500, 50), None),
    ])
    assert run.failed == "no parsable content in the response"


def test_invoke_never_sends_the_consequence_field_to_the_vendor():
    _, seen = _invoke([
        _Message([_TOOL_BLOCK], "tool_use", _Usage()),
        _Message([_Block("text")], "end_turn", _Usage(500, 50), _Answer(verdict="ok")),
    ])
    wire = seen[0][1]["tools"][0]
    assert "consequence" not in wire
    assert wire["strict"] is True


def test_invoke_rejects_a_non_pydantic_schema_as_programmer_error():
    completer = AnthropicCompleter(client=_StubClient([]))
    with pytest.raises(TypeError):
        completer.invoke(
            system="s", user="u", tools=(_SPEC,),
            execute=lambda c: ToolResult(c.call_id, ""), schema=dict,
        )
