"""The single seam between this book and the Anthropic SDK.

Every rung from Chapter 4 onward asks for model output through `Completer`
rather than importing `anthropic` directly. Three reasons, all of them paid for
later: the SDK pin will go stale, Chapter 8 adds a framework that churns faster
than the SDK does, and every rung above Level 1 has to be testable without a
key, without a network, and without spending money to run a book example.

`Completion` carries no exceptions in its contract. A rung asks for typed
output and gets back either the output or a reason, so nothing above this
module ever catches a vendor exception.

Chapter 7 adds the second method, `invoke`, and widening a protocol is a change
to every implementation of it - including the ones earlier chapters already
shipped. That is the argument for keeping a seam narrow stated as a cost rather
than as a preference. `invoke` runs exactly one round of tool calls and does not
execute anything itself: the caller passes the function that runs a tool, so the
decision about what a tool is permitted to do stays in the module that owns the
consequences.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from escalation_ladder.instrument import CostLedger, measured

MODEL = "claude-opus-5"
MAX_TOKENS = 1024

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Usage:
    """Token counts for one call, named to match the SDK's own field.

    `instrument.measured` reads `.usage.input_tokens` off whatever a rung
    returns, so matching the names is what lets one decorator price every rung
    in the book without knowing anything about this module.
    """

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Completion(Generic[T]):
    """One call's typed result, its bill, and why it failed if it did."""

    parsed: T | None
    usage: Usage
    failed: str | None = None

    @property
    def ok(self) -> bool:
        return self.parsed is not None and self.failed is None


@dataclass(frozen=True)
class ToolSpec:
    """One tool as the model sees it, plus the one field the model never sees.

    `consequence` has no default. A tool cannot be defined in this book without
    someone answering the question, which is the entire mechanism behind
    Chapter 7's Blast Radius Schema: the vendor is never told this field exists,
    and it is the field that decides what the vendor's model can cause.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the tool's input
    consequence: Literal["read", "write"]


@dataclass(frozen=True)
class ToolCall:
    """One tool the model asked for, and the arguments it chose."""

    call_id: str
    name: str
    arguments: dict[str, Any]

    def render(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.arguments.items()))
        return f"{self.name}({args})"


@dataclass(frozen=True)
class ToolResult:
    """What executing one tool produced, as the model will read it."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ToolRun(Generic[T]):
    """One round of tool calls, the answer built from them, and what it cost.

    `wanted_more` is the load-bearing field. Tools stay available on the second
    request, so a model that needs another round can ask for one - and this
    records that it did instead of quietly running it. A seam that dropped the
    tools before the final request would make that signal unobservable, which is
    the difference between issuing a Failure Receipt and asserting one.
    """

    parsed: T | None
    usage: Usage
    calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResult, ...] = ()
    failed: str | None = None
    wanted_more: bool = False

    @property
    def ok(self) -> bool:
        return self.parsed is not None and self.failed is None


Execute = Callable[[ToolCall], ToolResult]


class Completer(Protocol):
    """What a rung is allowed to ask a model for.

    Deliberately narrow. There is no `temperature`, no message history, and no
    streaming, because no rung in this book needs them - and a seam that
    exposes everything the SDK exposes is not a seam.
    """

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "low",
    ) -> Completion[T]: ...

    def invoke(
        self,
        *,
        system: str,
        user: str,
        tools: tuple[ToolSpec, ...],
        execute: Execute,
        schema: type[T],
        effort: str = "low",
    ) -> ToolRun[T]: ...


def _first(exc: ValidationError) -> str:
    """The first validation error, as one short line."""
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"]) or "<root>"
    return f"{location}: {error['msg']}"


def _api_error(exc: anthropic.APIError) -> str:
    """The exception class AND what the vendor actually said.

    The class alone is not a reason. A measurement run once returned 27
    `BadRequestError`s that were indistinguishable from each other and from a
    genuine refusal, because this seam threw the message away and kept the type
    name. Translating an exception into a string is the right contract;
    translating it into a string nobody can act on is this chapter's own
    complaint about opaque tool errors, aimed at the wrong layer.
    """
    detail = " ".join(str(exc).split())
    if len(detail) > 160:
        detail = detail[:157] + "..."
    return f"api error: {type(exc).__name__}: {detail}" if detail else (
        f"api error: {type(exc).__name__}"
    )


def _wire(spec: ToolSpec) -> dict[str, Any]:
    """A ToolSpec as the API wants it. `consequence` is deliberately not sent.

    `strict` is what makes the arguments trustworthy at all. Without it the
    input schema is advisory in exactly the way Chapter 4 measured - shape
    demoted into a description, enforced client-side after the tokens are
    billed. With it, `tool_use.input` is guaranteed to validate, so a tool
    implementation never has to defend against a missing required field.
    """
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.parameters,
        "strict": True,
    }


def _calls_from(content: Any) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(block.id, block.name, dict(block.input))
        for block in content
        if getattr(block, "type", None) == "tool_use"
    )


def _result_blocks(results: tuple[ToolResult, ...]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for result in results:
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": result.call_id,
            "content": result.content,
        }
        if result.is_error:
            block["is_error"] = True
        blocks.append(block)
    return blocks


class AnthropicCompleter:
    """The real thing. Wraps `client.messages.parse`, and nothing else."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = MODEL,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        self._client = client if client is not None else anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "low",
    ) -> Completion[T]:
        # Checked before the call so that the TypeError arm below can only mean
        # credentials. The SDK also raises TypeError when it cannot build a schema
        # from `output_format`, and reporting that as a missing key would send a
        # programmer chasing their environment. This one stays an exception:
        # per the book's convention, raising is reserved for programmer error.
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(
                f"schema must be a pydantic BaseModel subclass, got {schema!r}"
            )
        try:
            message = self._client.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                # `output_format` and `output_config` compose: the SDK merges
                # the schema in under a "format" key rather than replacing the
                # dict, so setting effort does not cost us the schema.
                output_format=schema,
                output_config={"effort": effort},
            )
        except ValidationError as exc:
            # A client-side rule was violated. The tokens were spent and the
            # ledger will never see them, so count these separately.
            return Completion(None, Usage(), f"schema violated: {_first(exc)}")
        except anthropic.APIError as exc:
            return Completion(None, Usage(), _api_error(exc))
        except TypeError:
            # No resolvable credentials. The SDK raises a bare TypeError for this
            # rather than an APIError, and it does it at request time rather than
            # at construction, so without this arm the most common reader
            # configuration - a fresh clone and no key - escapes the seam as a
            # traceback from inside this file. The catch is deliberately scoped
            # to the single SDK call above.
            return Completion(
                None, Usage(), "no credentials: set ANTHROPIC_API_KEY"
            )

        usage = Usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
        # A decline is a successful HTTP response, not an exception, so
        # `stop_reason` has to be read before the content is trusted.
        if message.stop_reason == "refusal":
            return Completion(None, usage, "the model declined the request")
        if message.parsed_output is None:
            return Completion(None, usage, "no parsable content in the response")
        return Completion(message.parsed_output, usage)


    def invoke(
        self,
        *,
        system: str,
        user: str,
        tools: tuple[ToolSpec, ...],
        execute: Execute,
        schema: type[T],
        effort: str = "low",
    ) -> ToolRun[T]:
        """Two requests, one round of tools in between, and no third request.

        The structure is the architecture. There is no loop here and no
        iteration cap standing in for one: the model picks its tools once, every
        result comes back at once, and the next response is taken as final. That
        is what keeps this rung on Level 4 rather than being a Level 6 agent
        with `max_iterations=1`, which Chapter 2 already rejected as a
        distinction without a difference.
        """
        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise TypeError(
                f"schema must be a pydantic BaseModel subclass, got {schema!r}"
            )
        wire = [_wire(spec) for spec in tools]
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]

        try:
            first = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
                tools=wire,
                output_config={"effort": effort},
            )
        except anthropic.APIError as exc:
            return ToolRun(None, Usage(), failed=_api_error(exc))
        except TypeError:
            return ToolRun(None, Usage(), failed="no credentials: set ANTHROPIC_API_KEY")

        spent = Usage(first.usage.input_tokens, first.usage.output_tokens)
        if first.stop_reason == "refusal":
            return ToolRun(None, spent, failed="the model declined the request")

        calls = _calls_from(first.content)
        if not calls:
            # Not an error at the vendor and not an answer at this rung. A
            # finding with no tool result behind it is a Level 1 answer wearing
            # a Level 4 cost, so it is refused rather than returned.
            return ToolRun(None, spent, failed="the model asked for no tools")

        results = tuple(execute(call) for call in calls)
        messages.append({"role": "assistant", "content": first.content})
        messages.append({"role": "user", "content": _result_blocks(results)})

        try:
            second = self._client.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
                # Tools stay on the request. Removing them here would force an
                # answer and destroy the only evidence that one round was not
                # enough - see `ToolRun.wanted_more`.
                tools=wire,
                output_format=schema,
                output_config={"effort": effort},
            )
        except ValidationError as exc:
            return ToolRun(
                None, spent, calls, results, f"schema violated: {_first(exc)}"
            )
        except anthropic.APIError as exc:
            return ToolRun(None, spent, calls, results, _api_error(exc))

        spent = Usage(
            spent.input_tokens + second.usage.input_tokens,
            spent.output_tokens + second.usage.output_tokens,
        )
        if second.stop_reason == "refusal":
            return ToolRun(
                None, spent, calls, results, "the model declined the request"
            )
        if _calls_from(second.content):
            return ToolRun(
                None,
                spent,
                calls,
                results,
                "one round of tools was not enough",
                wanted_more=True,
            )
        if second.parsed_output is None:
            return ToolRun(
                None, spent, calls, results, "no parsable content in the response"
            )
        return ToolRun(second.parsed_output, spent, calls, results)


def prompt_key(*, system: str, user: str) -> str:
    """A stable short name for one exact prompt."""
    material = f"{system}\x00{user}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:12]


# Recorded in place of an answer when the model asked for a second round.
WANTED_MORE = "__wanted_more__"


def answer_key(
    *, system: str, user: str, results: tuple[ToolResult, ...]
) -> str:
    """A stable short name for the second request in one round.

    Keyed on the tool results as well as the prompt, because that request is
    the only one whose input the model did not entirely determine. A recording
    keyed on the prompt alone would keep replaying an answer that was produced
    against different numbers, which is the same defect Chapter 6 hit when its
    own tooling orphaned a stage.
    """
    blob = "\x00".join(
        f"{result.call_id}|{int(result.is_error)}|{result.content}"
        for result in results
    )
    return prompt_key(system=system, user=f"{user}\x00{blob}")


@dataclass
class RecordedCompleter:
    """Replays recorded output. Deterministic, no key, no spend.

    Keyed on a hash of the exact prompt, so editing the system prompt makes
    every recording miss. That is the intended behavior rather than an
    inconvenience: a prompt change invalidates the evidence gathered under the
    old prompt, and a fake that quietly kept passing would hide it.
    """

    recordings: dict[str, str]
    usage: Usage = Usage()

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "low",
    ) -> Completion[T]:
        key = prompt_key(system=system, user=user)
        if key not in self.recordings:
            return Completion(None, Usage(), f"no recording for prompt {key}")
        try:
            parsed = schema.model_validate_json(self.recordings[key])
        except ValidationError as exc:
            return Completion(None, self.usage, f"schema violated: {_first(exc)}")
        return Completion(parsed, self.usage)

    def invoke(
        self,
        *,
        system: str,
        user: str,
        tools: tuple[ToolSpec, ...],
        execute: Execute,
        schema: type[T],
        effort: str = "low",
    ) -> ToolRun[T]:
        """Replay a recorded round, executing the tools for real.

        The tools genuinely run. They are deterministic fixtures, so replaying
        the model's choices and recomputing the results is free, exact, and
        catches the failure a recorded result would hide: a tool whose output
        drifted away from the answer that was recorded against it.
        """
        first = prompt_key(system=system, user=user)
        if first not in self.recordings:
            return ToolRun(None, Usage(), failed=f"no recording for prompt {first}")
        calls = tuple(
            ToolCall(entry["call_id"], entry["name"], entry["arguments"])
            for entry in json.loads(self.recordings[first])
        )
        if not calls:
            return ToolRun(None, self.usage, failed="the model asked for no tools")

        results = tuple(execute(call) for call in calls)
        second = answer_key(system=system, user=user, results=results)
        if second not in self.recordings:
            return ToolRun(
                None, self.usage, calls, results, f"no recording for answer {second}"
            )
        body = self.recordings[second]
        if body == WANTED_MORE:
            return ToolRun(
                None,
                self.usage,
                calls,
                results,
                "one round of tools was not enough",
                wanted_more=True,
            )
        try:
            parsed = schema.model_validate_json(body)
        except ValidationError as exc:
            return ToolRun(
                None, self.usage, calls, results, f"schema violated: {_first(exc)}"
            )
        return ToolRun(parsed, self.usage, calls, results)


@dataclass
class MeteredCompleter:
    """A `Completer` that bills every call to a ledger under the current label.

    Moved here from `workflow.py` in Chapter 7, because it is an implementation
    of this protocol and two rungs now need it - leaving it where it was would
    have made Level 4 import from Level 3 for a utility that has nothing to do
    with pipelines. `workflow` re-exports it, so nothing above changed.

    It wraps the seam rather than each caller, which is why `classify.classify`
    became measurable inside Chapter 6's pipeline with no edit to `classify.py`,
    and why every tool call in Chapter 7 is priced without `tools.py` knowing
    that anything is being measured.
    """

    inner: Completer
    ledger: CostLedger
    stage: str = "call"

    def parse(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        effort: str = "low",
    ) -> Completion[T]:
        billed = measured(self.stage, self.ledger)(self.inner.parse)
        return billed(system=system, user=user, schema=schema, effort=effort)

    def invoke(
        self,
        *,
        system: str,
        user: str,
        tools: tuple[ToolSpec, ...],
        execute: Execute,
        schema: type[T],
        effort: str = "low",
    ) -> ToolRun[T]:
        billed = measured(self.stage, self.ledger)(self.inner.invoke)
        return billed(
            system=system,
            user=user,
            tools=tools,
            execute=execute,
            schema=schema,
            effort=effort,
        )
