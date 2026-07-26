"""The single seam between this book and the Anthropic SDK.

Every rung from Chapter 4 onward asks for model output through `Completer`
rather than importing `anthropic` directly. Three reasons, all of them paid for
later: the SDK pin will go stale, Chapter 8 adds a framework that churns faster
than the SDK does, and every rung above Level 1 has to be testable without a
key, without a network, and without spending money to run a book example.

`Completion` carries no exceptions in its contract. A rung asks for typed
output and gets back either the output or a reason, so nothing above this
module ever catches a vendor exception.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

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


def _first(exc: ValidationError) -> str:
    """The first validation error, as one short line."""
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"]) or "<root>"
    return f"{location}: {error['msg']}"


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
            return Completion(None, Usage(), f"api error: {type(exc).__name__}")
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


def prompt_key(*, system: str, user: str) -> str:
    """A stable short name for one exact prompt."""
    material = f"{system}\x00{user}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:12]


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
