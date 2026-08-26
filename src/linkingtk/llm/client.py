"""A small LLM client abstraction over the official `openai`/`anthropic` SDKs.

Ollama serves an OpenAI-compatible endpoint, so [OpenAiClient][linkingtk.llm.client.OpenAiClient]
covers both local dev (pointed at a local Ollama server) and real OpenAI-style
frontier APIs; [AnthropicClient][linkingtk.llm.client.AnthropicClient] covers
Claude specifically, whose Messages API has a different shape (a separate
`system` prompt, and its own native JSON-schema output mechanism rather than
OpenAI's `response_format`). [create_client][linkingtk.llm.client.create_client]
picks between them from a single `"<provider>/<model>"` spec, e.g.
`"ollama/llama3.1"` or `"anthropic/claude-opus-4-6"`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from linkingtk.exceptions import OptionalDependencyError

if TYPE_CHECKING:
    import anthropic
    import openai

LlmRole = Literal["system", "user", "assistant"]


@dataclass
class LlmMessage:
    """A single turn in a conversation passed to an [LlmClient][linkingtk.llm.client.LlmClient]."""

    role: LlmRole
    content: str


class LlmClient(ABC):
    """Common interface over a chat-completion-style LLM provider.

    `complete` returns free text; `complete_structured` constrains the
    response to a given JSON schema and returns it already parsed --
    prompting linkers (EA/EL/WSD/WSA) need a reliable decision/score back,
    not text to scrape.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[LlmMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Send `messages` and return the model's free-text reply."""

    @abstractmethod
    def complete_structured(
        self,
        messages: list[LlmMessage],
        *,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Send `messages` and return a reply constrained to and parsed from `schema`.

        `schema` is a JSON Schema object describing the expected response.
        """


def _to_openai_messages(messages: list[LlmMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class OpenAiClient(LlmClient):
    """[LlmClient][linkingtk.llm.client.LlmClient] backed by `openai.OpenAI`.

    Also the client for Ollama (via `base_url`/`api_key`, see `create_client`)
    since Ollama serves the same OpenAI-compatible Chat Completions wire
    format.

    Args:
        model: Model name, e.g. `"gpt-4o"` or, against an Ollama server,
            `"llama3.1"`.
        base_url: Overrides the API base, e.g. a local Ollama server.
            Defaults to the real OpenAI API.
        api_key: Forwarded to `openai.OpenAI`. Defaults to `None`, which
            lets the SDK fall back to its own `OPENAI_API_KEY` env var.
        max_retries: Forwarded to `openai.OpenAI` -- the SDK's own retry
            handling, not a custom loop.
        timeout: Forwarded to `openai.OpenAI`, in seconds.
        client: An already-constructed `openai.OpenAI` (or a test double
            matching its `chat.completions.create` surface) to use instead
            of constructing one from the arguments above.

    Raises:
        OptionalDependencyError: If `openai` isn't installed and `client`
            wasn't given.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
        client: openai.OpenAI | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError as exc:
                raise OptionalDependencyError("OpenAiClient", "llm") from exc
            self._client = openai.OpenAI(
                base_url=base_url, api_key=api_key, max_retries=max_retries, timeout=timeout
            )

    def complete(
        self,
        messages: list[LlmMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=_to_openai_messages(messages),  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def complete_structured(
        self,
        messages: list[LlmMessage],
        *,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        response = self._client.chat.completions.create(  # type: ignore[call-overload]
            model=self.model,
            messages=_to_openai_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            },
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object response, got: {content!r}")
        return result


def _split_system(messages: list[LlmMessage]) -> tuple[str | None, list[LlmMessage]]:
    """Pull out `role="system"` messages (joined) for Anthropic's separate `system` param."""
    system_parts = [m.content for m in messages if m.role == "system"]
    turns = [m for m in messages if m.role != "system"]
    return ("\n\n".join(system_parts) if system_parts else None), turns


class AnthropicClient(LlmClient):
    """[LlmClient][linkingtk.llm.client.LlmClient] backed by `anthropic.Anthropic`.

    Args:
        model: Model name, e.g. `"claude-opus-4-6"`.
        api_key: Forwarded to `anthropic.Anthropic`. Defaults to `None`,
            which lets the SDK fall back to its own `ANTHROPIC_API_KEY` env
            var.
        max_retries: Forwarded to `anthropic.Anthropic`.
        timeout: Forwarded to `anthropic.Anthropic`, in seconds.
        client: An already-constructed `anthropic.Anthropic` (or a test
            double matching its `messages.create` surface) to use instead of
            constructing one from the arguments above.

    Raises:
        OptionalDependencyError: If `anthropic` isn't installed and `client`
            wasn't given.

    Note:
        The current Anthropic Messages API has no `temperature` sampling
        parameter (it was replaced by reasoning-effort controls unrelated to
        sampling), so `complete`/`complete_structured`'s `temperature`
        argument is accepted for interface parity with `OpenAiClient` but
        not forwarded -- verified directly against the installed SDK's
        `messages.create` signature, not assumed.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as exc:
                raise OptionalDependencyError("AnthropicClient", "llm") from exc
            self._client = anthropic.Anthropic(
                api_key=api_key, max_retries=max_retries, timeout=timeout
            )

    def complete(
        self,
        messages: list[LlmMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            **_anthropic_kwargs(messages),
        )
        return _extract_text(response)

    def complete_structured(
        self,
        messages: list[LlmMessage],
        *,
        schema: dict[str, Any],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            **_anthropic_kwargs(messages),
        )
        text = _extract_text(response)
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError(f"Expected a JSON object response, got: {text!r}")
        return result


def _anthropic_kwargs(messages: list[LlmMessage]) -> dict[str, Any]:
    """Build `messages`/`system` kwargs for `anthropic.Anthropic.messages.create`.

    `system` is only included when there is one -- the SDK's own default
    (an `Omit` sentinel, not `None`) means passing `system=None` explicitly
    would be a real type/behavior mismatch, not just a no-op.
    """
    system, turns = _split_system(messages)
    kwargs: dict[str, Any] = {"messages": [{"role": m.role, "content": m.content} for m in turns]}
    if system is not None:
        kwargs["system"] = system
    return kwargs


def _extract_text(response: Any) -> str:
    """Concatenate the `text` of every text content block in an Anthropic `Message`.

    Duck-typed on `block.type == "text"` rather than `isinstance(block,
    anthropic.types.TextBlock)` so a test double's `messages.create` response
    doesn't need the real `anthropic` package importable -- matching
    `AnthropicClient(..., client=<test double>)`'s whole point.
    """
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def create_client(model: str, **kwargs: Any) -> LlmClient:
    """Build an [LlmClient][linkingtk.llm.client.LlmClient] from a `"<provider>/<model>"` spec.

    - `"openai/gpt-4o"` -- [OpenAiClient][linkingtk.llm.client.OpenAiClient]
      against the real OpenAI API.
    - `"anthropic/claude-opus-4-6"` --
      [AnthropicClient][linkingtk.llm.client.AnthropicClient].
    - `"ollama/llama3.1"` -- an `OpenAiClient` pointed at a local Ollama
      server's OpenAI-compatible endpoint (`base_url` defaults to
      `"http://localhost:11434/v1"`, `api_key` to a dummy value since Ollama
      doesn't check one -- both overridable via `kwargs`).

    Args:
        model: A `"<provider>/<model>"` spec, e.g. `"ollama/llama3.1"`.
        **kwargs: Forwarded to the chosen client's constructor.

    Raises:
        ValueError: If `model` doesn't start with a known provider prefix.
        OptionalDependencyError: If the chosen provider's SDK isn't installed.
    """
    provider, sep, name = model.partition("/")
    if not sep:
        raise ValueError(
            f"model must be '<provider>/<model>' (e.g. 'ollama/llama3.1'), got: {model!r}"
        )
    if provider == "openai":
        return OpenAiClient(name, **kwargs)
    if provider == "anthropic":
        return AnthropicClient(name, **kwargs)
    if provider == "ollama":
        kwargs.setdefault("base_url", "http://localhost:11434/v1")
        kwargs.setdefault("api_key", "ollama")
        return OpenAiClient(name, **kwargs)
    raise ValueError(
        f"Unknown LLM provider {provider!r} in model spec {model!r}; "
        "expected 'openai/...', 'anthropic/...', or 'ollama/...'"
    )
