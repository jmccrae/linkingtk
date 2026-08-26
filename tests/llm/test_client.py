from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from linkingtk.exceptions import OptionalDependencyError
from linkingtk.llm.client import (
    AnthropicClient,
    LlmMessage,
    OpenAiClient,
    create_client,
)


class _FakeChoiceMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeChoiceMessage(content)


class _FakeOpenAiResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeOpenAiResponse:
        self.calls.append(kwargs)
        return _FakeOpenAiResponse(self.content)


class _FakeChat:
    def __init__(self, content: str | None) -> None:
        self.completions = _FakeCompletions(content)


class _FakeOpenAi:
    def __init__(self, content: str | None = "hello") -> None:
        self.chat = _FakeChat(content)


class TestOpenAiClient:
    def test_complete_returns_message_content(self) -> None:
        fake = _FakeOpenAi(content="the answer is yes")
        client = OpenAiClient("gpt-4o", client=fake)

        result = client.complete([LlmMessage(role="user", content="is this a match?")])

        assert result == "the answer is yes"
        call = fake.chat.completions.calls[0]  # type: ignore[attr-defined]
        assert call["model"] == "gpt-4o"
        assert call["messages"] == [{"role": "user", "content": "is this a match?"}]

    def test_complete_returns_empty_string_for_none_content(self) -> None:
        fake = _FakeOpenAi(content=None)
        client = OpenAiClient("gpt-4o", client=fake)

        assert client.complete([LlmMessage(role="user", content="hi")]) == ""

    def test_complete_structured_parses_json_and_sets_response_format(self) -> None:
        fake = _FakeOpenAi(content='{"match": true, "confidence": 0.9}')
        client = OpenAiClient("gpt-4o", client=fake)
        schema = {
            "type": "object",
            "properties": {"match": {"type": "boolean"}, "confidence": {"type": "number"}},
        }

        result = client.complete_structured(
            [LlmMessage(role="user", content="score this")], schema=schema
        )

        assert result == {"match": True, "confidence": 0.9}
        call = fake.chat.completions.calls[0]  # type: ignore[attr-defined]
        assert call["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema, "strict": True},
        }

    def test_complete_structured_rejects_non_object_json(self) -> None:
        fake = _FakeOpenAi(content="[1, 2, 3]")
        client = OpenAiClient("gpt-4o", client=fake)

        with pytest.raises(ValueError, match="JSON object"):
            client.complete_structured([LlmMessage(role="user", content="x")], schema={})

    def test_missing_openai_raises_optional_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "openai", None)

        with pytest.raises(OptionalDependencyError):
            OpenAiClient("gpt-4o")


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
        self.calls.append(kwargs)
        return _FakeAnthropicResponse(self.text)


class _FakeAnthropic:
    def __init__(self, text: str = "hello") -> None:
        self.messages = _FakeMessages(text)


class TestAnthropicClient:
    def test_complete_extracts_text_and_splits_system_message(self) -> None:
        fake = _FakeAnthropic(text="the answer is yes")
        client = AnthropicClient("claude-opus-4-6", client=fake)

        result = client.complete(
            [
                LlmMessage(role="system", content="You are a linker."),
                LlmMessage(role="user", content="is this a match?"),
            ]
        )

        assert result == "the answer is yes"
        call = fake.messages.calls[0]  # type: ignore[attr-defined]
        assert call["system"] == "You are a linker."
        assert call["messages"] == [{"role": "user", "content": "is this a match?"}]
        assert call["max_tokens"] == 1024

    def test_complete_omits_system_kwarg_when_no_system_message(self) -> None:
        fake = _FakeAnthropic()
        client = AnthropicClient("claude-opus-4-6", client=fake)

        client.complete([LlmMessage(role="user", content="hi")])

        assert "system" not in fake.messages.calls[0]  # type: ignore[attr-defined]

    def test_complete_structured_parses_json_and_sets_output_config(self) -> None:
        fake = _FakeAnthropic(text='{"match": false}')
        client = AnthropicClient("claude-opus-4-6", client=fake)
        schema = {"type": "object", "properties": {"match": {"type": "boolean"}}}

        result = client.complete_structured(
            [LlmMessage(role="user", content="score this")], schema=schema
        )

        assert result == {"match": False}
        call = fake.messages.calls[0]  # type: ignore[attr-defined]
        assert call["output_config"] == {"format": {"type": "json_schema", "schema": schema}}

    def test_complete_structured_rejects_non_object_json(self) -> None:
        fake = _FakeAnthropic(text="[1, 2, 3]")
        client = AnthropicClient("claude-opus-4-6", client=fake)

        with pytest.raises(ValueError, match="JSON object"):
            client.complete_structured([LlmMessage(role="user", content="x")], schema={})

    def test_missing_anthropic_raises_optional_dependency_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "anthropic", None)

        with pytest.raises(OptionalDependencyError):
            AnthropicClient("claude-opus-4-6")


class TestCreateClient:
    def test_openai_prefix(self) -> None:
        client = create_client("openai/gpt-4o", client=_FakeOpenAi())
        assert isinstance(client, OpenAiClient)
        assert client.model == "gpt-4o"

    def test_anthropic_prefix(self) -> None:
        client = create_client("anthropic/claude-opus-4-6", client=_FakeAnthropic())
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-opus-4-6"

    def test_ollama_prefix_defaults_base_url_and_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class _CapturingOpenAi:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)
                self.chat = _FakeChat("hello")

        module = types.ModuleType("openai")
        module.OpenAI = _CapturingOpenAi  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", module)

        client = create_client("ollama/llama3.1")

        assert isinstance(client, OpenAiClient)
        assert client.model == "llama3.1"
        assert captured["base_url"] == "http://localhost:11434/v1"
        assert captured["api_key"] == "ollama"

    def test_ollama_prefix_kwargs_override_defaults(self) -> None:
        client = create_client(
            "ollama/llama3.1", base_url="http://example.com/v1", client=_FakeOpenAi()
        )
        assert isinstance(client, OpenAiClient)

    def test_missing_separator_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            create_client("gpt-4o")

    def test_unknown_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_client("bedrock/some-model")
