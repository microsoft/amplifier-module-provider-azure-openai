"""Tests for llm:response event canonical usage keys (Task 5).

Verifies that the Azure OpenAI provider emits llm:response with canonical
usage keys:
  input_tokens, output_tokens, cache_read_tokens  (not input/output)

The fix is inherited from OpenAI provider: ChatResponse is built FIRST (via
_convert_to_chat_response), then the event is emitted from canonical
chat_response.usage fields.
"""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_azure_openai import _create_azure_provider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OrderRecordingHooks:
    """Hooks that record event names and payloads in emission order."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))

    def payload_for(self, event_name: str) -> dict | None:
        for name, payload in self.events:
            if name == event_name:
                return payload
        return None


class FakeCoordinator:
    def __init__(self):
        self.hooks = OrderRecordingHooks()


def _make_dummy_response(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cached_tokens: int | None = None,
) -> SimpleNamespace:
    """Create a minimal OpenAI Responses API stub with configurable usage."""
    usage_attrs: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cached_tokens is not None:
        usage_attrs["input_tokens_details"] = SimpleNamespace(
            cached_tokens=cached_tokens
        )

    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hello!")],
            )
        ],
        usage=SimpleNamespace(**usage_attrs),
        status="completed",
        id="resp_test",
        model_dump=lambda: {"id": "resp_test", "status": "completed"},
    )


def _make_provider():
    from amplifier_module_provider_openai import OpenAIProvider

    return _create_azure_provider(
        OpenAIProvider,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config={"max_retries": 0, "use_streaming": False},
    )


def _simple_request() -> ChatRequest:
    return ChatRequest(messages=[Message(role="user", content="Hello")])


# ---------------------------------------------------------------------------
# Tests: canonical usage keys in llm:response event
# ---------------------------------------------------------------------------


class TestLLMResponseEventCanonicalKeys:
    def test_event_usage_uses_input_tokens_key(self):
        """llm:response event usage must have 'input_tokens' key (not 'input')."""
        provider = _make_provider()
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None, "llm:response event must be emitted"
        usage = payload.get("usage", {})
        assert "input_tokens" in usage, (
            f"Expected 'input_tokens' key in usage, got keys: {list(usage.keys())}"
        )

    def test_event_usage_uses_output_tokens_key(self):
        """llm:response event usage must have 'output_tokens' key (not 'output')."""
        provider = _make_provider()
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None, "llm:response event must be emitted"
        usage = payload.get("usage", {})
        assert "output_tokens" in usage, (
            f"Expected 'output_tokens' key in usage, got keys: {list(usage.keys())}"
        )

    def test_event_usage_does_not_have_old_input_key(self):
        """llm:response event usage must NOT have old 'input' key."""
        provider = _make_provider()
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None, "llm:response event must be emitted"
        usage = payload.get("usage", {})
        assert "input" not in usage, (
            f"Old 'input' key must be removed, found it in usage: {usage}"
        )

    def test_event_usage_does_not_have_old_output_key(self):
        """llm:response event usage must NOT have old 'output' key."""
        provider = _make_provider()
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None, "llm:response event must be emitted"
        usage = payload.get("usage", {})
        assert "output" not in usage, (
            f"Old 'output' key must be removed, found it in usage: {usage}"
        )

    def test_event_usage_includes_cache_read_tokens_when_present(self):
        """llm:response event usage must include 'cache_read_tokens' when cache data is present."""
        provider = _make_provider()
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response(cached_tokens=800)
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None, "llm:response event must be emitted"
        usage = payload.get("usage", {})
        assert "cache_read_tokens" in usage, (
            f"Expected 'cache_read_tokens' key in usage, got keys: {list(usage.keys())}"
        )
        assert usage["cache_read_tokens"] == 800, (
            f"Expected cache_read_tokens=800, got {usage.get('cache_read_tokens')}"
        )

    def test_event_usage_input_tokens_value_correct(self):
        """llm:response event usage.input_tokens reflects actual API token count."""
        provider = _make_provider()
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response(input_tokens=150, output_tokens=75)
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None, "llm:response event must be emitted"
        usage = payload.get("usage", {})
        assert usage.get("input_tokens") == 150, (
            f"Expected input_tokens=150, got {usage.get('input_tokens')}"
        )
        assert usage.get("output_tokens") == 75, (
            f"Expected output_tokens=75, got {usage.get('output_tokens')}"
        )

    def test_provider_name_is_azure_openai(self):
        """llm:response event must report provider='azure-openai' (not 'openai')."""
        provider = _make_provider()
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
        provider.client.responses.create = AsyncMock(
            return_value=_make_dummy_response()
        )

        asyncio.run(provider.complete(_simple_request()))

        hooks = cast(FakeCoordinator, provider.coordinator).hooks
        payload = hooks.payload_for("llm:response")
        assert payload is not None, "llm:response event must be emitted"
        assert payload.get("provider") == "azure-openai", (
            f"Expected provider='azure-openai', got '{payload.get('provider')}'"
        )
