import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import Message
from amplifier_core.message_models import ToolCallBlock
from amplifier_module_provider_azure_openai import AzureOpenAIProvider
from amplifier_module_provider_azure_openai import _create_azure_provider
from amplifier_module_provider_azure_openai import mount


class DummyResponse:
    """Minimal response stub mirroring OpenAI responses."""

    def __init__(self, output=None):
        self.output = output or []
        self.usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
        self.stop_reason = "stop"


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


class MockOpenAIProvider:
    """Minimal stand-in for OpenAIProvider to test Azure-specific overrides."""

    def __init__(self, *, api_key=None, config=None, coordinator=None, client=None):
        self._api_key = api_key
        self.config = config or {}
        self.coordinator = coordinator


def test_extended_thinking_matches_openai_behaviour():
    provider = AzureOpenAIProvider(
        base_url="https://example", api_key="test-key", config={"max_tokens": 1024}
    )
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    messages = [Message(role="user", content="Hello")]
    request = ChatRequest(messages=messages)

    asyncio.run(
        provider.complete(request, extended_thinking=True, thinking_budget_tokens=6000)
    )

    provider.client.responses.create.assert_awaited()
    call_kwargs = provider.client.responses.create.await_args_list[0].kwargs

    assert call_kwargs["reasoning"]["effort"] == "high"
    assert call_kwargs["max_output_tokens"] == 7024


def test_tool_call_repair_emits_azure_provider_name():
    """Azure provider repairs missing tool results and emits correct provider name."""
    provider = AzureOpenAIProvider(base_url="https://example", api_key="test-key")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="do_something", input={"value": 1})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    # Should repair and succeed (not raise)
    asyncio.run(provider.complete(request))

    provider.client.responses.create.assert_awaited_once()

    # Should emit repair event with azure-openai provider name
    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["provider"] == "azure-openai"
    assert repair_events[0][1]["repair_count"] == 1


def test_mount_returns_cleanup_and_closes_client():
    class StubCoordinator:
        def __init__(self):
            self.mounted: list[tuple[str, AzureOpenAIProvider, str]] = []
            self.hooks = FakeHooks()

        async def mount(
            self, slot: str, provider: AzureOpenAIProvider, name: str
        ) -> None:
            self.mounted.append((slot, provider, name))

    coordinator = StubCoordinator()
    cleanup = asyncio.run(
        mount(
            cast(ModuleCoordinator, coordinator),
            {
                "azure_endpoint": "https://example-resource.openai.azure.com",
                "api_key": "test-key",
            },
        )
    )

    assert cleanup is not None
    assert coordinator.mounted
    slot, provider, name = coordinator.mounted[0]
    assert slot == "providers"
    assert name == "azure-openai"

    provider.client.close = AsyncMock()
    asyncio.run(cleanup())
    provider.client.close.assert_awaited_once()


def test_incomplete_tool_call_removed_for_azure():
    provider = AzureOpenAIProvider(base_url="https://example", api_key="test-key")
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    messages = [
        Message(role="user", content="start"),
        Message(
            role="assistant",
            content=[ToolCallBlock(id="call_az_1", name="azure_tool", input={})],
        ),
        Message(role="user", content="follow-up"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    provider.client.responses.create.assert_awaited_once()


def test_azure_client_sdk_retries_disabled():
    """Azure OpenAI client must be created with max_retries=0.
    Amplifier manages retries via retry_with_backoff. Without max_retries=0,
    the SDK default (2) causes up to 18 HTTP requests instead of 6.
    """
    provider = _create_azure_provider(
        MockOpenAIProvider,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
    )
    assert provider.client.max_retries == 0


def test_token_provider_callable_passed_as_api_key():
    """When api_key is None and token_provider is an async callable, the callable
    is passed directly to AsyncOpenAI as api_key.
    This works because the OpenAI SDK (>= 1.0) natively supports:
        api_key: str | Callable[[], Awaitable[str]] | None
    The SDK stores the callable and calls it per-request for token refresh.
    """

    async def fake_token_provider() -> str:
        return "test-token"

    with patch("amplifier_module_provider_azure_openai.AsyncOpenAI") as MockAsyncOpenAI:
        MockAsyncOpenAI.return_value = MagicMock()
        provider = _create_azure_provider(
            MockOpenAIProvider,
            base_url="https://example.openai.azure.com/openai/v1/",
            token_provider=fake_token_provider,
        )
        _ = provider.client  # Triggers lazy client creation

        MockAsyncOpenAI.assert_called_once_with(
            base_url="https://example.openai.azure.com/openai/v1/",
            api_key=fake_token_provider,
            max_retries=0,
        )
