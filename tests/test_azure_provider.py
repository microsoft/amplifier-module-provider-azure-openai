import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_module_provider_azure_openai import AzureOpenAIProvider
from amplifier_module_provider_azure_openai import mount


class DummyResponse:
    """Minimal response stub mirroring OpenAI responses."""

    def __init__(self, output=None):
        self.output = output or []
        self.usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        self.stop_reason = "stop"


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def test_extended_thinking_matches_openai_behaviour():
    provider = AzureOpenAIProvider(base_url="https://example", api_key="test-key", config={"max_tokens": 1024})
    provider.client.responses.create = AsyncMock(return_value=DummyResponse())

    messages = [{"role": "user", "content": "Hello"}]

    asyncio.run(provider.complete(messages, extended_thinking=True, thinking_budget_tokens=6000))

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
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "function": {"name": "do_something", "arguments": '{"value": 1}'}},
            ],
        },
        {"role": "user", "content": "No tool result present"},
    ]

    # Should repair and succeed (not raise)
    asyncio.run(provider.complete(messages))

    provider.client.responses.create.assert_awaited_once()

    # Should emit repair event with azure-openai provider name
    repair_events = [e for e in fake_coordinator.hooks.events if e[0] == "provider:tool_sequence_repaired"]
    assert len(repair_events) == 1
    assert repair_events[0][1]["provider"] == "azure-openai"
    assert repair_events[0][1]["repair_count"] == 1


def test_mount_returns_cleanup_and_closes_client():
    class StubCoordinator:
        def __init__(self):
            self.mounted: list[tuple[str, AzureOpenAIProvider, str]] = []
            self.hooks = FakeHooks()

        async def mount(self, slot: str, provider: AzureOpenAIProvider, name: str) -> None:
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
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_az_1", "function": {"name": "azure_tool", "arguments": "{}"}},
            ],
        },
        {"role": "user", "content": "follow-up"},
    ]

    asyncio.run(provider.complete(messages))

    provider.client.responses.create.assert_awaited_once()
