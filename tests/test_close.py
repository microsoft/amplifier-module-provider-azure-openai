"""Tests for the close() method on the Azure OpenAI provider."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from amplifier_core import ModuleCoordinator
from amplifier_module_provider_azure_openai import _create_azure_provider
from amplifier_module_provider_azure_openai import mount


class MockOpenAIProvider:
    """Minimal stand-in for OpenAIProvider base class."""

    def __init__(self, *, api_key=None, config=None, coordinator=None, client=None):
        self._api_key = api_key
        self.config = config or {}
        self.coordinator = coordinator


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()
        self.mounted: list = []

    async def mount(self, slot: str, provider, name: str) -> None:
        self.mounted.append((slot, provider, name))


def _make_provider(
    base_url: str = "https://example.openai.azure.com/openai/v1/",
    api_key: str = "fake-key",
):
    """Create an Azure OpenAI provider using the dynamic class factory."""
    return _create_azure_provider(
        MockOpenAIProvider, base_url=base_url, api_key=api_key
    )


class TestAzureOpenAIProviderClose:
    """Tests for the async close() method on the dynamic _AzureOpenAIProvider class."""

    def test_close_calls_client_close_when_initialized(self):
        """close() should call _azure_client.close() and nil the reference."""
        provider = _make_provider()

        # Simulate an initialized client
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        provider._azure_client = mock_client

        asyncio.run(provider.close())

        mock_client.close.assert_awaited_once()
        assert provider._azure_client is None

    def test_close_is_safe_when_client_is_none(self):
        """close() should not crash when _azure_client is None."""
        provider = _make_provider()

        # Confirm client is None before close
        assert provider._azure_client is None

        # Should not raise
        asyncio.run(provider.close())

        assert provider._azure_client is None

    def test_close_can_be_called_twice(self):
        """close() called twice should only close the client once."""
        provider = _make_provider()

        # Simulate an initialized client
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        provider._azure_client = mock_client

        asyncio.run(provider.close())
        asyncio.run(provider.close())

        mock_client.close.assert_awaited_once()
        assert provider._azure_client is None


class TestMountCleanupBugFix:
    """Tests for the mount() cleanup function not triggering lazy client init."""

    def test_mount_cleanup_does_not_trigger_lazy_init(self):
        """Calling the cleanup function should not create a client via .client property."""
        coordinator = FakeCoordinator()

        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
                "AZURE_OPENAI_API_KEY": "fake-key",
            },
        ):
            cleanup_ref = asyncio.run(mount(cast(ModuleCoordinator, coordinator), {}))

        assert cleanup_ref is not None

        # Get the provider that was mounted
        _, provider, _ = coordinator.mounted[0]

        # Confirm no client exists before cleanup
        assert provider._azure_client is None

        # Calling cleanup should NOT raise or create a client
        # Before the fix, cleanup accessed provider.client which triggers lazy init
        asyncio.run(cleanup_ref())

        # After cleanup, _azure_client should still be None (not lazily initialized)
        assert provider._azure_client is None
