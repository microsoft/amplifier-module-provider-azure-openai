"""Tests for the close() method on the Azure OpenAI provider."""

import asyncio
import logging
import time
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from amplifier_core import ModuleCoordinator
from amplifier_module_provider_azure_openai import _create_azure_provider
from amplifier_module_provider_azure_openai import mount


class MockOpenAIProvider:
    """Minimal stand-in for OpenAIProvider base class."""

    def __init__(self, *, api_key=None, config=None, coordinator=None, client=None, add_cost=None):
        self._api_key = api_key
        self.config = config or {}
        self.coordinator = coordinator
        self._add_cost = add_cost or (lambda cost: None)


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def register(self, event: str, handler) -> None:
        pass  # no-op — tests that call mount() don't need hook delivery

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()
        self.mounted: list = []

    async def mount(self, slot: str, provider, name: str) -> None:
        self.mounted.append((slot, provider, name))

    def register_contributor(self, *args, **kwargs) -> None:
        pass  # no-op — tests that call mount() don't need contributor delivery


def _make_provider(
    base_url: str = "https://example.openai.azure.com/openai/v1/",
    api_key: str = "fake-key",
    config: dict | None = None,
):
    """Create an Azure OpenAI provider using the dynamic class factory."""
    return _create_azure_provider(
        MockOpenAIProvider, base_url=base_url, api_key=api_key, config=config
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

    def test_close_is_bounded_when_client_close_never_returns(self, caplog):
        """A client whose close() never returns must not hang cleanup.

        Regression guard: close() previously awaited
        ``self._azure_client.close()`` with no ceiling, and mount()'s
        cleanup() ran its own equally unbounded copy -- so a wedged httpx
        transport hung session cleanup for the whole process.
        """
        provider = _make_provider(config={"close_timeout": 0.05})
        assert provider._close_timeout == 0.05

        async def scenario():
            release = asyncio.Event()

            class _UnclosableClient:
                async def close(self):
                    # Never returns until the test explicitly releases it.
                    await release.wait()

            provider._azure_client = _UnclosableClient()

            started = time.monotonic()
            await provider.close()  # must not raise, must not hang
            elapsed = time.monotonic() - started

            # Let the abandoned close task finish so the loop shuts down clean.
            release.set()
            await asyncio.sleep(0)
            return elapsed

        with caplog.at_level(logging.WARNING):
            elapsed = asyncio.run(scenario())

        assert elapsed < 2.0, f"close() took {elapsed:.2f}s; expected ~0.05s"
        assert "did not complete within" in caplog.text
        assert "abandoning client" in caplog.text
        assert "azure-openai" in caplog.text
        # Client reference dropped so the lazy-init property can rebuild.
        assert provider._azure_client is None

    def test_close_normal_client_logs_no_warning(self, caplog):
        """A well-behaved client closes once, quietly, and is released."""
        provider = _make_provider()
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        provider._azure_client = mock_client

        with caplog.at_level(logging.WARNING):
            asyncio.run(provider.close())

        mock_client.close.assert_awaited_once()
        assert caplog.text == ""
        assert provider._azure_client is None

    def test_close_timeout_defaults_to_five_seconds(self):
        """Unconfigured providers get the 5.0s default ceiling."""
        assert _make_provider()._close_timeout == 5.0

    def test_close_timeout_coerces_string_and_falls_back_on_garbage(self, caplog):
        """settings.yaml strings coerce; garbage warns and uses the default."""
        assert _make_provider(config={"close_timeout": "2.5"})._close_timeout == 2.5

        with caplog.at_level(logging.WARNING):
            provider = _make_provider(config={"close_timeout": "not-a-number"})
        assert provider._close_timeout == 5.0
        assert "close_timeout" in caplog.text


class TestMountCleanupIsBounded:
    """mount()'s cleanup() must inherit close()'s time bound."""

    def test_mount_cleanup_is_bounded(self, caplog):
        """cleanup() used to carry its own unbounded copy of the close."""
        coordinator = FakeCoordinator()

        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
                "AZURE_OPENAI_API_KEY": "fake-key",
            },
        ):
            cleanup_ref = asyncio.run(
                mount(
                    cast(ModuleCoordinator, coordinator),
                    {"close_timeout": 0.05},
                )
            )

        assert cleanup_ref is not None
        _, provider, _ = coordinator.mounted[0]

        async def scenario():
            release = asyncio.Event()

            class _UnclosableClient:
                async def close(self):
                    await release.wait()

            provider._azure_client = _UnclosableClient()
            started = time.monotonic()
            await cleanup_ref()
            elapsed = time.monotonic() - started
            release.set()
            await asyncio.sleep(0)
            return elapsed

        with caplog.at_level(logging.WARNING):
            elapsed = asyncio.run(scenario())

        assert elapsed < 2.0, f"cleanup() took {elapsed:.2f}s; expected ~0.05s"
        assert "abandoning client" in caplog.text
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

        # cleanup() delegates to the bounded close(); with no client ever
        # built, that is a no-op rather than a construct-then-destroy.

        # After cleanup, _azure_client should still be None (not lazily initialized)
        assert provider._azure_client is None
