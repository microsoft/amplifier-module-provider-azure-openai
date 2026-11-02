"""Mount rollback regressions using the installed Core coordinator."""

import asyncio

import httpx
import pytest
import pytest_asyncio
from amplifier_core import AmplifierSession
from amplifier_module_provider_azure_openai import decisions, mount


@pytest_asyncio.fixture
async def coordinator():
    session = AmplifierSession(
        {
            "session": {
                "orchestrator": {"module": "unused-fixture-loop"},
                "context": {"module": "unused-fixture-context"},
            },
            "providers": [],
        }
    )
    try:
        yield session.coordinator
    finally:
        await session.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["second_registration", "later_cost_contributor"])
@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("replace", [False, True])
async def test_failed_mount_releases_only_owned_decisions(
    coordinator, monkeypatch, phase, error, replace
):
    unrelated, replacement = object(), object()
    coordinator.register_capability("decision.backends", {"unrelated": unrelated})
    transports = []
    original_transport = decisions.FoundryDecisions

    class CapturedTransport(original_transport):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            def forbidden(request):
                raise AssertionError(
                    "No network request is authorized by this offline test"
                )

            self._client = httpx.AsyncClient(transport=httpx.MockTransport(forbidden))
            transports.append(self)

    monkeypatch.setattr(decisions, "FoundryDecisions", CapturedTransport)
    original = coordinator.register_contributor

    def failing_contributor(channel, identity, callback):
        if (phase == "second_registration" and identity == "second") or (
            phase == "later_cost_contributor"
            and channel == "session.cost"
            and identity == "provider-azure-openai"
        ):
            if replace:
                rows = dict(coordinator.get_capability("decision.backends"))
                rows["first"] = replacement
                coordinator.register_capability("decision.backends", rows)
            raise error("synthetic failed mount")
        return original(channel, identity, callback)

    monkeypatch.setattr(coordinator, "register_contributor", failing_contributor)
    try:
        with pytest.raises(error):
            await mount(
                coordinator,
                {
                    "azure_endpoint": "https://fixture.services.ai.azure.com",
                    "api_key": "synthetic-only",
                    "decisions": {
                        "enabled": True,
                        "backends": {
                            "first": {"enabled": True, "model": "first"},
                            "second": {"enabled": True, "model": "second"},
                        },
                    },
                },
            )
        current = coordinator.get_capability("decision.backends")
        assert current == (
            {"unrelated": unrelated, "first": replacement}
            if replace
            else {"unrelated": unrelated}
        ), "Failed mount left an owned decision backend executable"
        assert all(t.backend.closed and t._client.is_closed for t in transports), (
            "Failed mount leaked backend/transport"
        )
    finally:
        # Review fixture owns its captured objects even when product rollback fails.
        for transport in transports:
            await transport.close()
        coordinator.register_capability("decision.backends", {"unrelated": unrelated})


@pytest.mark.asyncio
async def test_constructor_failure_closes_account_and_preserves_cancellation(
    coordinator, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import amplifier_module_provider_azure_openai as azure

    closed = AsyncMock(side_effect=RuntimeError("synthetic cleanup failure"))
    account = SimpleNamespace(
        base_url="https://fixture.openai.azure.com",
        api_version="fixture",
        auth_summary="fixture",
        _api_key="synthetic",
        _token_provider=None,
        config={},
        close=closed,
    )
    monkeypatch.setattr(azure, "_resolve_azure_account", lambda config: account)
    cancellation = asyncio.CancelledError("synthetic constructor cancellation")

    def fail(*args, **kwargs):
        raise cancellation

    monkeypatch.setattr(azure, "_create_azure_provider", fail)
    with pytest.raises(asyncio.CancelledError) as caught:
        await mount(coordinator, {})
    assert caught.value is cancellation
    closed.assert_awaited_once()
    assert any("RuntimeError" in note for note in cancellation.__notes__)
