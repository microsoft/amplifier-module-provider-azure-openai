"""Tests for _accumulate hook, register_contributor, and PTU short-circuit in mount().

Verifies that mount() registers:
  - an `llm:response` hook (_accumulate) that sums cost_usd into a closure-captured dict
  - a lazy contributor callback on session.cost channel under name 'provider-azure-openai'
  - PTU deployments short-circuit to cost_usd = None
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from amplifier_module_provider_azure_openai import mount


# ---------------------------------------------------------------------------
# Mock coordinator fixture
# ---------------------------------------------------------------------------


class _MockHooks:
    def __init__(self):
        self._handlers: dict = {}

    def register(self, event: str, handler) -> None:
        self._handlers[event] = handler

    async def emit(self, event: str, data: dict) -> None:
        if event in self._handlers:
            await self._handlers[event](event, data)


class _MockCoordinator:
    def __init__(self):
        self.hooks = _MockHooks()
        self.registered_hooks = self.hooks._handlers  # shared reference
        self.registered_contributors: dict = {}

    async def mount(self, *args, **kwargs) -> None:
        pass

    def register_contributor(self, channel: str, name: str, callback) -> None:
        self.registered_contributors[(channel, name)] = callback

    def get_capability(self, *args, **kwargs):
        return None


@pytest.fixture
def mock_coordinator():
    return _MockCoordinator()


# ---------------------------------------------------------------------------
# test_contributor_registered_at_mount
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_registered_at_mount(mock_coordinator):
    """mount() must register a contributor on ('session.cost', 'provider-azure-openai')."""
    await mount(
        mock_coordinator,
        config={"azure_endpoint": "https://test.openai.azure.com", "api_key": "test-key"},
    )
    assert (
        "session.cost",
        "provider-azure-openai",
    ) in mock_coordinator.registered_contributors


# ---------------------------------------------------------------------------
# test_contributor_returns_none_before_any_calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_returns_none_before_any_calls(mock_coordinator):
    """Contributor callback returns None when no llm:response events have fired."""
    await mount(
        mock_coordinator,
        config={"azure_endpoint": "https://test.openai.azure.com", "api_key": "test-key"},
    )
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-azure-openai")
    ]
    assert callback() is None


# ---------------------------------------------------------------------------
# test_contributor_accumulates_after_llm_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_accumulates_after_llm_response(mock_coordinator):
    """_accumulate sums cost_usd over multiple events; callback returns Decimal total."""
    await mount(
        mock_coordinator,
        config={"azure_endpoint": "https://test.openai.azure.com", "api_key": "test-key"},
    )

    accumulate = mock_coordinator.registered_hooks["llm:response"]
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-azure-openai")
    ]

    await accumulate("llm:response", {"usage": {"cost_usd": "0.05"}})
    await accumulate("llm:response", {"usage": {"cost_usd": "0.03"}})

    result = callback()
    assert result is not None, "Callback should return a dict after cost events"
    assert "cost_usd" in result
    assert result["cost_usd"] == Decimal("0.08"), (
        f"Expected Decimal('0.08'), got {result['cost_usd']!r}"
    )
    assert isinstance(result["cost_usd"], Decimal), (
        f"cost_usd must be Decimal, got {type(result['cost_usd'])}"
    )


# ---------------------------------------------------------------------------
# test_contributor_ignores_none_cost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contributor_ignores_none_cost(mock_coordinator):
    """_accumulate ignores events where cost_usd is None; has_data stays False."""
    await mount(
        mock_coordinator,
        config={"azure_endpoint": "https://test.openai.azure.com", "api_key": "test-key"},
    )

    accumulate = mock_coordinator.registered_hooks["llm:response"]
    callback = mock_coordinator.registered_contributors[
        ("session.cost", "provider-azure-openai")
    ]

    await accumulate("llm:response", {"usage": {"cost_usd": None}})

    assert callback() is None, (
        "Callback should still return None after a None-cost event"
    )


# ---------------------------------------------------------------------------
# test_ptu_deployment_cost_is_none
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ptu_deployment_cost_is_none():
    """PTU deployment: _convert_to_chat_response must set cost_usd=None."""
    from amplifier_module_provider_azure_openai import _create_azure_provider

    try:
        from amplifier_module_provider_openai import OpenAIProvider
    except ImportError:
        pytest.skip("amplifier-module-provider-openai not installed")

    provider = _create_azure_provider(
        OpenAIProvider,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config={"deployment_type": "PTU", "use_streaming": False},
    )

    # Build a fake response with usage (non-None)
    fake_response = SimpleNamespace(
        output=[],
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=200,
            total_tokens=1200,
            prompt_tokens_details=None,
            input_tokens_details=None,
            output_tokens_details=None,
        ),
        model="gpt-5.4",
        id="resp-ptu-test",
        status="completed",
        finish_reason="stop",
        output_text=None,
    )

    chat_response = provider._convert_to_chat_response(fake_response)
    assert chat_response.usage is not None
    assert chat_response.usage.cost_usd is None, (
        f"PTU deployment must have cost_usd=None, got {chat_response.usage.cost_usd!r}"
    )


# ---------------------------------------------------------------------------
# test_payg_deployment_computes_cost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_payg_deployment_computes_cost():
    """PAYG deployment: _convert_to_chat_response must compute non-None cost for known model."""
    from amplifier_module_provider_azure_openai import _create_azure_provider

    try:
        from amplifier_module_provider_openai import OpenAIProvider
    except ImportError:
        pytest.skip("amplifier-module-provider-openai not installed")

    provider = _create_azure_provider(
        OpenAIProvider,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config={"use_streaming": False},  # No deployment_type = PAYG
    )

    fake_response = SimpleNamespace(
        output=[],
        usage=SimpleNamespace(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            total_tokens=1_000_000,
            prompt_tokens_details=None,
            input_tokens_details=None,
            output_tokens_details=None,
        ),
        model="gpt-5.4",
        id="resp-payg-test",
        status="completed",
        finish_reason="stop",
        output_text=None,
    )

    chat_response = provider._convert_to_chat_response(fake_response)
    assert chat_response.usage is not None
    assert chat_response.usage.cost_usd is not None, (
        "PAYG deployment must compute cost_usd for known model"
    )
    assert Decimal(str(chat_response.usage.cost_usd)) == Decimal("2.50"), (
        f"Expected $2.50 for 1M prompt_tokens on gpt-5.4, got {chat_response.usage.cost_usd!r}"
    )
