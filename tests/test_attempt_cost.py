"""Current OpenAI parents price each Azure attempt and accumulate exactly once."""

import asyncio
from decimal import Decimal
from types import SimpleNamespace as S
from unittest.mock import AsyncMock, patch

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_module_provider_azure_openai import _create_azure_provider
from amplifier_module_provider_openai import OpenAIProvider


def response(*, model="gpt-5.4", status="completed", usage=True):
    return S(
        model=model,
        # Public OpenAI service-tier pricing must not replace Azure PAYG rates.
        service_tier="flex",
        status=status,
        incomplete_details=S(reason="max_output_tokens") if status == "incomplete" else None,
        id=f"resp_{status}",
        output=[],
        usage=S(
            input_tokens=1000,
            output_tokens=100,
            input_tokens_details=S(cached_tokens=200),
        ) if usage else None,
        model_dump=dict,
    )


def provider(costs, *, deployment="Standard"):
    p = _create_azure_provider(
        OpenAIProvider,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="fixture",
        config={
            "default_model": "gpt-5.4",
            "deployment_type": deployment,
            "use_streaming": False,
            "max_retries": 0,
        },
        add_cost=costs.append,
        coordinator=S(hooks=S(emit=AsyncMock())),
    )
    p._guard_assembled_params_with_provider_count = AsyncMock(return_value=1)
    return p


def test_direct_conversion_is_azure_priced_without_accumulation():
    costs = []
    p = provider(costs)
    with patch(
        "amplifier_module_provider_openai.compute_cost",
        side_effect=AssertionError("Public OpenAI pricing must not run"),
    ):
        out = p._convert_to_chat_response(response())
    # 800 fresh *2.5/M +200 cached *.25/M +100 output *15/M.
    assert out.usage.cost_usd == Decimal(".00355")
    assert costs == []


@pytest.mark.parametrize("continued", [False, True])
@pytest.mark.parametrize("deployment", ["Standard", "PTU"])
def test_complete_accounts_once_and_publishes_matching_cost(continued, deployment):
    async def check():
        costs = []
        p = provider(costs, deployment=deployment)
        p.client.responses.create = AsyncMock(
            side_effect=(
                [response(status="incomplete")] if continued else []
            ) + [response()]
        )
        try:
            out = await p.complete(
                ChatRequest(messages=[Message(role="user", content="test")])
            )
            expected = (
                None if deployment == "PTU"
                else Decimal(".00355") * (2 if continued else 1)
            )
            assert out.usage.cost_usd == expected
            assert costs == ([] if expected is None else [expected])
            events = [
                call.args[1]
                for call in p.coordinator.hooks.emit.await_args_list
                if call.args[0] == "llm:response"
            ]
            assert events[-1]["usage"]["cost_usd"] == (
                None if expected is None else str(expected)
            )
            assert p.client.responses.create.await_count == (2 if continued else 1)
        finally:
            await p.close()

    asyncio.run(check())


@pytest.mark.parametrize("missing", ["usage", "model"])
def test_unknown_constituent_is_not_priced_as_zero(missing):
    async def check():
        costs = []
        p = provider(costs)
        p.client.responses.create = AsyncMock(side_effect=[
            response(
                status="incomplete",
                usage=missing != "usage",
                model="unknown-deployment" if missing == "model" else "gpt-5.4",
            ),
            response(),
        ])
        try:
            out = await p.complete(
                ChatRequest(messages=[Message(role="user", content="test")])
            )
            assert out.usage.cost_usd is None
            assert costs == []
        finally:
            await p.close()

    asyncio.run(check())


def test_vendor_dict_and_object_usage_have_same_cost():
    p = provider([])
    raw = {
        "model": "gpt-5.4",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 0},
            "input_tokens_details": {"cached_tokens": 200},
        },
    }
    # Explicit Azure cached zero wins over the compatibility fallback.
    assert p._compute_attempt_cost(raw) == Decimal(".004")
    raw["usage"]["prompt_tokens_details"]["cached_tokens"] = None
    assert p._compute_attempt_cost(raw) == Decimal(".00355")