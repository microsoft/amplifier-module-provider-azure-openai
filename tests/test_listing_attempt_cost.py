"""Configured setup and hidden deployment inference preserve attempt-cost ownership."""

from decimal import Decimal
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from amplifier_core import ChatRequest, Message
import amplifier_module_provider_azure_openai as azure
from amplifier_module_provider_openai import OpenAIProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("continued", [False, True])
@pytest.mark.parametrize("mode", ["known", "ptu", "unknown-model", "missing-usage"])
async def test_listing_and_real_sdk_attempt_cost_share_no_policy(
    continued, mode, monkeypatch
):
    """Listing filters cannot alter dispatch, inherited billing or callback count."""
    config = {
        "deployments": ["Customer-Blue", "Research/Alias"],
        "default_model": "Customer-Blue",
        "model_filters": {"include": []},
        "deployment_type": "PTU" if mode == "ptu" else "Standard",
        "use_streaming": False,
        "max_retries": 0,
    }
    setup = azure.AzureOpenAIProvider(config=config)
    assert setup.get_model_list_info()["source"] == "configured"
    assert await setup.list_models() == []
    assert [m.id for m in await setup.list_models(filters={})] == [
        "Customer-Blue",
        "Research/Alias",
    ]
    costs, calls = [], []
    hooks = SimpleNamespace(emit=AsyncMock())
    provider = azure._create_azure_provider(
        OpenAIProvider,
        base_url="https://azure.invalid/openai/v1/",
        api_key="fixture-only",
        config=config,
        coordinator=SimpleNamespace(hooks=hooks),
        add_cost=costs.append,
    )
    assert provider._uses_attempt_cost is True
    assert type(provider).list_models is type(setup).list_models
    assert await provider.list_models() == []
    assert provider._azure_client is None

    generation_count = 0

    def handler(request):
        nonlocal generation_count
        assert request.headers["authorization"] == "Bearer fixture-only"
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        assert body["model"] == "Hidden/Exact"
        if request.url.path.endswith("/responses/input_tokens"):
            return httpx.Response(200, json={"input_tokens": 5})
        assert request.url.path == "/openai/v1/responses"
        assert not set(body) & {"deployments", "model_filters", "filters"}
        generation_count += 1
        incomplete = continued and generation_count == 1
        return httpx.Response(
            200,
            json={
                "id": f"response_{generation_count}",
                "object": "response",
                "created_at": 0,
                "status": "incomplete" if incomplete else "completed",
                "incomplete_details": {"reason": "max_output_tokens"}
                if incomplete
                else None,
                "model": "unpriced-deployment"
                if mode == "unknown-model"
                else "gpt-5.4",
                "service_tier": "flex",
                "output": [],
                "usage": None
                if mode == "missing-usage"
                else {
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "total_tokens": 1100,
                    "input_tokens_details": {"cached_tokens": 200},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    def forbidden_public_pricing(*args, **kwargs):
        raise AssertionError("Azure must keep endpoint attempt-price ownership")

    monkeypatch.setattr(
        "amplifier_module_provider_openai.compute_cost", forbidden_public_pricing
    )
    provider._azure_client = openai.AsyncOpenAI(
        base_url="https://azure.invalid/openai/v1/",
        api_key="fixture-only",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    client = provider._azure_client
    try:
        result = await provider.complete(
            ChatRequest(
                model="Hidden/Exact",
                max_output_tokens=1000,
                messages=[Message(role="user", content="test")],
            )
        )
        expected = (
            Decimal(".00355") * (2 if continued else 1) if mode == "known" else None
        )
        assert getattr(result.usage, "cost_usd", None) == expected
        assert costs == ([] if expected is None else [expected])
        assert generation_count == (2 if continued else 1)
        assert calls
        events = [
            call.args[1]
            for call in hooks.emit.await_args_list
            if call.args[0] == "llm:response"
        ]
        assert events[-1].get("usage", {}).get("cost_usd") == (
            str(expected) if expected is not None else None
        )
        assert await provider.list_models() == []
    finally:
        await provider.close()
    assert client.is_closed()
    assert provider._azure_client is None
