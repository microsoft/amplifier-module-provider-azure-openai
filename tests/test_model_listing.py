"""Configured Azure deployment listing, without management credentials or login."""

import json

import httpx
import openai
import pytest
from amplifier_core.llm_errors import ProviderUnavailableError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_openai import OpenAIProvider

from amplifier_module_provider_azure_openai import _create_azure_provider
from amplifier_module_provider_azure_openai._model_listing import ModelFilters


def make(config):
    return _create_azure_provider(
        OpenAIProvider,
        base_url="https://azure.invalid/openai/v1/",
        api_key="test-key",
        config=config,
    )


def ids(models):
    return [m.id for m in models]


@pytest.mark.asyncio
async def test_configured_aliases_all_and_manual_source_without_client():
    config = {
        "deployments": ["production-chat", "Team/Custom:Q4", "legacy"],
        "default_model": "not-in-list",
    }
    p = make(config)
    config["deployments"].clear()
    assert ids(await p.list_models()) == ["Team/Custom:Q4", "legacy", "production-chat"]
    assert p._azure_client is None
    assert p.get_model_list_info()["source"] == "configured"
    assert all(
        m.context_window == 0 and m.max_output_tokens == 0 and not m.capabilities
        for m in await p.list_models()
    )
    assert ids(await p.list_models(filters={"include": ["TEAM/*"]})) == [
        "Team/Custom:Q4"
    ]
    assert ids(await p.list_models(filters={"exclude": ["legacy"]})) == [
        "Team/Custom:Q4",
        "production-chat",
    ]
    assert await p.list_models(filters={"include": []}) == []
    assert make({"deployments": []}).get_model_list_info()["source"] == "configured"
    assert await make({"deployments": []}).list_models() == []


@pytest.mark.asyncio
async def test_unsupported_does_not_invent_inherited_default_or_request_auth():
    p = make({})
    assert p.get_model_list_info()["source"] == "unsupported"
    with pytest.raises(ProviderUnavailableError, match="exact deployment"):
        await p.list_models()
    assert p._azure_client is None
    for key in ("default_model", "deployment_name", "default_deployment"):
        assert ids(await make({key: "arbitrary-alias"}).list_models()) == [
            "arbitrary-alias"
        ]


@pytest.mark.asyncio
async def test_filter_snapshot_replace_metadata_and_hidden_inference():
    cfg = {"include": ["prod*"], "exclude": []}
    p = make(
        {"deployments": ["prod", "dev"], "model_filters": cfg, "use_streaming": False}
    )
    cfg["include"].clear()
    assert ids(await p.list_models()) == ["prod"]
    assert ids(await p.list_models(filters=None)) == ["prod"]
    assert ids(await p.list_models(filters={})) == ["dev", "prod"]
    assert ids(await p.list_models(filters={"include": ["dev"]})) == ["dev"]
    info = p.get_model_list_info()
    info["default_filters"]["include"].clear()
    info["filters_schema"]["properties"].clear()
    assert p.get_model_list_info()["default_filters"]["include"] == ["prod*"]
    assert "include" in p.get_model_list_info()["filters_schema"]["properties"]
    calls = []

    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        assert body["model"] == "hidden-exact-deployment"
        assert body["max_output_tokens"] == 100
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": body["model"],
                "output": [],
            },
        )

    p._azure_client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://azure.invalid/openai/v1/",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        await p.complete(
            ChatRequest(
                model="hidden-exact-deployment",
                max_output_tokens=100,
                messages=[Message(role="user", content="test")],
            )
        )
        assert len(calls) == 1 and calls[0].url.path == "/openai/v1/responses"
    finally:
        await p.close()


@pytest.mark.parametrize("value", [None, "prod", [""], [1], [{}]])
def test_invalid_deployments(value):
    with pytest.raises((TypeError, ValueError)):
        make({"deployments": value})


@pytest.mark.parametrize(
    "value",
    [None, [], {"include": "*"}, {"exclude": [1]}, {"hide_dated": 1}, {"unknown": []}],
)
def test_invalid_filters(value):
    with pytest.raises((TypeError, ValueError)):
        make({"model_filters": value})


def test_common_vectors():
    fixture = ["Custom:Q4", "Custom:Q8", "future", "future-2099-01-01"]
    assert ModelFilters.parse({}).select(fixture) == sorted(fixture)
    assert ModelFilters.parse({"include": ["CUSTOM:*"], "exclude": ["*:q8"]}).select(
        fixture
    ) == ["Custom:Q4"]
    assert ModelFilters.parse({"hide_dated": True}).select(fixture) == [
        "Custom:Q4",
        "Custom:Q8",
        "future",
    ]
    assert ModelFilters.parse({"include": []}).select(fixture) == []
