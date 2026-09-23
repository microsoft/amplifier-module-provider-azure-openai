"""Model work waits for completion; only explicit deadlines interrupt it."""

import json

import openai
import pytest
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_azure_openai import _create_azure_provider
from amplifier_module_provider_openai import OpenAIProvider
from openai import _base_client

# Azure tracks the OpenAI SDK transport (httpx / httpx2).
httpx = getattr(_base_client, "httpx2", None) or _base_client.httpx


def request(**kwargs):
    return ChatRequest(messages=[Message(role="user", content="Fixture")], **kwargs)


def completed():
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 1,
        "model": "gpt-6-astra",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["create", "raw", "stream"])
async def test_actual_sdk_model_transports_override_hidden_read_deadline(mode):
    model_calls = []

    async def handle(req):
        if req.url.path.endswith("input_tokens"):
            return httpx.Response(200, json={"input_tokens": 100})
        model_calls.append(req)
        if mode == "stream":
            body = "".join(
                "data: "
                + json.dumps(
                    {"type": kind, "sequence_number": seq, "response": completed()}
                )
                + "\n\n"
                for seq, kind in enumerate(["response.created", "response.completed"])
            )
            return httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=completed())

    # Even an injected SDK client with a short timeout must not silently impose
    # that limit on model work when the caller did not request a deadline.
    sdk = openai.AsyncOpenAI(
        api_key="fixture",
        timeout=0.001,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    provider = _create_azure_provider(
        OpenAIProvider,
        base_url="https://test.openai.azure.com/openai/v1/",
        api_key="fixture",
        config={
            "default_model": "gpt-6-astra",
            "use_streaming": mode == "stream",
            "enable_long_context": True,
            "max_retries": 0,
        },
    )
    provider._azure_client = sdk
    try:
        if mode == "raw":
            await provider._create_response(
                {"model": "gpt-6-astra", "input": [], "tools": [{"type": "computer"}]}
            )
        else:
            await provider.complete(request())
        assert len(model_calls) == 1
        assert model_calls[0].extensions["timeout"] == {
            "connect": 5.0,
            "pool": 5.0,
            "read": None,
            "write": None,
        }
    finally:
        await provider.close()


@pytest.mark.parametrize("timeout", [None, 30])
def test_azure_client_keeps_transport_bounds_and_parent_policy(timeout):
    provider = _create_azure_provider(
        OpenAIProvider,
        base_url="https://test.openai.azure.com/openai/v1/",
        api_key="fixture",
        config={"timeout": timeout},
    )
    assert provider.timeout == timeout
    assert provider.client.timeout.read == timeout
    assert provider.client.timeout.connect == provider.client.timeout.pool == 5
    assert provider.get_info().defaults["timeout"] is None


def test_azure_inherits_explicit_request_deadline_precedence():
    provider = _create_azure_provider(
        OpenAIProvider,
        base_url="https://test.openai.azure.com/openai/v1/",
        api_key="fixture",
        config={"timeout": 30},
    )
    assert provider._request_timeout(request(timeout=3)) == 3
    assert provider._request_timeout(request(timeout=None)) is None


@pytest.mark.asyncio
async def test_azure_does_not_invent_native_compaction_support():
    provider = _create_azure_provider(
        OpenAIProvider,
        base_url="https://test.openai.azure.com/openai/v1/",
        api_key="fixture",
    )
    assert not provider.supports_native_compaction()
    with pytest.raises(NotImplementedError):
        await provider.compact_context(request())
