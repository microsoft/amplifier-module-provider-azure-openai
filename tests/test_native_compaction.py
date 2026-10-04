"""Azure compaction reuses the Responses transport, not OpenAI endpoint claims."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core import ChatRequest, Message
from amplifier_module_provider_openai import OpenAIProvider

from amplifier_module_provider_azure_openai import _create_azure_provider


def provider(enabled=True, endpoint="https://example.openai.azure.com/openai/v1/"):
    return _create_azure_provider(
        OpenAIProvider,
        base_url=endpoint,
        api_key="test",
        config={"default_model": "gpt-5.6-terra", "native_compaction": enabled},
    )


@pytest.mark.parametrize(
    "enabled,endpoint,expected",
    [
        (True, "https://example.openai.azure.com/openai/v1/", True),
        (True, "https://example.services.ai.azure.com/openai/v1/", True),
        (False, "https://example.openai.azure.com/openai/v1/", False),
        ("false", "https://example.openai.azure.com/openai/v1/", False),
        (True, "https://proxy.example/openai/v1/", False),
        (True, "https://example.openai.azure.com/legacy/", False),
        (True, "https://example.openai.azure.com.evil.example/openai/v1/", False),
    ],
)
def test_compaction_endpoint_capability_is_separate_from_counting(
    enabled, endpoint, expected
):
    p = provider(enabled, endpoint)
    assert p.supports_native_compaction() is expected
    assert p._provider_count_available() is False
    assert ("native_compaction" in p.get_info().capabilities) is expected
    assert "request_budget:provider_count" not in p.get_info().capabilities


@pytest.mark.asyncio
async def test_entire_window_survives_count_continuation_and_recompaction():
    p = provider()
    output = [
        {"role": "user", "type": "message", "content": "retained original"},
        {"type": "compaction", "encrypted_content": "opaque-fixture", "id": "compact1"},
    ]
    p.client.responses.compact = AsyncMock(
        return_value=SimpleNamespace(output=deepcopy(output), usage=None)
    )
    req = ChatRequest(messages=[Message(role="user", content="first")])
    original = req.model_dump()
    result = await p.compact_context(req)
    assert req.model_dump() == original
    assert p.validate_compacted_context(result["message"])
    continued = ChatRequest(
        messages=[Message(**result["message"]), Message(role="user", content="next")]
    )
    wire = p._budget_params(continued)
    assert wire["input"][:2] == output
    assert "Native compacted" not in str(wire)
    # No fabricated token count for encrypted state. Context-managed falls back.
    assert p.request_budget(continued, context_estimate=0) is None
    await p.compact_context(continued)
    assert p.client.responses.compact.call_args.kwargs["input"][:2] == output


@pytest.mark.asyncio
async def test_cross_resource_checkpoint_is_rejected_before_dispatch():
    p = provider()
    p.client.responses.compact = AsyncMock(
        return_value=SimpleNamespace(
            output=[{"type": "compaction", "encrypted_content": "opaque"}], usage=None
        )
    )
    result = await p.compact_context(
        ChatRequest(messages=[Message(role="user", content="first")])
    )
    other = provider(endpoint="https://other.openai.azure.com/openai/v1/")
    assert other.validate_compacted_context(result["message"]) is False
    with pytest.raises(ValueError, match="another Azure resource"):
        other._budget_params(ChatRequest(messages=[Message(**result["message"])]))


@pytest.mark.asyncio
async def test_disabled_compaction_does_not_dispatch():
    p = provider(False)
    p.client.responses.compact = AsyncMock()
    with pytest.raises(NotImplementedError):
        await p.compact_context(
            ChatRequest(messages=[Message(role="user", content="first")])
        )
    p.client.responses.compact.assert_not_called()
