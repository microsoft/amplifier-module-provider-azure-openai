"""Optional F1 candidate integration: Azure's honest original-position fallback.

Run with an explicitly selected Foundation media-v1 source on PYTHONPATH.
Foundation is a TEST integration dependency only, never an Azure runtime import.
These tests qualify host projection + actual SDK text wire, not inherited media
retry/diagnostic conformance, which requires the accepted OpenAI-provider lane.
All sockets are fenced and ambient inference credentials are removed.
"""

import copy
import io
import json
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from amplifier_core.message_models import ChatRequest, Message, TextBlock, ToolCallBlock
from amplifier_module_provider_azure_openai import _create_azure_provider
from amplifier_module_provider_openai import OpenAIProvider

NAMESPACE = "experimental.media.v1"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import os

    for name in tuple(os.environ):
        if name.startswith(("AZURE_", "OPENAI_", "ANTHROPIC_", "GOOGLE_", "GEMINI_")):
            monkeypatch.delenv(name)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline media fixture attempted a real socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def media(tmp_path):
    media = pytest.importorskip("amplifier_foundation.session.media", reason="requires selected Foundation media-v1 test candidate")
    image = pytest.importorskip("PIL.Image", reason="requires Foundation media test extra")
    store = media.SessionMediaStore(
        tmp_path / "media", "azure-fixture",
        policy=media.MediaPolicy(1024 * 1024, 10000, 10),
        checkpoint_verifier=lambda receipt: receipt["refs"],
    )
    try:
        yield media, store, image
    finally:
        store.close()


async def committed_ref(media, *, fmt="PNG"):
    module, store, image = media
    stream = io.BytesIO()
    image.new("RGB", (3, 2), "red").save(stream, format=fmt)
    origin = {"kind": "tool", "session_id": store.session_id, "id": "call-image"}
    staged = await store.ingest(stream.getvalue(), source_format="image-bytes", origin=origin, policy=None)
    ref = staged["value"]
    await store.publish([ref], origin=origin, disposition="retain")
    store.acknowledge_checkpoint({"refs": [ref]})
    return ref


def request_for(ref, role):
    content = [TextBlock(text="before"), TextBlock(text=f"[Session image {ref['sha256']}]"), TextBlock(text="after")]
    observed = Message(
        role=role, tool_call_id="call-image" if role == "tool" else None,
        content=content, metadata={NAMESPACE: {"refs": [ref]}, "owner": {"provenance": "opaque-kept"}},
    )
    messages = [Message(role="user", content="start")]
    if role == "tool":
        messages.append(Message(role="assistant", content=[ToolCallBlock(id="call-image", name="image_tool", input={})]))
    messages.extend([observed, Message(role="user", content="end")])
    return ChatRequest(model="Customer-Blue", max_output_tokens=1000, messages=messages), len(messages) - 2


def identity(provider):
    return {
        "provider": "azure-openai", "account": "synthetic-account", "generation": 1,
        "media_support": provider.get_media_request_support(model="Customer-Blue", request_options={}),
    }


def make_provider(cap=None, **config):
    hooks = SimpleNamespace(emit=AsyncMock())
    coordinator = SimpleNamespace(hooks=hooks, get_capability=lambda key: cap if key == NAMESPACE else None)
    return _create_azure_provider(
        OpenAIProvider, base_url="https://azure.invalid/openai/v1/", api_key="fixture-only",
        coordinator=coordinator, config={"use_streaming": False, "max_retries": 0, **config},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
async def test_unavailable_before_budget_preserves_position_and_exact_sdk_input(media, role, fmt, monkeypatch):
    module, store, _ = media
    ref = await committed_ref(media, fmt=fmt)
    canonical, index = request_for(ref, role)
    before = canonical.model_dump()
    cap = module.SessionMediaCapability(store, enabled=True)
    provider = make_provider(cap)
    # Unsupported Azure positions must not even open pixels to inspect them,
    # during either preparation or later validation.
    monkeypatch.setattr(store, "resolve_ref", lambda *a, **k: pytest.fail("unqualified Azure route resolved pixels"))
    prepared = await cap.prepare_request(
        canonical, provider_identity=identity(provider), model="Customer-Blue", request_options={},
        purpose="count", execution_scope="root",
    )
    bodies = []
    try:
        private = prepared.request
        assert canonical.model_dump() == before
        assert private is not canonical
        assert [m.role for m in private.messages] == [m.role for m in canonical.messages]
        assert private.messages[index].tool_call_id == canonical.messages[index].tool_call_id
        assert private.messages[index].metadata == canonical.messages[index].metadata
        blocks = private.messages[index].content
        assert [b.type for b in blocks] == ["text", "text", "text"]
        assert blocks[0].text == "before" and blocks[2].text == "after"
        assert "Image unavailable" in blocks[1].text and "no pixels supplied" in blocks[1].text
        assert cap.validate(private, purpose="count", request_options={}) == {"valid": True}
        budget_input = provider._budget_params(private)["input"]
        # Azure cannot advertise the public OpenAI native counter on a custom endpoint.
        assert provider._provider_count_available() is False
        budget = provider.request_budget(private, context_estimate=100, request_options={})
        assert budget is None or budget.get("measurement", {}).get("kind") != "provider_count"

        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            assert body["model"] == "Customer-Blue"
            assert body["input"] == budget_input
            serialized = json.dumps(body)
            assert "base64" not in serialized and "input_image" not in serialized
            assert NAMESPACE not in serialized  # canonical ref metadata is not wire
            assert serialized.index("before") < serialized.index("Image unavailable") < serialized.index("after")
            return httpx.Response(200, json={
                "id": "response-fixture", "object": "response", "created_at": 0,
                "status": "completed", "model": "Customer-Blue", "output": [],
                "usage": {"input_tokens": 20, "output_tokens": 0, "total_tokens": 20},
            })

        provider._azure_client = openai.AsyncOpenAI(
            base_url="https://azure.invalid/openai/v1/", api_key="fixture-only", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        assert cap.validate(private, purpose="transport", request_options={}) == {"valid": True}
        await provider.complete(private)
        assert len(bodies) == 1
        assert canonical.model_dump() == before
        # No image transport existed in this fallback. This is NOT general
        # inherited raw-event/image privacy qualification.
        assert "input_image" not in str(provider.coordinator.hooks.emit.await_args_list)
    finally:
        prepared.release()
        await provider.close()
    assert prepared.released
    assert prepared.request.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["forged", "uncommitted"])
async def test_unsupported_is_not_a_bypass_for_invalid_or_uncommitted_refs(media, bad):
    module, store, _ = media
    ref = await committed_ref(media)
    if bad == "forged":
        ref = copy.deepcopy(ref)
        ref["provenance"]["id"] = "another-tool"
    else:
        origin = {"kind": "tool", "session_id": store.session_id, "id": "not-committed"}
        with store.resolve_ref(ref) as stream:
            ref = (await store.ingest(stream.read(), source_format="image-bytes", origin=origin, policy=None))["value"]
        await store.publish([ref], origin=origin, disposition="retain")
    canonical, _ = request_for(ref, "tool")
    cap = module.SessionMediaCapability(store, enabled=True)
    provider = make_provider(cap)
    with pytest.raises(module.MediaRefusal):
        await cap.prepare_request(canonical, provider_identity=identity(provider), model="Customer-Blue",
                                  request_options={}, purpose="count", execution_scope="root")
    assert provider._azure_client is None
