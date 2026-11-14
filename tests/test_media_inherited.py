"""Optional exact OpenAI media-base + Foundation integration, offline only.

These fixtures qualify Azure's inherited guards and original-position fallback,
NOT real Azure image support. The selected deployment has no positive facts.
Legacy diagnostic fixtures intentionally exercise already-supported inline input
without a media capability; their MockTransport is not service/vision evidence.
"""

import base64
import copy
import hashlib
import io
import json
import warnings

import httpx
import openai
import pytest
from amplifier_core.llm_errors import InvalidRequestError, LLMError
from amplifier_core.message_models import ChatRequest, ImageBlock, Message, TextBlock

from tests.test_media_projection_integration import (
    NAMESPACE, committed_ref, identity, make_provider, media, offline, request_for,
)


@pytest.fixture(autouse=True)
def candidate_base():
    pytest.importorskip("amplifier_module_provider_openai._media", reason="requires selected OpenAI media-base test candidate")


def response(*, output=None):
    return {
        "id": "response-fixture", "object": "response", "created_at": 0,
        "status": "completed", "model": "Customer-Blue", "output": output or [],
        "usage": {"input_tokens": 20, "output_tokens": 0, "total_tokens": 20},
    }


def sdk(provider, handler):
    provider._azure_client = openai.AsyncOpenAI(
        base_url="https://azure.invalid/openai/v1/", api_key="fixture-only", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False),
    )
    return provider._azure_client


def audit_cap(monkeypatch, cap):
    calls = []
    original = cap.validate

    def validate(value, *, purpose, request_options=None):
        calls.append((value, purpose, copy.deepcopy(request_options)))
        return original(value, purpose=purpose, request_options=request_options)

    monkeypatch.setattr(cap, "validate", validate)
    return calls


def test_actual_lazy_azure_sdk_is_nonredirecting():
    """Inspect the production factory client; no request/credentials are used."""
    import asyncio
    p = make_provider()
    client = p.client
    try:
        assert client.max_retries == 0
        assert client._client.follow_redirects is False
    finally:
        asyncio.run(p.close())


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("stream", [False, True])
async def test_real_cap_wrapper_fallback_budget_send_stream(media, role, stream, monkeypatch):
    from amplifier_foundation.session.media_provider import MediaPreparedProvider

    module, store, _ = media
    ref = await committed_ref(media)
    req, index = request_for(ref, role)
    before = req.model_dump()
    cap = module.SessionMediaCapability(store, enabled=True)
    validations = audit_cap(monkeypatch, cap)
    p = make_provider(cap, use_streaming=stream, raw=True)
    calls, events = [], []

    async def emit(name, data):
        events.append((name, copy.deepcopy(data)))
        assert "input_image" not in json.dumps(data)
        # An observer can change its diagnostic COPY, never the fitted wire.
        if name == "llm:request" and "raw" in data:
            data["raw"]["model"] = "must-not-reach-wire"
            data["raw"]["input"] = []

    p.coordinator.hooks.emit = emit

    def receive(request):
        body = json.loads(request.content)
        calls.append(body)
        assert body["model"] == "Customer-Blue"
        assert "input_image" not in json.dumps(body)
        assert "Image unavailable" in json.dumps(body)
        if stream:
            reply = response()
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text="".join(
                "data: " + json.dumps(event) + "\n\n" for event in [
                    {"type": "response.created", "response": reply},
                    {"type": "response.completed", "response": reply},
                ]
            ))
        return httpx.Response(200, json=response())

    client = sdk(p, receive)
    wrapped = MediaPreparedProvider(p, cap, provider_identity=identity(p), model="Customer-Blue", execution_scope="root")
    monkeypatch.setattr(store, "resolve_ref", lambda *a, **k: pytest.fail("unqualified Azure route resolved pixels"))
    try:
        budget = await wrapped.request_budget(req, context_estimate=100, request_options={})
        assert budget is None or budget.get("measurement", {}).get("kind") != "provider_count"
        assert calls == []  # Real Azure route does not advertise native count.
        private = wrapped._candidate.prepared.request
        budget_input = p._budget_params(private)["input"]
        await wrapped.complete(req)
        assert len(calls) == 1 and calls[0]["input"] == budget_input
        assert req.model_dump() == before
        assert len({id(v) for v, _, _ in validations}) == 1
        assert private is not req
        assert {purpose for _, purpose, _ in validations} == {"count", "transport"}
        assert all(opts == {} for _, _, opts in validations)
        assert any(name == "llm:response" for name, _ in events)
        assert private.messages == []  # wrapper releases its projection after result
    finally:
        wrapped.close()
        await p.close()
    assert client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["budget", "complete", "compact"])
async def test_real_cap_rejects_missing_private_association(media, operation):
    module, store, _ = media
    cap = module.SessionMediaCapability(store, enabled=True)
    # Even ordinary no-media requests need an association when realcap is present.
    req = ChatRequest(model="Customer-Blue", messages=[Message(role="user", content="text")])
    p = make_provider(cap)
    calls = []
    sdk(p, lambda r: calls.append(r) or httpx.Response(200, json=response()))
    try:
        with pytest.raises(InvalidRequestError):
            if operation == "budget":
                p.request_budget(req, context_estimate=1)
            elif operation == "compact":
                await p.compact_context(req)
            else:
                await p.complete(req)
        assert calls == []
    finally:
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["revoke", "private", "options"])
async def test_fitted_mutation_fails_before_transport(media, mutation, monkeypatch):
    module, store, _ = media
    ref = await committed_ref(media)
    canonical, _ = request_for(ref, "tool")
    cap = module.SessionMediaCapability(store, enabled=True)
    p = make_provider(cap)
    calls = []
    sdk(p, lambda r: calls.append(r) or httpx.Response(200, json=response()))
    prepared = await cap.prepare_request(canonical, provider_identity=identity(p), model="Customer-Blue", request_options={}, purpose="count", execution_scope="root")
    try:
        p.request_budget(prepared.request, context_estimate=100)
        if mutation == "revoke":
            store.epoch += 1
        elif mutation == "private":
            prepared.request.messages[-1].content = "mutated"
        kwargs = {"reasoning_effort": "low"} if mutation == "options" else {}
        with pytest.raises(InvalidRequestError):
            await p.complete(prepared.request, **kwargs)
        assert calls == []
    finally:
        prepared.release()
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("revoke", [False, True])
async def test_known_rejection_retry_revalidates_exact_fallback(media, revoke, monkeypatch):
    module, store, _ = media
    ref = await committed_ref(media)
    canonical, _ = request_for(ref, "tool")
    cap = module.SessionMediaCapability(store, enabled=True)
    validations = audit_cap(monkeypatch, cap)
    p = make_provider(cap, max_retries=1, min_retry_delay=0, max_retry_delay=0)
    calls, at_attempt = [], []

    def receive(request):
        calls.append(json.loads(request.content))
        at_attempt.append(len(validations))
        if len(calls) == 1:
            if revoke:
                store.epoch += 1
            return httpx.Response(503, json={"error": {"message": "known fixture rejection"}})
        return httpx.Response(200, json=response())

    sdk(p, receive)
    prepared = await cap.prepare_request(canonical, provider_identity=identity(p), model="Customer-Blue", request_options={}, purpose="count", execution_scope="root")
    try:
        if revoke:
            with pytest.raises(InvalidRequestError):
                await p.complete(prepared.request)
            assert len(calls) == 1
        else:
            await p.complete(prepared.request)
            assert len(calls) == 2 and calls[0] == calls[1]
            assert at_attempt[1] > at_attempt[0]  # validation happens inside owned retry
        assert all(value is prepared.request for value, _, _ in validations)
        assert p.client.max_retries == 0
    finally:
        prepared.release()
        await p.close()


@pytest.mark.asyncio
async def test_unknown_submission_not_replayed(media):
    module, store, _ = media
    ref = await committed_ref(media)
    canonical, _ = request_for(ref, "user")
    cap = module.SessionMediaCapability(store, enabled=True)
    p = make_provider(cap, max_retries=2, min_retry_delay=0, max_retry_delay=0)
    calls = []

    def fail(request):
        calls.append(request)
        raise httpx.ReadTimeout("unknown fixture submission")

    sdk(p, fail)
    prepared = await cap.prepare_request(canonical, provider_identity=identity(p), model="Customer-Blue", request_options={}, purpose="count", execution_scope="root")
    try:
        with pytest.raises(LLMError) as caught:
            await p.complete(prepared.request)
        assert not caught.value.retryable
        assert len(calls) == 1
    finally:
        prepared.release()
        await p.close()


@pytest.mark.asyncio
async def test_azure_compaction_is_explicitly_unavailable_after_witness_validation(media, monkeypatch):
    module, store, _ = media
    ref = await committed_ref(media)
    canonical, _ = request_for(ref, "tool")
    cap = module.SessionMediaCapability(store, enabled=True)
    validations = audit_cap(monkeypatch, cap)
    p = make_provider(cap)
    calls = []
    sdk(p, lambda r: calls.append(r) or httpx.Response(200, json=response()))
    prepared = await cap.prepare_request(canonical, provider_identity=identity(p), model="Customer-Blue", request_options={}, purpose="count", execution_scope="root")
    try:
        assert p.supports_native_compaction() is False
        with pytest.raises(NotImplementedError, match="does not advertise"):
            await p.compact_context(prepared.request)
        assert calls == []
        assert validations and all(value is prepared.request for value, _, _ in validations)
    finally:
        prepared.release()
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe", ["redirect", "request_hook", "response_hook"])
async def test_protected_sdk_transport_policy_rejects_before_send(media, unsafe):
    module, store, _ = media
    cap = module.SessionMediaCapability(store, enabled=True)
    req = ChatRequest(model="Customer-Blue", messages=[Message(role="user", content="text")])
    p = make_provider(cap)
    calls = []
    client = sdk(p, lambda r: calls.append(r) or httpx.Response(200, json=response()))
    if unsafe == "redirect":
        client._client.follow_redirects = True
    else:
        async def observer(*args):
            pytest.fail("unqualified transport observer ran")
        key = "request" if unsafe == "request_hook" else "response"
        client._client.event_hooks[key] = [observer]
    prepared = await cap.prepare_request(req, provider_identity=identity(p), model="Customer-Blue", request_options={}, purpose="count", execution_scope="root")
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(prepared.request)
        assert calls == []
    finally:
        prepared.release()
        await p.close()


@pytest.mark.asyncio
async def test_media_enabled_dynamic_azure_token_callback_refuses_before_use(media):
    """Auth callbacks may hide post-witness requests; no token is acquired."""
    module, store, _ = media
    cap = module.SessionMediaCapability(store, enabled=True)
    req = ChatRequest(model="Customer-Blue", messages=[Message(role="user", content="text")])
    p = make_provider(cap)
    calls, token_calls = [], []
    client = sdk(p, lambda r: calls.append(r) or httpx.Response(200, json=response()))

    async def token():
        token_calls.append(True)
        return "must-not-be-acquired"

    # Match the supported Azure factory's AsyncOpenAI(api_key=callable) path.
    client.api_key = ""
    client._api_key_provider = token
    prepared = await cap.prepare_request(req, provider_identity=identity(p), model="Customer-Blue", request_options={}, purpose="count", execution_scope="root")
    try:
        with pytest.raises(InvalidRequestError):
            await p.complete(prepared.request)
        assert calls == [] and token_calls == []
    finally:
        prepared.release()
        await p.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["key", "client", "wire_model"])
@pytest.mark.parametrize("when", ["after_budget", "request_hook"])
async def test_azure_actual_connection_drift_before_send_refuses(media, change, when):
    module, store, _ = media
    ref = await committed_ref(media)
    canonical, _ = request_for(ref, "user")
    cap = module.SessionMediaCapability(store, enabled=True)
    p = make_provider(cap)
    calls, clients = [], []
    clients.append(sdk(p, lambda r: calls.append(r) or httpx.Response(200, json=response())))
    prepared = await cap.prepare_request(canonical, provider_identity=identity(p), model="Customer-Blue", request_options={}, purpose="count", execution_scope="root")

    def drift():
        if change == "key":
            p.client.api_key = "changed-fixture-account"
        elif change == "wire_model":
            p.extra_request_params["model"] = "Other-Deployment"
        else:
            clients.append(sdk(p, lambda r: calls.append(r) or httpx.Response(200, json=response())))

    if when == "request_hook":
        async def mutate(name, data):
            if name == "llm:request":
                drift()
        p.coordinator.hooks.emit = mutate
    try:
        p.request_budget(prepared.request, context_estimate=100)
        if when == "after_budget":
            drift()
        with pytest.raises(InvalidRequestError):
            await p.complete(prepared.request)
        assert calls == []
    finally:
        prepared.release()
        await p.close()
        for client in clients:
            await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["user", "tool"])
async def test_legacy_image_wire_diagnostic_copy_is_safe_before_hooks(media, role):
    # NO real deployment capability is asserted here. Exercise legacy serializer
    # solely to put synthetic bytes through diagnostic boundaries without a cap.
    _, _, image = media
    out = io.BytesIO()
    image.new("RGB", (3, 2), "red").save(out, format="PNG")
    pixels = out.getvalue()
    encoded = base64.b64encode(pixels).decode()
    source = {"type": "base64", "media_type": "image/png", "data": encoded}
    p = make_provider(None, raw=True)
    if role == "tool":
        from amplifier_core.message_models import ToolCallBlock
        messages = [Message(role="user", content="start"), Message(role="assistant", content=[ToolCallBlock(id="c1", name="image_tool", input={})]), Message(role="tool", tool_call_id="c1", content=[TextBlock(text="before"), ImageBlock(source=source), TextBlock(text="after")])]
    else:
        messages = [Message(role="user", content=[TextBlock(text="before"), ImageBlock(source=source), TextBlock(text="after")])]
    req = ChatRequest(model="Customer-Blue", max_output_tokens=1000, messages=messages)
    original = req.model_dump()
    events, wire = [], []

    async def emit(name, data):
        serialized = json.dumps(data)
        assert encoded not in serialized and "data:image" not in serialized
        events.append((name, copy.deepcopy(data)))
        if name == "llm:request":
            assert "not retained" in serialized
            data["raw"]["input"].clear()

    p.coordinator.hooks.emit = emit

    def receive(request):
        body = json.loads(request.content)
        wire.append(body)
        final = body["input"][-1]
        parts = final["content"] if role == "user" else final["output"]
        if role == "tool":
            assert final["call_id"] == "c1"
        assert [part["type"] for part in parts] == ["input_text", "input_image", "input_text"]
        decoded = base64.b64decode(parts[1]["image_url"].split(",", 1)[1], validate=True)
        assert hashlib.sha256(decoded).digest() == hashlib.sha256(pixels).digest()
        # Known response image fields must be removed before the observer too.
        return httpx.Response(200, json=response(output=[{
            "type": "function_call_output", "call_id": "echo-fixture",
            "output": [{"type": "input_image", "image_url": "data:image/png;base64," + encoded}],
        }]))

    sdk(p, receive)
    try:
        # SDK models may retain malformed/unexpected server payloads. Their
        # serializer warnings must not become a diagnostic pixel side channel.
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            await p.complete(req)
        assert not any("Pydantic serializer warnings" in str(w.message) for w in emitted), (
            "SDK serializer diagnostics expose raw image-source fragments before omission"
        )
        assert req.model_dump() == original
        assert len(wire) == 1
        assert any(name == "llm:response" and "not retained" in json.dumps(data) for name, data in events)
    finally:
        await p.close()
