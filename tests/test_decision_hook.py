"""Offline real-hook lifecycle and shared Azure account authentication contracts."""
import json
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from amplifier_core import AmplifierSession
from amplifier_decisions import DecisionRequest, Question
from amplifier_module_provider_azure_openai import decision_hook, decisions

pytestmark = pytest.mark.asyncio
REQUEST = DecisionRequest(
    "Synthetic Save", (Question("visible", "predicate", "Is Save visible?"),),
    allow_external_state=True,
)


def coordinator():
    values = {}
    return SimpleNamespace(
        mount=AsyncMock(), get_capability=values.get,
        register_capability=values.__setitem__, register_contributor=lambda *args: None,
        hooks=SimpleNamespace(emit=AsyncMock(return_value=SimpleNamespace(action="continue"))),
        process_hook_result=AsyncMock(side_effect=lambda result, *args: result),
    )


@pytest.fixture
def captured(monkeypatch):
    import amplifier_module_provider_azure_openai as azure
    for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY",
                 "AZURE_OPENAI_KEY", "AZURE_USE_DEFAULT_CREDENTIAL", "AZURE_USE_MANAGED_IDENTITY"):
        monkeypatch.delenv(name, raising=False)
    seen, transports = [], []

    def no_chat():
        raise AssertionError("A decision-only hook must not load a chat provider")

    def response(request):
        seen.append(request)
        return httpx.Response(200, json={"model": "microsoft-decision-1", "answers": {
            "visible": {"type": "noul", "noul": .99}}})

    original = decisions.FoundryDecisions
    class Transport(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._client = httpx.AsyncClient(transport=httpx.MockTransport(response))
            transports.append(self)

    monkeypatch.setattr(azure, "_get_openai_provider_class", no_chat)
    monkeypatch.setattr(decisions, "FoundryDecisions", Transport)
    return seen, transports


@pytest.mark.parametrize("endpoint_var", ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_BASE_URL"])
@pytest.mark.parametrize("override", [None, "account", "decision"])
async def test_hook_reuses_account_aliases_and_explicit_overrides(captured, monkeypatch, endpoint_var, override):
    seen, transports = captured
    monkeypatch.setenv(endpoint_var, "https://environment.services.ai.azure.com/models")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "environment-key")
    config = {"chat_enabled": False, "decisions": {"enabled": True, "model": "deployment"}}
    resource, key = "https://environment.services.ai.azure.com", "environment-key"
    if override == "account":
        resource = config["azure_endpoint"] = "https://account.services.ai.azure.com"
        key = config["api_key"] = "account-key"
    if override == "decision":
        resource = config["decisions"]["endpoint"] = "https://gateway.test/prefix"
    before = deepcopy(config)
    host = coordinator()
    cleanup = await decision_hook.mount(host, config)
    assert seen == [] and not host.mount.called
    assert config == before
    backend = host.get_capability("decision.backends")["azure-decisions"]
    result = await backend.decide(REQUEST)
    assert result.answers["visible"].value == .99
    assert str(seen[0].url) == resource + "/providers/microsoft/v1/systemone"
    assert seen[0].headers["api-key"] == key
    await cleanup()
    assert host.get_capability("decision.backends") == {}
    assert transports[0]._client.is_closed


async def test_disabled_hook_never_resolves_auth_or_constructs_transport(monkeypatch):
    import amplifier_module_provider_azure_openai as azure
    def forbidden(config):
        raise AssertionError("disabled account must not resolve authentication")
    monkeypatch.setattr(azure, "_resolve_azure_account", forbidden)
    assert await decision_hook.mount(coordinator(), {"decisions": {"enabled": False}}) is None


async def test_token_file_is_deferred_refreshed_and_expiry_checked(captured, tmp_path):
    seen, _ = captured
    token = tmp_path / "synthetic-token.json"
    host = coordinator()
    cleanup = await decision_hook.mount(host, {
        "azure_endpoint": "https://resource.services.ai.azure.com",
        "azure_ad_token_file": str(token),
        "decisions": {"enabled": True, "model": "deployment"},
    })
    assert seen == [] and not token.exists()
    backend = host.get_capability("decision.backends")["azure-decisions"]
    for value in ["first", "refreshed"]:
        token.write_text(json.dumps({"accessToken": value, "expires_on": time.time()+120}))
        assert (await backend.decide(REQUEST)).answers["visible"].status == "answered"
    assert [request.headers["authorization"] for request in seen] == ["Bearer first", "Bearer refreshed"]
    token.write_text(json.dumps({"accessToken": "expired", "expires_on": 0}))
    result = await backend.decide(REQUEST)
    assert result.answers["visible"].reason == "credential_unavailable" and len(seen) == 2
    await cleanup()


async def test_default_credential_is_lazy_reused_and_closed(captured, monkeypatch):
    import azure.identity
    seen, _ = captured
    tokens, closed, constructed = [], [], []
    credential = SimpleNamespace(close=lambda: closed.append(True))
    def factory(**kwargs):
        constructed.append(kwargs)
        return credential
    def bearer(owner, scope):
        assert owner is credential and scope == "https://cognitiveservices.azure.com/.default"
        def get():
            tokens.append(True)
            return "fresh-token"
        return get
    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", factory)
    monkeypatch.setattr(azure.identity, "get_bearer_token_provider", bearer)
    host = coordinator()
    cleanup = await decision_hook.mount(host, {
        "azure_endpoint": "https://resource.services.ai.azure.com", "use_default_credential": True,
        "decisions": {"enabled": True, "model": "deployment"},
    })
    assert constructed == [{"exclude_managed_identity_credential": True}]
    assert tokens == [] and seen == []
    await host.get_capability("decision.backends")["azure-decisions"].decide(REQUEST)
    assert tokens == [True] and seen[0].headers["authorization"] == "Bearer fresh-token"
    await cleanup()
    await cleanup()
    assert closed == [True]


async def test_partial_registration_failure_closes_transports_and_rolls_back(captured):
    seen, transports = captured
    host = coordinator()
    def contributor(kind, identity, callback):
        if identity == "second":
            raise RuntimeError("synthetic contributor failure")
    host.register_contributor = contributor
    with pytest.raises(RuntimeError, match="synthetic contributor failure"):
        await decision_hook.mount(host, {
            "azure_endpoint": "https://resource.services.ai.azure.com", "api_key": "fixture",
            "decisions": {"enabled": True, "backends": {
                "first": {"enabled": True, "model": "first"},
                "second": {"enabled": True, "model": "second"},
            }},
        })
    assert host.get_capability("decision.backends") == {} and seen == []
    assert len(transports) == 2 and all(t._client.is_closed for t in transports)


async def test_real_core_mounts_hook_and_cleans_up_without_chat_provider(captured):
    seen, transports = captured
    session = AmplifierSession({
        "session": {"orchestrator": {"module": "loop-streaming"}, "context": {"module": "context-simple"}},
        "providers": [],
        "hooks": [{"module": "hooks-azure-openai-decisions", "config": {
            "azure_endpoint": "https://resource.services.ai.azure.com", "api_key": "fixture",
            "decisions": {"enabled": True, "model": "deployment"},
        }}],
    })
    await session.initialize()
    try:
        assert session.coordinator.get("providers") == {}
        assert seen == []
        backend = session.coordinator.get_capability("decision.backends")["azure-decisions"]
        assert (await backend.decide(REQUEST)).answers["visible"].status == "answered"
    finally:
        await session.cleanup()
    assert session.coordinator.get_capability("decision.backends") == {}
    assert len(transports) == 1 and transports[0]._client.is_closed


async def test_catalog_keeps_ordinary_azure_default_and_declares_exact_hook_condition():
    row = decisions.catalog()
    assert row["runtime"] == {"kind": "hook", "module": "hooks-azure-openai-decisions", "when": {"chat_enabled": False}}
    assert not row["backends"][0].get("decision_only")
    assert row["backends"][0]["connection"]["auth"] == "optional-key"


@pytest.mark.parametrize("close_fails", [False, True])
async def test_bearer_construction_failure_unwinds_before_optional_fallback(
    captured, monkeypatch, caplog, close_fails
):
    import azure.identity
    from amplifier_decisions.hooks import UnsafeAccountCleanup

    seen, transports = captured
    closed, outcomes = [], []
    def close():
        closed.append(True)
        if close_fails:
            raise RuntimeError("PRIVATE CLOSE DETAIL")
    credential = SimpleNamespace(close=close)
    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", lambda **kwargs: credential)
    def broken_bearer(*args):
        raise RuntimeError("PRIVATE AUTH DETAIL")
    monkeypatch.setattr(azure.identity, "get_bearer_token_provider", broken_bearer)
    host = coordinator()
    async def lifecycle(binding_id, module, outcome, error):
        outcomes.append(outcome)
        return True  # A host cannot permit fallback after unsafe cleanup.
    host.register_capability("decision.account_lifecycle", lifecycle)
    config = {
        "azure_endpoint": "https://resource.services.ai.azure.com", "use_default_credential": True,
        "decisions": {"enabled": True, "model": "deployment"},
        "_decision_account": {"binding_id": "synthetic-account"},
    }
    if close_fails:
        with pytest.raises(UnsafeAccountCleanup, match="construction cleanup failed") as failed:
            await decision_hook.mount(host, config)
        assert "PRIVATE" not in str(failed.value)
        assert outcomes == ["unsafe"]
    else:
        assert await decision_hook.mount(host, config) is None
        assert outcomes == ["unavailable"]
    assert closed == [True] and seen == [] and transports == []
    assert host.get_capability("decision.backends") is None
    assert "PRIVATE" not in caplog.text
