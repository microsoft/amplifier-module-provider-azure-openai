from types import SimpleNamespace
from copy import deepcopy
from unittest.mock import AsyncMock
import httpx
import pytest
from amplifier_decisions import DecisionRequest, Question
from amplifier_module_provider_azure_openai.decisions import FoundryDecisions

REQUEST=DecisionRequest('Visible Save',(Question('visible','predicate','Is Save visible?'),),allow_external_state=True)
@pytest.mark.asyncio
async def test_shared_key_uses_foundry_route_and_bearer_is_refreshed():
    seen=[]
    def handler(req):
        seen.append(req);return httpx.Response(200,json={'model':'microsoft-decision-1','answers':{'visible':{'type':'noul','noul':.99}}})
    provider=SimpleNamespace(config={'azure_endpoint':'https://fixture.services.ai.azure.com/models'},_api_key='saved-key',_token_provider=None)
    transport=FoundryDecisions(provider,{'model':'deployment'})
    transport._client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert (await transport.backend.decide(REQUEST)).answers['visible'].value==.99
        assert seen[0].headers['api-key']=='saved-key' and 'authorization' not in seen[0].headers
        assert str(seen[0].url)=='https://fixture.services.ai.azure.com/providers/microsoft/v1/systemone'
        provider._api_key=None
        tokens=iter(['fresh-one','fresh-two'])
        async def credential():return next(tokens)
        provider._token_provider=credential
        await transport.backend.decide(REQUEST);await transport.backend.decide(REQUEST)
        assert seen[-2].headers['authorization']=='Bearer fresh-one'
        assert seen[-1].headers['authorization']=='Bearer fresh-two'
    finally:await transport.close()

@pytest.mark.parametrize('suffix',['','/','/models','/models/','/providers/microsoft/v1/systemone'])
def test_foundry_portal_urls(suffix):
    provider=SimpleNamespace(config={'azure_endpoint':'https://fixture.services.ai.azure.com'+suffix})
    assert FoundryDecisions(provider,{'model':'deployment'}).endpoint=='https://fixture.services.ai.azure.com/providers/microsoft/v1/systemone'

def test_endpoint_override_does_not_mutate_chat_config():
    provider=SimpleNamespace(config={'azure_endpoint':'https://chat.openai.azure.com'})
    transport=FoundryDecisions(provider,{'model':'deployment','endpoint':'https://gateway.test/prefix'})
    assert transport.endpoint=='https://gateway.test/prefix/providers/microsoft/v1/systemone'
    assert provider.config['azure_endpoint']=='https://chat.openai.azure.com'
    with pytest.raises(ValueError):FoundryDecisions(provider,{'model':'deployment','endpoint':'https://user:secret@gateway.test'})


def mount_coordinator():
    capabilities = {}
    return SimpleNamespace(
        mount=AsyncMock(),
        register_contributor=lambda *args: None,
        get_capability=capabilities.get,
        register_capability=capabilities.__setitem__,
        hooks=SimpleNamespace(emit=AsyncMock(return_value=SimpleNamespace(action="continue"))),
        process_hook_result=AsyncMock(side_effect=lambda result, *args: result),
    )


@pytest.fixture(params=["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_BASE_URL"])
def account_endpoint(monkeypatch, request):
    for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(request.param, "https://environment.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "environment-key")
    return "https://environment.openai.azure.com"


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [None, "account", "decision"])
async def test_mount_reuses_resolved_account_for_decisions(monkeypatch, account_endpoint, override):
    import json
    from amplifier_module_provider_azure_openai import mount
    from amplifier_module_provider_azure_openai import decisions

    config = {
        "default_model": "chat-deployment",
        "decisions": {"enabled": True, "model": "decision-deployment"},
    }
    expected_resource = account_endpoint
    expected_key = "environment-key"
    if override == "account":
        expected_resource = config["azure_endpoint"] = "https://account.openai.azure.com"
        expected_key = config["api_key"] = "account-key"
    decision_resource = expected_resource
    if override == "decision":
        decision_resource = config["decisions"]["endpoint"] = "https://decision-gateway.test/prefix"
    original = deepcopy(config)
    requests = []
    transports = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={
            "model": "microsoft-decision-1",
            "answers": {"visible": {"type": "noul", "noul": .99}},
        })

    class CapturedTransport(FoundryDecisions):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            transports.append(self)

    monkeypatch.setattr(decisions, "FoundryDecisions", CapturedTransport)
    coordinator = mount_coordinator()
    cleanup = await mount(coordinator, config)
    assert cleanup is not None
    try:
        provider = coordinator.mount.await_args.args[1]
        assert provider._base_url == expected_resource + "/openai/v1/"
        assert provider.default_model == "chat-deployment"
        assert provider._azure_client is None
        backend = coordinator.get_capability("decision.backends")["azure-decisions"]
        assert (await backend.decide(REQUEST)).answers["visible"].value == .99
        assert len(requests) == 1
        assert str(requests[0].url) == decision_resource + "/providers/microsoft/v1/systemone"
        assert requests[0].headers["api-key"] == expected_key
        assert "authorization" not in requests[0].headers
        assert json.loads(requests[0].content)["model"] == "decision-deployment"
        assert config == original
    finally:
        await cleanup()
    assert not coordinator.get_capability("decision.backends")
    assert transports[0]._client.is_closed


@pytest.mark.asyncio
async def test_mount_environment_account_without_decisions(monkeypatch, account_endpoint):
    from amplifier_module_provider_azure_openai import mount
    from amplifier_module_provider_azure_openai import decisions

    def unexpected_transport(*args, **kwargs):
        raise AssertionError("Disabled decisions must not construct a transport")

    monkeypatch.setattr(decisions, "FoundryDecisions", unexpected_transport)
    config = {"default_model": "chat-deployment", "decisions": {"enabled": False}}
    original = deepcopy(config)
    coordinator = mount_coordinator()
    cleanup = await mount(coordinator, config)
    assert cleanup is not None
    try:
        provider = coordinator.mount.await_args.args[1]
        assert provider._base_url == account_endpoint + "/openai/v1/"
        assert provider._api_key == "environment-key"
        assert provider._azure_client is None
        assert not coordinator.get_capability("decision.backends")
        assert config == original
    finally:
        await cleanup()
