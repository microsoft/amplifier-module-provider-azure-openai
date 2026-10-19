from types import SimpleNamespace
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
