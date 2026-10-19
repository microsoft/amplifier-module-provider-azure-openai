"""Foundry Decision-1 using the Azure provider's existing account credentials."""
from urllib.parse import urlsplit
import httpx


def validate(config):
    from amplifier_decisions.config import validate as common
    return common(config)


def catalog():
    return {'schema_version': 2, 'provider_module': 'provider-azure-openai', 'validate': validate,
            'backends': [{'kind': 'microsoft-decision-1', 'label': 'Microsoft Decision-1',
                          'default_model': '', 'model_label': 'Deployment name',
                          'endpoint_optional': True, 'configuration_key': 'decisions'}]}


def decision_info():
    return {'schema_version': 1, 'provider': 'azure-openai', 'available': True,
            'requires_explicit_enable': True, 'kinds': ['predicate', 'choice', 'score'],
            'images': False, 'models': [], 'configuration_key': 'decisions'}


class FoundryDecisions:
    def __init__(self, provider, config, coordinator=None):
        from amplifier_decisions.config import endpoint
        from amplifier_decisions.runtime import SystemOneBackend
        resource = endpoint(config.get('endpoint') or provider.config.get('azure_endpoint'))
        url = urlsplit(resource)
        if url.path == '/models' and (url.hostname or '').endswith('.services.ai.azure.com'):
            resource = url._replace(path='').geturl()
        self.endpoint = resource if resource.endswith('/providers/microsoft/v1/systemone') else resource+'/providers/microsoft/v1/systemone'
        self.provider, self._client = provider, None
        self.backend = SystemOneBackend('microsoft-decision-1', config['model'], self.send, coordinator=coordinator)

    async def send(self, content, timeout):
        from amplifier_decisions.runtime import DecisionFailure
        from amplifier_decisions.response import read_json
        headers = {'Content-Type': 'application/json'}
        if self.provider._api_key:
            headers['api-key'] = self.provider._api_key
        elif self.provider._token_provider:
            try:
                headers['Authorization'] = 'Bearer '+await self.provider._token_provider()
            except Exception:
                raise DecisionFailure('unavailable', 'credential_unavailable') from None
        else:
            raise DecisionFailure('unavailable', 'missing_credential')
        # The resource API is distinct from Azure OpenAI's /openai/v1/ SDK
        # route. Authentication is shared; never repurpose a chat client URL.
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=False, trust_env=False)
        try:
            async with self._client.stream('POST', self.endpoint, content=content, headers=headers,
                                           timeout=timeout) as response:
                return await read_json(response)
        except httpx.TimeoutException:
            raise DecisionFailure('unavailable', 'timeout') from None
        except httpx.HTTPError:
            raise DecisionFailure('unavailable', 'transport_error') from None

    async def close(self):
        await self.backend.close()
        if self._client is not None:
            await self._client.aclose()
