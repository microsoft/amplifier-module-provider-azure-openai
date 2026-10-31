"""Azure deployment capability qualifies inherited OpenAI wire optimization."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.message_models import ChatRequest, Message
from amplifier_module_provider_openai import OpenAIProvider
from amplifier_module_provider_azure_openai import _create_azure_provider


@pytest.mark.parametrize('model,deployment,supported', [
    ('gpt-6.1-sol', 'Standard', True), ('gpt-5.6-sol', 'Standard', True),
    ('gpt-6.1-sol', 'PTU', False), ('gpt-5.6-sol', 'PTU', False),
    ('my-deployment', 'Standard', False), ('gpt-4o', 'Standard', False),
])
@pytest.mark.parametrize('continued', [False, True])
def test_deployment_boundaries_reach_initial_and_continued_requests(model, deployment, supported, continued):
    supplied = [{'type': 'function_call_output', 'call_id': 'c', 'output': 'result'}]
    provider = _create_azure_provider(OpenAIProvider,
        base_url='https://example.openai.azure.com/openai/v1/', api_key='test-key',
        config={'default_model': model, 'deployment_type': deployment, 'use_streaming': False,
                'max_retries': 0, 'extra_request_params': {'input': supplied}})
    responses = [SimpleNamespace(status='incomplete', id='partial', output=[], incomplete_details=None),
                 SimpleNamespace(status='completed', id='done', output=[], usage=None)]
    if not continued:
        responses = responses[1:]
    provider._azure_client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(side_effect=responses)))
    asyncio.run(provider.complete(ChatRequest(messages=[Message(role='user', content='hello')])))
    calls = provider.client.responses.create.call_args_list
    assert len(calls) == (2 if continued else 1)
    for call in calls:
        assert isinstance(call.kwargs['input'][0]['output'], list) is supported
    assert supplied[0]['output'] == 'result'
