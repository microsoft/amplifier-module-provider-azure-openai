"""Azure media support facts; offline only, no deployment-name inference."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from amplifier_module_provider_azure_openai import AzureOpenAIProvider, _create_azure_provider


@pytest.mark.parametrize("model", ["Customer-Blue", "gpt-5.4", "gpt-4o", "vision", "", "unknown"])
@pytest.mark.parametrize("options", [None, {}, {"model": "gpt-4o", "vision": True, "media_support": {"v": 1, "positions": ["user", "tool"]}}])
def test_aliases_and_options_cannot_qualify_azure_vision(model, options, mock_openai_provider_cls):
    config = {"deployments": ["gpt-4o", "vision"], "default_model": "gpt-4o"}
    # Simulate a future/installed parent advertising public OpenAI model support.
    class VisionBase(mock_openai_provider_cls):
        def get_media_request_support(self, **kwargs):
            raise AssertionError("Azure must not inherit public-model vision inference")

    setup = AzureOpenAIProvider(config)
    runtime = _create_azure_provider(VisionBase, config=config)
    original = deepcopy(options)
    with patch("amplifier_module_provider_azure_openai.AsyncOpenAI", side_effect=AssertionError("support lookup created a client")):
        for provider in (setup, runtime):
            support = provider.get_media_request_support(model=model, request_options=options)
            assert support == {
                "v": 1, "positions": [], "mime_types": ["image/png", "image/jpeg"],
                "reason": "Azure deployment image support is unqualified on this route; no pixels supplied",
            }
    assert options == original
    assert runtime._azure_client is None


def test_support_is_a_fresh_value_and_matches_setup_runtime(mock_openai_provider_cls):
    setup = AzureOpenAIProvider()
    runtime = _create_azure_provider(mock_openai_provider_cls)
    first = runtime.get_media_request_support(model="customer")
    first["positions"].append("tool")
    first["mime_types"].clear()
    assert runtime.get_media_request_support(model="customer") == setup.get_media_request_support(model="customer")
    assert runtime.get_media_request_support(model="customer")["positions"] == []
    assert type(runtime).get_media_request_support is type(setup).get_media_request_support


def test_public_support_needs_no_openai_provider_import(monkeypatch):
    def forbidden():
        raise AssertionError("support facts must not import the inference provider")

    monkeypatch.setattr("amplifier_module_provider_azure_openai._get_openai_provider_class", forbidden)
    assert AzureOpenAIProvider().get_media_request_support(model="gpt-4o")["positions"] == []
