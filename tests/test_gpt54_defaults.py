"""Tests for GPT-5.4 default values in Azure OpenAI provider.

Verifies that the provider defaults have been updated from GPT-5.1 to GPT-5.4
with the correct context_window and model values.
"""

from amplifier_module_provider_azure_openai import (
    _create_azure_provider,
    _get_azure_provider_info,
)


def test_provider_info_model_default_is_gpt54():
    """ProviderInfo defaults must specify gpt-5.4 as the model."""
    info = _get_azure_provider_info()
    assert info.defaults["model"] == "gpt-5.4"


def test_provider_info_context_window_is_272000():
    """ProviderInfo defaults must specify context_window of 272000."""
    info = _get_azure_provider_info()
    assert info.defaults["context_window"] == 272000


def test_provider_info_max_output_tokens_unchanged():
    """ProviderInfo max_output_tokens should remain 128000."""
    info = _get_azure_provider_info()
    assert info.defaults["max_output_tokens"] == 128000


def test_default_deployment_is_gpt54(mock_openai_provider_cls):
    """The default_deployment fallback must be gpt-5.4."""
    provider = _create_azure_provider(
        mock_openai_provider_cls,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config={},
    )
    assert provider.default_model == "gpt-5.4"


def test_deployment_name_only_config_sets_default_model(mock_openai_provider_cls):
    """A config with ONLY deployment_name set must resolve default_model from it.

    Regression test: the setup wizard's configure_provider() short-circuit
    (amplifier-app-cli's provider_config_utils.py) writes default_model FROM
    the user's deployment_name answer, so wizard-driven configs already
    worked. But a hand-written config setting only deployment_name (no
    default_model, no default_deployment) previously fell all the way through
    to the hardcoded "gpt-5.4" default, silently ignoring the user's
    deployment_name -- even though deployment_name is an advertised,
    documented ConfigField. See EXTRA_KNOWN_CONFIG_KEYS which already
    declares deployment_name as a recognized (not "unrecognized") key.
    """
    provider = _create_azure_provider(
        mock_openai_provider_cls,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config={"deployment_name": "my-custom-gpt5-deployment"},
    )
    assert provider.default_model == "my-custom-gpt5-deployment"


def test_default_model_takes_priority_over_deployment_name(mock_openai_provider_cls):
    """default_model, when explicitly set, must win over deployment_name."""
    provider = _create_azure_provider(
        mock_openai_provider_cls,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config={
            "default_model": "explicit-model",
            "deployment_name": "some-deployment",
        },
    )
    assert provider.default_model == "explicit-model"


def test_deployment_name_takes_priority_over_default_deployment(
    mock_openai_provider_cls,
):
    """deployment_name must win over the legacy default_deployment alias."""
    provider = _create_azure_provider(
        mock_openai_provider_cls,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config={
            "deployment_name": "new-style-deployment",
            "default_deployment": "legacy-deployment",
        },
    )
    assert provider.default_model == "new-style-deployment"


def test_readme_has_no_gpt51_references():
    """README.md must not contain any gpt-5.1 references."""
    import pathlib

    readme_path = pathlib.Path(__file__).parent.parent / "README.md"
    content = readme_path.read_text()
    assert "gpt-5.1" not in content, "README.md still contains gpt-5.1 references"
