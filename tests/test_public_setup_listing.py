"""Public setup class and mounted runtime share a credential-free local catalog."""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from amplifier_core.llm_errors import ProviderUnavailableError
import amplifier_module_provider_azure_openai as azure


@pytest.mark.asyncio
async def test_public_setup_needs_no_runtime_account_or_client(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("setup must not load runtime or acquire credentials")

    for name in (
        "_get_openai_provider_class",
        "_resolve_azure_account",
        "_create_azure_provider",
        "AsyncOpenAI",
    ):
        monkeypatch.setattr(azure, name, forbidden)
    config = {
        "deployments": ["Customer-Blue", "Research/Alias"],
        "model_filters": {"include": ["customer-*"]},
        "use_default_credential": True,
        "azure_ad_token_file": "/must-not-read",
    }
    p = azure.AzureOpenAIProvider(config=config)
    config["deployments"].clear()
    config["model_filters"]["include"].clear()
    assert azure.AzureOpenAIProvider.get_info().id == "azure-openai"
    assert [m.id for m in await p.list_models()] == ["Customer-Blue"]
    assert [m.id for m in await p.list_models(filters=None)] == ["Customer-Blue"]
    assert [m.id for m in await p.list_models(filters={})] == [
        "Customer-Blue",
        "Research/Alias",
    ]
    assert [m.id for m in await p.list_models(filters={"include": ["RESEARCH/*"]})] == [
        "Research/Alias"
    ]
    assert await p.list_models(filters={"include": []}) == []
    info = p.get_model_list_info()
    assert info["source"] == "configured"
    info["default_filters"]["include"].clear()
    info["filters_schema"]["properties"].clear()
    assert p.get_model_list_info()["default_filters"]["include"] == ["customer-*"]
    assert "include" in p.get_model_list_info()["filters_schema"]["properties"]
    assert not hasattr(p, "complete")
    p = azure.AzureOpenAIProvider()
    assert p.get_model_list_info()["source"] == "unsupported"
    with pytest.raises(ProviderUnavailableError):
        await p.list_models()
    assert await azure.AzureOpenAIProvider({"deployments": []}).list_models() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filters",
    [
        None,
        {},
        {"include": []},
        {"include": ["research/*"]},
        {"exclude": ["customer-*"]},
    ],
)
async def test_setup_runtime_use_identical_listing_methods(
    mock_openai_provider_cls, filters
):
    config = {
        "deployments": ["Customer-Blue", "Research/Alias"],
        "model_filters": {"include": ["customer-*"]},
    }
    setup = azure.AzureOpenAIProvider(config)
    runtime = azure._create_azure_provider(mock_openai_provider_cls, config=config)
    assert type(setup).list_models is type(runtime).list_models
    assert type(setup).get_model_list_info is type(runtime).get_model_list_info
    assert setup.get_model_list_info() == runtime.get_model_list_info()
    assert await setup.list_models(filters=filters) == await runtime.list_models(
        filters=filters
    )


@pytest.mark.parametrize(
    "config",
    [
        {"deployments": None},
        {"deployments": [""]},
        {"model_filters": None},
        {"model_filters": {"include": "*"}},
    ],
)
def test_public_setup_rejects_invalid_configuration(config):
    with pytest.raises((ValueError, TypeError)):
        azure.AzureOpenAIProvider(config)


def test_public_metadata_and_listing_without_optional_provider_or_identity_import():
    # New interpreter: no already-imported OpenAI provider can mask an eager import.
    source = """
import asyncio, importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("amplifier_module_provider_openai", "azure.identity")):
            raise AssertionError("Optional runtime dependency imported during setup: " + fullname)
sys.meta_path.insert(0, BlockOptional())
from amplifier_module_provider_azure_openai import AzureOpenAIProvider
assert AzureOpenAIProvider.get_info().id == "azure-openai"
p = AzureOpenAIProvider({"deployments": ["Customer-Blue"], "use_default_credential": True})
assert p.get_model_list_info()["source"] == "configured"
assert [m.id for m in asyncio.run(p.list_models())] == ["Customer-Blue"]
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_mount_still_requires_runtime_dependency_before_account(monkeypatch):
    monkeypatch.setattr(azure, "_get_openai_provider_class", lambda: None)

    def forbidden(*args):
        raise AssertionError("must not resolve credentials without runtime provider")

    monkeypatch.setattr(azure, "_resolve_azure_account", forbidden)
    assert await azure.mount(object(), {"deployments": ["Customer-Blue"]}) is None


@pytest.mark.asyncio
async def test_invalid_listing_during_mount_releases_account(
    monkeypatch, mock_openai_provider_cls
):
    account = SimpleNamespace(
        base_url="https://azure.invalid/openai/v1/",
        api_version="fixture",
        auth_summary="fixture",
        _api_key="test-only",
        _token_provider=None,
        config={"deployments": [""]},
        close=AsyncMock(),
    )
    monkeypatch.setattr(
        azure, "_get_openai_provider_class", lambda: mock_openai_provider_cls
    )
    monkeypatch.setattr(azure, "_resolve_azure_account", lambda config: account)
    with pytest.raises(ValueError, match="deployments"):
        await azure.mount(object(), account.config)
    account.close.assert_awaited_once()
