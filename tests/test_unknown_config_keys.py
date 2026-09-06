"""Regression coverage for the provider-openai unknown-config-key false positive.

Background: amplifier-module-provider-openai#69 added a generic
unknown-config-key sweep (`_warn_unknown_config_keys` / `_KNOWN_CONFIG_KEYS`)
that warns at construction time about any config key the module doesn't
read. Because `AzureOpenAIProvider` SUBCLASSES `OpenAIProvider` and passes
its own config straight through the same `config` dict the parent
constructor reads (see `_create_azure_provider` in
`amplifier_module_provider_azure_openai/__init__.py`), every legitimate
azure config key (azure_endpoint, api_version, use_managed_identity, ...)
tripped the sweep as "unrecognized" -- a false positive on valid config.

amplifier-module-provider-openai#70 added `EXTRA_KNOWN_CONFIG_KEYS`, a
class-level extension point subclasses use to declare their own recognized
keys. This module declares `_AZURE_EXTRA_CONFIG_KEYS` via that mechanism
(see `amplifier_module_provider_azure_openai/__init__.py`).

These tests run against the REAL `OpenAIProvider` from
amplifier-module-provider-openai (not the local `MockOpenAIProvider` stand-in
used elsewhere in this test suite), because the sweep itself lives in that
module -- a mock base class would not exercise it at all.
"""

import logging

from amplifier_module_provider_azure_openai import _create_azure_provider
from amplifier_module_provider_openai import OpenAIProvider

_UNKNOWN_KEY_MARKER = "Unrecognized config key"


def _make_real_azure_provider(config: dict):
    return _create_azure_provider(
        OpenAIProvider,
        base_url="https://example-resource.openai.azure.com/openai/v1/",
        api_key="test-key",
        config=config,
    )


class TestAzureUnknownConfigKeyFalsePositive:
    def test_full_realistic_azure_config_produces_zero_unknown_key_warnings(
        self, caplog
    ):
        """A realistic, fully-populated azure config -- every key this
        module itself reads or declares, PLUS the infrastructure keys an
        app/kernel may place alongside a provider's config -- must not
        produce a single unknown-key warning.

        This is the primary regression test for the false positive: before
        amplifier-module-provider-openai#70 (and before this module declared
        EXTRA_KNOWN_CONFIG_KEYS), this exact config warned on
        'api_version', 'azure_endpoint', 'deployment_name', and
        'use_managed_identity' (the reported bug, verbatim).
        """
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")

        real_config = {
            # infrastructure / entry metadata (already known to provider-openai)
            "id": "azure-primary",
            "module": "provider-azure-openai",
            "source": "~/.amplifier/settings.yaml",
            "api_key": "${AZURE_OPENAI_API_KEY}",
            "priority": 10,
            "default_model": "gpt-5.4",
            # azure's own keys (declared via EXTRA_KNOWN_CONFIG_KEYS)
            "azure_endpoint": "https://example-resource.openai.azure.com",
            "api_version": "2024-10-01-preview",
            "use_managed_identity": False,
            "use_default_credential": False,
            "managed_identity_client_id": "11111111-2222-3333-4444-555555555555",
            "deployment_name": "gpt-5-4-prod",
            "deployment_type": "PAYG",
            "default_deployment": "gpt-5.4",
        }

        _make_real_azure_provider(real_config)

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert matches == [], (
            f"unexpected unknown-key warning(s) on a legitimate azure config: {matches}"
        )

    def test_genuine_typo_still_warns_with_suggestion(self, caplog):
        """A real typo of azure's own key must still warn -- the fix widens
        the recognized set, it does not disable the sweep for this
        consumer."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")

        _make_real_azure_provider({"azure_endpont": "https://example.openai.azure.com"})

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert len(matches) == 1, f"expected exactly one sweep warning, got: {matches}"
        message = matches[0]
        assert "'azure_endpont'" in message
        assert "azure_endpoint" in message
        assert "did you mean" in message.lower()

    def test_unrelated_typo_still_warns(self, caplog):
        """A key unrelated to any known or azure-declared key still warns,
        confirming the sweep is still active for this consumer (not
        silenced entirely)."""
        caplog.set_level(logging.WARNING, logger="amplifier_module_provider_openai")

        _make_real_azure_provider({"totally_bogus_azure_setting_xyz": True})

        matches = [
            r.message for r in caplog.records if _UNKNOWN_KEY_MARKER in r.message
        ]
        assert len(matches) == 1
        assert "'totally_bogus_azure_setting_xyz'" in matches[0]

    def test_declared_extra_keys_match_module_constant(self):
        """Sanity check: the class attribute wired onto the dynamically
        created provider class is exactly the module's own audited key set
        (guards against the two ever drifting apart)."""
        from amplifier_module_provider_azure_openai import _AZURE_EXTRA_CONFIG_KEYS

        provider = _make_real_azure_provider({})
        assert provider.EXTRA_KNOWN_CONFIG_KEYS == _AZURE_EXTRA_CONFIG_KEYS
        assert _AZURE_EXTRA_CONFIG_KEYS == frozenset(
            {
                # Read by THIS module's __init__ (bounded close()), not
                # inherited -- the base class is resolved dynamically and may
                # be a provider-openai build that does not know the key.
                "close_timeout",
                "azure_endpoint",
                "api_version",
                "use_managed_identity",
                "use_default_credential",
                "managed_identity_client_id",
                "deployment_name",
                "deployment_type",
                "default_deployment",
            }
        )
