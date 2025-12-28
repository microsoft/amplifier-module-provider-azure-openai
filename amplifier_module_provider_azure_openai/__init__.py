"""
Azure OpenAI provider module for Amplifier.

Wraps the OpenAI provider implementation while adding Azure-specific authentication.
"""

__all__ = ["mount", "AzureOpenAIProvider"]

# Amplifier module metadata
__amplifier_module_type__ = "provider"

import logging
import os
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from amplifier_core import ConfigField
from amplifier_core import ModelInfo
from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderInfo
from openai import AsyncOpenAI

# Runtime dependency: OpenAI provider module
# This is imported at runtime rather than as a build dependency to avoid the
# transitive dependency issue where `uv pip install -e` would install the
# openai provider non-editable, overwriting any existing editable install.
try:
    from amplifier_module_provider_openai import OpenAIProvider

    OPENAI_PROVIDER_AVAILABLE = True
except ImportError:
    OPENAI_PROVIDER_AVAILABLE = False
    OpenAIProvider = None  # type: ignore[misc, assignment]

try:
    from azure.identity import DefaultAzureCredential
    from azure.identity import ManagedIdentityCredential
    from azure.identity import get_bearer_token_provider

    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Azure OpenAI provider and return a cleanup coroutine if successful."""
    # Check runtime dependency on OpenAI provider module
    if not OPENAI_PROVIDER_AVAILABLE:
        logger.error(
            "Azure OpenAI provider requires the OpenAI provider module. "
            "Ensure 'provider-openai' is installed before 'provider-azure-openai'. "
            "Run 'amplifier init' or install manually."
        )
        return None

    config = config or {}

    azure_endpoint = (
        config.get("azure_endpoint")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_BASE_URL")
    )

    if not azure_endpoint:
        logger.warning("No azure_endpoint found for Azure OpenAI provider")
        return None

    base_url = f"{azure_endpoint.rstrip('/')}/openai/v1/"
    api_version = config.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-10-01-preview"

    api_key = config.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")

    use_managed_identity = _get_bool(config, "use_managed_identity", os.environ.get("AZURE_USE_MANAGED_IDENTITY"))
    use_default_credential = _get_bool(config, "use_default_credential", os.environ.get("AZURE_USE_DEFAULT_CREDENTIAL"))
    managed_identity_client_id = config.get("managed_identity_client_id") or os.environ.get(
        "AZURE_MANAGED_IDENTITY_CLIENT_ID"
    )

    token_provider: Callable[[], Awaitable[str]] | None = None

    if api_key:
        auth_summary = "api key"
    elif use_managed_identity or use_default_credential:
        if not AZURE_IDENTITY_AVAILABLE:
            logger.error(
                "Managed identity authentication requires the 'azure-identity' package. "
                "Install with: pip install azure-identity"
            )
            return None

        try:
            if use_default_credential:
                credential = DefaultAzureCredential(exclude_managed_identity_credential=True)
                auth_summary = "DefaultAzureCredential"
            elif managed_identity_client_id:
                credential = ManagedIdentityCredential(client_id=managed_identity_client_id)
                auth_summary = f"ManagedIdentityCredential (client_id={managed_identity_client_id})"
            else:
                credential = ManagedIdentityCredential()
                auth_summary = "ManagedIdentityCredential (system-assigned)"

            sync_token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

            async def async_token_provider() -> str:
                return sync_token_provider()

            token_provider = async_token_provider
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Failed to initialize Azure credential: %s", exc)
            return None
    else:
        logger.warning("No authentication method configured for Azure OpenAI provider")
        return None

    provider = AzureOpenAIProvider(
        base_url=base_url,
        api_key=api_key,
        token_provider=token_provider,
        config=config,
        coordinator=coordinator,
    )

    await coordinator.mount("providers", provider, name=provider.name)
    logger.info(
        "Mounted AzureOpenAIProvider (Responses API, endpoint: %s, version: %s, auth: %s)",
        base_url,
        api_version,
        auth_summary,
    )

    async def cleanup():
        if hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


# Conditionally define AzureOpenAIProvider based on whether OpenAI provider is available.
# This allows the module to be imported even when the dependency is missing,
# with a clear error at mount() time rather than import time.
if OPENAI_PROVIDER_AVAILABLE:

    class AzureOpenAIProvider(OpenAIProvider):  # type: ignore[misc]
        """Azure-specific wrapper around the OpenAI provider implementation."""

        name = "azure-openai"
        api_label = "Azure OpenAI"

        def __init__(
            self,
            *,
            base_url: str,
            api_key: str | None = None,
            token_provider: Callable[[], Awaitable[str]] | None = None,
            config: dict[str, Any] | None = None,
            coordinator: ModuleCoordinator | None = None,
        ):
            if not base_url:
                raise ValueError("base_url is required")
            if api_key is None and token_provider is None:
                raise ValueError("api_key or token_provider must be provided")

            client = AsyncOpenAI(base_url=base_url, api_key=api_key or token_provider)

            super().__init__(api_key=api_key, config=config, coordinator=coordinator, client=client)

            self.base_url = base_url
            self.token_provider = token_provider
            self._auth_mode = "api_key" if api_key else "token_provider"

            # Azure deployments commonly use "default_deployment" config keys
            self.default_model = self.config.get("default_model") or self.config.get("default_deployment", "gpt-5.1")

            logger.debug(
                "AzureOpenAIProvider configured (auth=%s, default_model=%s, base_url=%s)",
                self._auth_mode,
                self.default_model,
                base_url,
            )

        def get_info(self) -> ProviderInfo:
            """Get provider metadata."""
            return ProviderInfo(
                id="azure-openai",
                display_name="Azure OpenAI",
                credential_env_vars=["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
                capabilities=["streaming", "tools", "vision", "reasoning", "batch", "json_mode"],
                defaults={
                    "model": "gpt-5.1",
                    "max_tokens": 16384,
                    "temperature": None,
                    "timeout": 300.0,
                    "context_window": 400000,
                    "max_output_tokens": 128000,
                },
                config_fields=[
                    ConfigField(
                        id="azure_endpoint",
                        display_name="Azure Endpoint",
                        field_type="text",
                        prompt="Enter your Azure OpenAI endpoint URL",
                        env_var="AZURE_OPENAI_ENDPOINT",
                    ),
                    ConfigField(
                        id="api_key",
                        display_name="API Key",
                        field_type="secret",
                        prompt="Enter your Azure OpenAI API key (or leave empty for managed identity)",
                        env_var="AZURE_OPENAI_API_KEY",
                        required=False,
                    ),
                    ConfigField(
                        id="api_version",
                        display_name="API Version",
                        field_type="text",
                        prompt="Enter API version",
                        env_var="AZURE_OPENAI_API_VERSION",
                        default="2024-10-01-preview",
                        required=False,
                    ),
                    ConfigField(
                        id="use_managed_identity",
                        display_name="Use Managed Identity",
                        field_type="boolean",
                        prompt="Use Azure Managed Identity for authentication?",
                        env_var="AZURE_USE_MANAGED_IDENTITY",
                        default="false",
                        required=False,
                    ),
                    ConfigField(
                        id="managed_identity_client_id",
                        display_name="Managed Identity Client ID",
                        field_type="text",
                        prompt="Enter Managed Identity Client ID (for user-assigned identity)",
                        env_var="AZURE_MANAGED_IDENTITY_CLIENT_ID",
                        required=False,
                        show_when={"use_managed_identity": "true"},
                    ),
                    ConfigField(
                        id="deployment_name",
                        display_name="Deployment Name",
                        field_type="text",
                        prompt="Enter your Azure OpenAI deployment name",
                        required=False,
                    ),
                ],
            )

        async def list_models(self) -> list[ModelInfo]:
            """
            List available Azure OpenAI models.

            Returns empty list since Azure deployments are customer-specific.
            User should specify model/deployment name directly.
            """
            return []

else:
    # Placeholder class when OpenAI provider is not available
    class AzureOpenAIProvider:  # type: ignore[no-redef]
        """Placeholder - requires amplifier-module-provider-openai to be installed."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "Azure OpenAI provider requires the OpenAI provider module. "
                "Ensure 'provider-openai' is installed before 'provider-azure-openai'."
            )


def _get_bool(config: dict[str, Any], key: str, env_value: str | None) -> bool:
    """Resolve a boolean configuration value from config dict or environment."""
    if key in config:
        return bool(config[key])
    if env_value is None:
        return False
    return env_value.lower() in {"true", "1", "yes"}
