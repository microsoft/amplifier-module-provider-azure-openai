"""
Azure OpenAI provider module for Amplifier.

Wraps the OpenAI provider implementation while adding Azure-specific authentication.
"""

__all__ = ["mount", "AzureOpenAIProvider"]

import logging
import os
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any

from amplifier_core import ModuleCoordinator
from amplifier_module_provider_openai import OpenAIProvider
from openai import AsyncOpenAI

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
                credential = DefaultAzureCredential()
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


class AzureOpenAIProvider(OpenAIProvider):
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
        self.default_model = self.config.get("default_model") or self.config.get("default_deployment", "gpt-5")

        logger.debug(
            "AzureOpenAIProvider configured (auth=%s, default_model=%s, base_url=%s)",
            self._auth_mode,
            self.default_model,
            base_url,
        )


def _get_bool(config: dict[str, Any], key: str, env_value: str | None) -> bool:
    """Resolve a boolean configuration value from config dict or environment."""
    if key in config:
        return bool(config[key])
    if env_value is None:
        return False
    return env_value.lower() in {"true", "1", "yes"}
