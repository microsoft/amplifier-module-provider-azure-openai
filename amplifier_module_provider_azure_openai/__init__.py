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

logger = logging.getLogger(__name__)

# Cached reference to OpenAIProvider (lazy loaded)
_OpenAIProvider: type | None = None
_openai_import_attempted: bool = False


def _get_openai_provider_class() -> type | None:
    """Lazily import and cache the OpenAI provider class.

    This function handles the runtime dependency on the OpenAI provider module.
    The import is done lazily (not at module load time) to handle the case where
    provider modules are installed in alphabetical order during `amplifier init`.

    Returns:
        The OpenAIProvider class if available, None otherwise.
    """
    import importlib
    import sys

    global _OpenAIProvider, _openai_import_attempted

    # Return cached value if we've already successfully imported
    if _OpenAIProvider is not None:
        return _OpenAIProvider

    # Clear any cached import failures - the module might have been installed
    # since this module was first loaded (e.g., during provider auto-install)
    importlib.invalidate_caches()

    # Remove any cached failed import attempts from sys.modules
    # This handles the case where an earlier import attempt failed
    module_name = "amplifier_module_provider_openai"
    if module_name in sys.modules and sys.modules[module_name] is None:
        del sys.modules[module_name]

    # Retry import each time if previous attempt failed
    # (the module might have been installed since last attempt)
    try:
        from amplifier_module_provider_openai import OpenAIProvider

        _OpenAIProvider = OpenAIProvider
        _openai_import_attempted = True
        return _OpenAIProvider
    except ImportError:
        _openai_import_attempted = True
        return None


def _is_azure_identity_available() -> bool:
    """Check if azure-identity package is available."""
    try:
        from azure.identity import DefaultAzureCredential  # noqa: F401

        return True
    except ImportError:
        return False


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Azure OpenAI provider and return a cleanup coroutine if successful."""
    # Check runtime dependency on OpenAI provider module (lazy check)
    OpenAIProviderClass = _get_openai_provider_class()
    if OpenAIProviderClass is None:
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
    api_version = (
        config.get("api_version")
        or os.environ.get("AZURE_OPENAI_API_VERSION")
        or "2024-10-01-preview"
    )

    api_key = (
        config.get("api_key")
        or os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("AZURE_OPENAI_KEY")
    )

    use_managed_identity = _get_bool(
        config, "use_managed_identity", os.environ.get("AZURE_USE_MANAGED_IDENTITY")
    )
    use_default_credential = _get_bool(
        config, "use_default_credential", os.environ.get("AZURE_USE_DEFAULT_CREDENTIAL")
    )
    managed_identity_client_id = config.get(
        "managed_identity_client_id"
    ) or os.environ.get("AZURE_MANAGED_IDENTITY_CLIENT_ID")

    token_provider: Callable[[], Awaitable[str]] | None = None

    if api_key:
        auth_summary = "api key"
    elif use_managed_identity or use_default_credential:
        if not _is_azure_identity_available():
            logger.error(
                "Managed identity authentication requires the 'azure-identity' package. "
                "Install with: pip install azure-identity"
            )
            return None

        try:
            from azure.identity import DefaultAzureCredential
            from azure.identity import ManagedIdentityCredential
            from azure.identity import get_bearer_token_provider

            if use_default_credential:
                credential = DefaultAzureCredential(
                    exclude_managed_identity_credential=True
                )
                auth_summary = "DefaultAzureCredential"
            elif managed_identity_client_id:
                credential = ManagedIdentityCredential(
                    client_id=managed_identity_client_id
                )
                auth_summary = f"ManagedIdentityCredential (client_id={managed_identity_client_id})"
            else:
                credential = DefaultAzureCredential()
                auth_summary = (
                    "DefaultAzureCredential (with managed identity preference)"
                )

            sync_token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )

            async def async_token_provider() -> str:
                return sync_token_provider()

            token_provider = async_token_provider
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.error("Failed to initialize Azure credential: %s", exc)
            return None
    else:
        logger.warning("No authentication method configured for Azure OpenAI provider")
        return None

    # Create the provider instance using the dynamically loaded class
    provider = _create_azure_provider(
        OpenAIProviderClass,
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
        if hasattr(provider, "client") and hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


def _create_azure_provider(
    base_class: type,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    token_provider: Callable[[], Awaitable[str]] | None = None,
    config: dict[str, Any] | None = None,
    coordinator: ModuleCoordinator | None = None,
) -> Any:
    """Create an AzureOpenAIProvider instance that inherits from OpenAIProvider.

    This factory function creates the provider class dynamically at runtime,
    allowing us to defer the inheritance until OpenAIProvider is actually available.
    """

    class _AzureOpenAIProvider(base_class):  # type: ignore[valid-type,misc]
        """Azure-specific wrapper around the OpenAI provider implementation."""

        name = "azure-openai"
        api_label = "Azure OpenAI"

        def __init__(
            self,
            *,
            base_url: str | None = None,
            api_key: str | None = None,
            token_provider: Callable[[], Awaitable[str]] | None = None,
            config: dict[str, Any] | None = None,
            coordinator: ModuleCoordinator | None = None,
        ):
            """Initialize Azure OpenAI provider."""
            # Store for lazy client creation
            self._base_url = base_url
            self._token_provider = token_provider
            self._azure_client: AsyncOpenAI | None = None

            # Call parent with no client - we override the client property
            super().__init__(
                api_key=api_key, config=config, coordinator=coordinator, client=None
            )

            # Override base_url from parent
            self.base_url = base_url
            self.token_provider = token_provider
            self._auth_mode = (
                "api_key" if api_key else "token_provider" if token_provider else "none"
            )

            # Azure deployments commonly use "default_deployment" config keys
            self.default_model = self.config.get("default_model") or self.config.get(
                "default_deployment", "gpt-5.1"
            )

            if base_url:
                logger.debug(
                    "AzureOpenAIProvider configured (auth=%s, default_model=%s, base_url=%s)",
                    self._auth_mode,
                    self.default_model,
                    base_url,
                )

        @property
        def client(self) -> AsyncOpenAI:
            """Lazily initialize the Azure OpenAI client on first access."""
            if self._azure_client is None:
                if not self._base_url:
                    raise ValueError("base_url is required for API calls")
                if self._api_key is None and self._token_provider is None:
                    raise ValueError(
                        "api_key or token_provider must be provided for API calls"
                    )
                self._azure_client = AsyncOpenAI(
                    base_url=self._base_url,
                    api_key=self._api_key or self._token_provider,
                )
            return self._azure_client

        def get_info(self) -> ProviderInfo:
            """Get provider metadata."""
            return _get_azure_provider_info()

        async def list_models(self) -> list[ModelInfo]:
            """List available Azure OpenAI models.

            Returns empty list since Azure deployments are customer-specific.
            """
            return []

    return _AzureOpenAIProvider(
        base_url=base_url,
        api_key=api_key,
        token_provider=token_provider,
        config=config,
        coordinator=coordinator,
    )


def _get_azure_provider_info() -> ProviderInfo:
    """Get Azure OpenAI provider metadata.

    This is a standalone function so it can be called without needing
    the OpenAI provider to be available (for discovery/display purposes).
    """
    return ProviderInfo(
        id="azure-openai",
        display_name="Azure OpenAI",
        credential_env_vars=["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        capabilities=["streaming", "tools", "reasoning", "batch", "json_mode"],
        defaults={
            "model": "gpt-5.1",
            "max_tokens": 16384,
            "temperature": None,
            "timeout": 600.0,
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


class AzureOpenAIProvider:
    """Azure OpenAI provider - requires OpenAI provider module at runtime.

    This class provides static metadata via get_info() for discovery purposes,
    but requires the OpenAI provider module to be installed for actual use.
    The real implementation is created dynamically in mount() to handle
    the runtime dependency properly.
    """

    name = "azure-openai"
    api_label = "Azure OpenAI"

    @staticmethod
    def get_info() -> ProviderInfo:
        """Get provider metadata.

        This works without the OpenAI provider being installed,
        allowing provider discovery to succeed.
        """
        return _get_azure_provider_info()


def _get_bool(config: dict[str, Any], key: str, env_value: str | None) -> bool:
    """Resolve a boolean configuration value from config dict or environment."""
    if key in config:
        return bool(config[key])
    if env_value is None:
        return False
    return env_value.lower() in {"true", "1", "yes"}
