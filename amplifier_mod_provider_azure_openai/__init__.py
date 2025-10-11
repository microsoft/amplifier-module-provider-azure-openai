"""
Azure OpenAI provider module for Amplifier.
Integrates with Azure OpenAI Service using deployment names.
"""

import logging
import os
from typing import Any

from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderResponse
from amplifier_core import ToolCall
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ToolCallContent
from openai import AsyncAzureOpenAI

# Try to import Azure Identity for managed identity support
try:
    from azure.identity import DefaultAzureCredential
    from azure.identity import ManagedIdentityCredential

    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Azure OpenAI provider."""
    config = config or {}

    # Get Azure endpoint from config or environment (try both env var names)
    azure_endpoint = (
        config.get("azure_endpoint")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_BASE_URL")
    )

    if not azure_endpoint:
        logger.warning("No azure_endpoint found for Azure OpenAI provider")
        return None

    # Get API version from config or environment with sensible default
    api_version = config.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-02-15-preview"

    # Get authentication configuration
    api_key = config.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")
    azure_ad_token = config.get("azure_ad_token") or os.environ.get("AZURE_OPENAI_AD_TOKEN")

    # Get managed identity configuration from config or environment
    use_managed_identity = config.get("use_managed_identity", False)
    if not use_managed_identity:
        env_val = os.environ.get("AZURE_USE_MANAGED_IDENTITY", "").lower()
        use_managed_identity = env_val in ("true", "1", "yes")

    managed_identity_client_id = config.get("managed_identity_client_id") or os.environ.get(
        "AZURE_MANAGED_IDENTITY_CLIENT_ID"
    )

    use_default_credential = config.get("use_default_credential", False)
    if not use_default_credential:
        env_val = os.environ.get("AZURE_USE_DEFAULT_CREDENTIAL", "").lower()
        use_default_credential = env_val in ("true", "1", "yes")

    # Get provider configuration from config or environment
    # These will be passed to the AzureOpenAIProvider via the config dict
    if "default_deployment" not in config and os.environ.get("AZURE_OPENAI_DEFAULT_DEPLOYMENT"):
        config["default_deployment"] = os.environ.get("AZURE_OPENAI_DEFAULT_DEPLOYMENT")

    if "default_model" not in config and os.environ.get("AZURE_OPENAI_DEFAULT_MODEL"):
        config["default_model"] = os.environ.get("AZURE_OPENAI_DEFAULT_MODEL")

    max_tokens_env = os.environ.get("AZURE_OPENAI_MAX_TOKENS")
    if "max_tokens" not in config and max_tokens_env:
        try:
            config["max_tokens"] = int(max_tokens_env)
        except ValueError:
            logger.warning("Invalid AZURE_OPENAI_MAX_TOKENS value, using default")

    temperature_env = os.environ.get("AZURE_OPENAI_TEMPERATURE")
    if "temperature" not in config and temperature_env:
        try:
            config["temperature"] = float(temperature_env)
        except ValueError:
            logger.warning("Invalid AZURE_OPENAI_TEMPERATURE value, using default")

    # Priority-based authentication
    # Priority 1: API Key
    if api_key:
        provider = AzureOpenAIProvider(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
            config=config,
        )
        logger.info("Using API key authentication for Azure OpenAI")
    # Priority 2: Azure AD Token
    elif azure_ad_token:
        provider = AzureOpenAIProvider(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            config=config,
        )
        logger.info("Using Azure AD token authentication for Azure OpenAI")
    # Priority 3: Managed Identity / Default Credential
    elif use_managed_identity or use_default_credential:
        if not AZURE_IDENTITY_AVAILABLE:
            logger.error(
                "Managed identity authentication requires 'azure-identity' package. "
                "Install with: pip install azure-identity"
            )
            return None

        try:
            if use_default_credential:
                credential = DefaultAzureCredential()
                logger.info("Using DefaultAzureCredential for Azure OpenAI")
            elif managed_identity_client_id:
                credential = ManagedIdentityCredential(client_id=managed_identity_client_id)
                logger.info(f"Using user-assigned managed identity (client_id: {managed_identity_client_id})")
            else:
                credential = ManagedIdentityCredential()
                logger.info("Using system-assigned managed identity for Azure OpenAI")

            provider = AzureOpenAIProvider(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                credential=credential,
                config=config,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure credential: {e}")
            return None
    else:
        logger.warning("No authentication method configured for Azure OpenAI provider")
        return None

    await coordinator.mount("providers", provider, name="azure-openai")
    logger.info(f"Mounted AzureOpenAIProvider (endpoint: {azure_endpoint}, version: {api_version})")

    # Return cleanup function
    async def cleanup():
        if hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


class AzureOpenAIProvider:
    """Azure OpenAI Service integration."""

    name = "azure-openai"

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str | None = None,
        azure_ad_token: str | None = None,
        api_version: str = "2024-02-15-preview",
        credential: Any = None,  # Azure credential object
        config: dict[str, Any] | None = None,
    ):
        """Initialize Azure OpenAI provider."""
        # Create client with appropriate authentication
        if api_key:
            self.client = AsyncAzureOpenAI(azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version)
            logger.debug("Using API key authentication")
        elif azure_ad_token:
            # Azure AD token authentication
            self.client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint, azure_ad_token=azure_ad_token, api_version=api_version
            )
            logger.debug("Using Azure AD token authentication")
        elif credential:
            # Managed Identity / DefaultAzureCredential authentication
            # Create a token provider function for the OpenAI client
            def get_azure_ad_token():
                """Get token from Azure credential for OpenAI scope."""
                token = credential.get_token("https://cognitiveservices.azure.com/.default")
                return token.token

            self.client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint, azure_ad_token_provider=get_azure_ad_token, api_version=api_version
            )
            logger.debug("Using Azure credential authentication")
        else:
            raise ValueError("No authentication method provided")

        self.config = config or {}

        # Azure-specific configuration
        self.deployment_mapping = self.config.get("deployment_mapping", {})
        self.default_deployment = self.config.get("default_deployment")

        # Standard configuration (same as OpenAI)
        self.default_model = self.config.get("default_model", "gpt-4")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)

        if self.deployment_mapping:
            logger.info(f"Using deployment mapping: {self.deployment_mapping}")
        if self.default_deployment:
            logger.info(f"Default deployment: {self.default_deployment}")

    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> ProviderResponse:
        """Generate completion using Azure OpenAI."""

        # Get the requested model (or use default)
        model = kwargs.get("model", self.default_model)

        # Resolve model name to Azure deployment name
        deployment = self._resolve_deployment(model)
        logger.debug(f"Resolved model '{model}' to deployment '{deployment}'")

        # Prepare request parameters
        # Note: API versions 2024-08-01-preview and later use max_completion_tokens instead of max_tokens
        max_tokens_value = kwargs.get("max_tokens", self.max_tokens)
        params = {
            "model": deployment,  # Use deployment name as model
            "messages": messages,
            "max_completion_tokens": max_tokens_value,  # New API versions use max_completion_tokens
            "temperature": kwargs.get("temperature", self.temperature),
        }

        # Add tools if provided
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = self._convert_tools(kwargs["tools"])
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add JSON mode if requested
        if kwargs.get("response_format"):
            params["response_format"] = kwargs["response_format"]

        try:
            # Call Azure OpenAI API (same API as OpenAI once deployment is resolved)
            response = await self.client.chat.completions.create(**params)

            # Parse response (same format as OpenAI)
            message = response.choices[0].message
            content = message.content or ""

            # Parse tool calls if present
            tool_calls = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            tool=tc.function.name,
                            arguments=tc.function.arguments,  # Keep as JSON string
                            id=tc.id,
                        )
                    )

            # Build content_blocks for structured content
            content_blocks = []

            # Add text content if present
            if content:
                content_blocks.append(TextContent(text=content))

            # Add tool call content blocks
            if tool_calls:
                import json

                for tc in tool_calls:
                    # Parse arguments from JSON string to dict
                    try:
                        input_dict = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                    except json.JSONDecodeError:
                        input_dict = {"raw": tc.arguments}

                    content_blocks.append(ToolCallContent(id=tc.id, name=tc.tool, input=input_dict))

            # Return standardized response
            return ProviderResponse(
                content=content,
                content_blocks=content_blocks,
                raw=response,
                usage={
                    "input": response.usage.prompt_tokens if response.usage else 0,
                    "output": response.usage.completion_tokens if response.usage else 0,
                    "total": response.usage.total_tokens if response.usage else 0,
                },
                tool_calls=tool_calls,
            )

        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise

    def parse_tool_calls(self, response: ProviderResponse) -> list[ToolCall]:
        """Parse tool calls from provider response."""
        return response.tool_calls or []

    def _resolve_deployment(self, model: str) -> str:
        """
        Resolve model name to Azure deployment name.

        Resolution order:
        1. Check deployment_mapping for exact match
        2. Use default_deployment if configured
        3. Pass through model name as-is (assume it's a deployment name)
        """
        # First: Check explicit mapping
        if self.deployment_mapping and model in self.deployment_mapping:
            deployment = self.deployment_mapping[model]
            logger.debug(f"Using mapped deployment: {model} -> {deployment}")
            return deployment

        # Second: Use default deployment if configured
        if self.default_deployment:
            logger.debug(f"Using default deployment: {self.default_deployment} (requested: {model})")
            return self.default_deployment

        # Third: Pass through model name as deployment name
        logger.debug(f"Using model name as deployment: {model}")
        return model

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Azure OpenAI format (same as OpenAI)."""
        azure_tools = []

        for tool in tools:
            # Get schema from tool if available
            input_schema = getattr(tool, "input_schema", {"type": "object", "properties": {}, "required": []})

            azure_tools.append(
                {
                    "type": "function",
                    "function": {"name": tool.name, "description": tool.description, "parameters": input_schema},
                }
            )

        return azure_tools
