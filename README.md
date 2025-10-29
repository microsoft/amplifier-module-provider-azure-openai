# Azure OpenAI Provider Module for Amplifier

A provider module that integrates Azure OpenAI Service with the Amplifier AI agent platform.

## Overview

This module enables Amplifier to use Azure OpenAI Service deployments for language model completions. It supports the same chat completion API as standard OpenAI, but routes requests through Azure-hosted deployments with Azure-specific authentication and endpoint configuration.

## Features

- **Azure OpenAI Service Integration**: Connect to your Azure-hosted OpenAI deployments
- **Deployment Name Mapping**: Map model names to Azure deployment names
- **Multiple Authentication Methods**: Support for API keys, Azure AD tokens, and Managed Identity
- **Compatible with OpenAI API**: Uses the same chat completions format
- **Tool Calling Support**: Full support for function calling/tools
- **Managed Identity Support**: Seamless authentication in Azure environments

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

Install the module using pip:

```bash
uv pip install -e amplifier-module-provider-azure-openai
```

For Managed Identity authentication support, install with the `azure` extra:

```bash
uv pip install -e amplifier-module-provider-azure-openai[azure]
```

Or add it to your Amplifier configuration for automatic installation.

## Quick Start with Environment Variables

The simplest way to configure the provider is with environment variables.

### For Local Development (Recommended)

Use DefaultAzureCredential with Azure CLI:

```bash
# Login to Azure
az login

# Set up endpoint
export AZURE_OPENAI_ENDPOINT="https://myresource.openai.azure.com"
export AZURE_USE_DEFAULT_CREDENTIAL="true"

# Optional: Configure API version and defaults
export AZURE_OPENAI_API_VERSION="2024-10-01-preview"
export AZURE_OPENAI_DEFAULT_MODEL="gpt-5"

# Run amplifier - no config file needed!
amplifier run
```

### For API Key Authentication

```bash
# Set up authentication
export AZURE_OPENAI_ENDPOINT="https://myresource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key-here"

# Optional: Configure API version and defaults
export AZURE_OPENAI_API_VERSION="2024-10-01-preview"
export AZURE_OPENAI_DEFAULT_MODEL="gpt-5"

# Run amplifier
amplifier run
```

### For Azure Deployments

For managed identity in Azure environments (VMs, App Service, etc.):

```bash
export AZURE_OPENAI_ENDPOINT="https://myresource.openai.azure.com"
export AZURE_USE_DEFAULT_CREDENTIAL="true"  # Will use managed identity when available
```

## Configuration

### Simple Configuration

Basic setup with API key authentication:

```toml
[[providers]]
name = "azure-openai"
[providers.config]
azure_endpoint = "https://myresource.openai.azure.com"
api_key = "your-api-key-here"
default_model = "gpt-5"
```

### Advanced Configuration

Full configuration with deployment mapping:

```toml
[[providers]]
name = "azure-openai"
[providers.config]
# Required: Your Azure OpenAI resource endpoint
azure_endpoint = "https://myresource.openai.azure.com"

# Authentication (use one of these, in order of priority)
api_key = "your-api-key-here"                    # Option 1: API Key
# azure_ad_token = "your-azure-ad-token"         # Option 2: Azure AD Token
# use_managed_identity = true                    # Option 3: Managed Identity
# use_default_credential = true                  # Option 4: DefaultAzureCredential

# For user-assigned managed identity (optional)
# managed_identity_client_id = "client-id-here"

# Optional: API version (defaults to 2024-02-15-preview)
api_version = "2024-10-01-preview"

# Optional: Map model names to Azure deployment names
[providers.config.deployment_mapping]
"gpt-5" = "my-gpt5-deployment"
"gpt-5" = "my-gpt5-deployment"
"gpt-5-mini" = "my-mini-deployment"

# Optional: Default deployment when no mapping matches
default_deployment = "my-default-deployment"

# Optional: Default model for requests
default_model = "gpt-5"

# Optional: Generation parameters
max_tokens = 4096
temperature = 0.7
```

## Deployment Name Mapping

Azure OpenAI uses deployment names instead of model names. This module provides flexible mapping:

### Resolution Order

1. **Explicit Mapping**: Check `deployment_mapping` for the requested model
2. **Default Deployment**: Use `default_deployment` if configured
3. **Pass-through**: Use the model name as-is (assumes it's a deployment name)

### Example Scenarios

```toml
[providers.config.deployment_mapping]
"gpt-5" = "production-gpt5"
"gpt-5-mini" = "fast-mini"

default_deployment = "fallback-deployment"
```

- Request for "gpt-5" → Uses "production-gpt5"
- Request for "gpt-5-mini" → Uses "fast-mini"
- Request for "claude-opus-4-1" → Uses "fallback-deployment" (not in mapping)
- Request for "my-custom-deploy" → Uses "my-custom-deploy" (if no default set)

## Authentication Options

The provider supports multiple authentication methods with the following priority:

1. **API Key** (highest priority)
2. **Azure AD Token**
3. **Managed Identity / Azure Credentials**

### API Key Authentication

The most common method. Set via configuration or environment variable:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://myresource.openai.azure.com"
```

```toml
[providers.config]
azure_endpoint = "https://myresource.openai.azure.com"
api_key = "your-api-key"
```

### Azure AD Token Authentication

For enterprise scenarios with Azure Active Directory:

```toml
[providers.config]
azure_endpoint = "https://myresource.openai.azure.com"
azure_ad_token = "your-azure-ad-token"
```

Or via environment:

```bash
export AZURE_OPENAI_AD_TOKEN="your-ad-token"
```

### Managed Identity Authentication

**Note:** This requires the `azure-identity` package to be installed.

#### System-Assigned Managed Identity

For Azure resources with system-assigned managed identity:

```toml
[providers.config]
azure_endpoint = "https://myresource.openai.azure.com"
use_managed_identity = true
```

#### User-Assigned Managed Identity

For Azure resources with user-assigned managed identity:

```toml
[providers.config]
azure_endpoint = "https://myresource.openai.azure.com"
use_managed_identity = true
managed_identity_client_id = "your-managed-identity-client-id"
```

#### DefaultAzureCredential (Recommended)

**This is the recommended authentication method** as it works in both local development and Azure deployments.

Uses Azure's credential chain (includes Azure CLI, environment variables, managed identity):

```toml
[provider.config]
azure_endpoint = "https://myresource.openai.azure.com"
use_default_credential = true
```

Or via environment variable:

```bash
export AZURE_USE_DEFAULT_CREDENTIAL=true
```

The `DefaultAzureCredential` tries multiple authentication methods in order:

1. Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
2. Managed identity (if running in Azure)
3. Azure CLI (if logged in locally with `az login`)
4. Azure PowerShell (if logged in)
5. Interactive browser authentication

**Note for WSL2 Users**: If you're developing in WSL2, use `DefaultAzureCredential` instead of `ManagedIdentityCredential` directly, as WSL2 doesn't have access to Azure IMDS. After running `az login`, `DefaultAzureCredential` will automatically use your CLI credentials.

## Environment Variables

The module supports these environment variables as fallbacks:

### Authentication & Connection

- `AZURE_OPENAI_ENDPOINT` or `AZURE_OPENAI_BASE_URL` - Azure OpenAI resource endpoint
- `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_KEY` - API key for authentication
- `AZURE_OPENAI_AD_TOKEN` - Azure AD token for authentication
- `AZURE_OPENAI_API_VERSION` - API version (defaults to `2024-02-15-preview`)

### Managed Identity Configuration

- `AZURE_USE_MANAGED_IDENTITY` - Enable managed identity authentication (`true`, `1`, or `yes`)
- `AZURE_USE_DEFAULT_CREDENTIAL` - Enable DefaultAzureCredential authentication (`true`, `1`, or `yes`)
- `AZURE_MANAGED_IDENTITY_CLIENT_ID` - Client ID for user-assigned managed identity

### Deployment & Model Configuration

- `AZURE_OPENAI_DEFAULT_DEPLOYMENT` - Default deployment name to use when no mapping matches
- `AZURE_OPENAI_DEFAULT_MODEL` - Default model to use for requests (defaults to `gpt-5`)

### Generation Parameters

- `AZURE_OPENAI_MAX_TOKENS` - Maximum tokens for completion (defaults to `4096`)
- `AZURE_OPENAI_TEMPERATURE` - Temperature for generation (defaults to `0.7`)

**Note**: Configuration file values take precedence over environment variables.

## Usage Example

Once configured, the Azure OpenAI provider works seamlessly with Amplifier:

```python
# In your Amplifier session
response = await session.send_message(
    "Hello, how are you?",
    provider="azure-openai",
    model="gpt-5"  # Will be mapped to your Azure deployment
)
```

## API Versions

The module defaults to API version `2024-02-15-preview`. You can override this:

```toml
[provider.config]
api_version = "2024-10-01-preview"  # Use a newer version
```

### Breaking Changes in Newer API Versions

**API versions 2024-08-01-preview and later** have introduced parameter changes:

- Use `max_completion_tokens` instead of `max_tokens`
- The provider automatically handles this for you

**Model-Specific Restrictions**: Some models (e.g., GPT-5 and later) have specific parameter requirements:

- May only support default temperature values
- Check Azure OpenAI documentation for your specific model's capabilities

Example config for newer models:

```toml
[provider.config]
api_version = "2025-03-01-preview"
default_model = "gpt-5"
temperature = 1.0  # Use model's default temperature
```

Check [Azure OpenAI documentation](https://docs.microsoft.com/azure/cognitive-services/openai/) for available API versions and model capabilities.

## Tool Calling

The provider fully supports OpenAI-style tool calling:

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get the weather in a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
]

response = await session.send_message(
    "What's the weather in Seattle?",
    provider="azure-openai",
    tools=tools
)
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**

   - Verify your API key or Azure AD token is correct
   - Check that the endpoint URL includes `https://` and ends with `.openai.azure.com`

2. **Deployment Not Found**

   - Ensure the deployment name exists in your Azure OpenAI resource
   - Check deployment mapping configuration
   - Verify the deployment is in a "Succeeded" state

3. **API Version Errors**

   - Some features may require specific API versions
   - Try using the default version or check Azure documentation

4. **Rate Limiting**
   - Azure OpenAI has deployment-specific rate limits
   - Consider implementing retry logic or using multiple deployments

### Debug Logging

Enable debug logging to see deployment resolution:

```python
import logging
logging.getLogger("amplifier_module_provider_azure_openai").setLevel(logging.DEBUG)
```

## Differences from Standard OpenAI

- **Endpoints**: Uses Azure resource endpoints instead of api.openai.com
- **Authentication**: Supports Azure-specific auth methods
- **Deployments**: References deployment names instead of model names directly
- **Rate Limits**: Azure-specific quotas per deployment
- **Regional Availability**: Limited to Azure regions with OpenAI service

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Support

For issues or questions:

- Check the [Amplifier documentation](https://github.com/amplifier-dev/amplifier)
- Review [Azure OpenAI documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)
- File issues in the Amplifier repository
