# Azure OpenAI Provider Module for Amplifier

A provider module that integrates Azure OpenAI Service with the Amplifier AI agent platform.

## Optional Microsoft Decision-1 capability

This development candidate adds Decision-1 to the existing Azure account. Install
this provider's `decisions` extra, which resolves the lightweight
`amplifier-lib-decisions` contract from its canonical Microsoft Git repository.
It installs no model weights or inference server. It reuses the provider's API
key, token callback, DefaultAzureCredential or managed identity. It does not
create a second credential store or change the configured chat deployment.

```yaml
providers:
  - module: provider-azure-openai
    config:
      azure_endpoint: https://your-resource.services.ai.azure.com
      use_default_credential: true
      default_model: your-chat-deployment
      chat_enabled: true  # Default: mount Azure chat alongside optional decisions.
      decisions:
        enabled: true
        backends:
          desktop-selection:
            enabled: true
            model: your-decision-deployment
```

Keep `chat_enabled` true (the default) to mount the Azure chat provider alongside
its optional decision backends. Named decision models share the account and do
not replace the chat deployment.

For a resource that only has a Decision-1 deployment, use the separate
`hooks-azure-openai-decisions` hook from this same package (with the `decisions`
extra installed). It reuses the same account endpoint, key/Entra/token-file
resolution without constructing or mounting a chat provider:

```yaml
hooks:
  - module: hooks-azure-openai-decisions
    config:
      azure_endpoint: https://your-resource.services.ai.azure.com
      use_default_credential: true
      decisions:
        enabled: true
        model: your-decision-deployment
```

An app that reads the decision catalog can project a saved Azure account with
`chat_enabled: false` to this hook. The catalog's exact `when` condition leaves
ordinary Azure chat-plus-decisions on the normal provider path. Directly putting
`chat_enabled: false` under `providers` is still invalid on Core 2.0.2: Core
requires a new chat-provider instance. Use the explicit hook or a host that
supports this account projection; another chat provider does not make an invalid
Azure provider mount valid.

The hook uses the shared account lifecycle for partial-failure/cancellation
rollback and cleanup. Disabled decisions do not construct credentials or
transports. Entra sign-in/refresh remains host-owned; this hook does not provide
a browser login flow. If registration fails, do not rely on leftover backends or
replay an uncertain action. This route requires the shared hook lifecycle support
in amplifier-lib-decisions; older library snapshots without that module are not
compatible.

The service uses `/providers/microsoft/v1/systemone`, a resource API separate
from the chat SDK route. An optional decision `endpoint` overrides the account
endpoint. Portal `/models` URLs are normalized only for Foundry resource hosts;
custom gateway prefixes are preserved. Predicate, choice and score are supported;
images are not advertised.

For a host-managed short-lived token, `azure_ad_token_file` can reference a private
JSON file containing `accessToken` and Unix `expires_on`. The provider rereads it
per request and rejects expired tokens. The host owns refresh and file permissions;
this option is not a durable sign-in. Never commit that file. API-key-disabled
resources require Entra authentication and an appropriate inference role on the
resource. Discovery and preference saving do not grant access or perform sign-in.

Consumers separately bind the backend and consent to state transfer. Missing
selection remains ordinary. Failures are redacted and are not retried; consumer
fallback does not replay an uncertain desktop/browser action.

Wire reference: [Microsoft Decision in Foundry](https://learn.microsoft.com/en-us/azure/foundry/foundry-models/how-to/use-foundry-models-microsoft-decision).

## Overview

This module enables Amplifier to use Azure OpenAI Service deployments for language model reasoning via the Responses API. Requests are routed through your Azure endpoint with Azure-specific authentication and deployment configuration.

### Deployment listing

For setup and Settings consumers, the publicly exported
`AzureOpenAIProvider(config=None)` supports `get_info()`, `get_model_list_info()`
and `list_models()` without importing the OpenAI provider, resolving credentials
or constructing a client. It is a setup/catalog object, not an inference
provider. `mount()` still creates the runtime provider and owns account cleanup.
Both use the same local listing implementation and selector policy.

`list_models(*, filters=None)` returns **configured**, not remotely enumerated,
deployment names. Set `deployments: [production-chat, team-custom]` to list multiple
aliases. Without that key, explicitly supplied `default_model`, `deployment_name`
and `default_deployment` names are listed. The hardcoded inference default is
never evidence of a deployment. `deployments: []` is a successful empty configured
view; no configured names raises a non-retryable `ProviderUnavailableError` and
the host should offer exact manual entry.

Azure inference access does not establish management-plane deployment-list access.
This implementation does not request a management token, force Azure login, or
mislabel the general model catalog as deployed aliases. `get_model_list_info()`
reports `source: configured` or `unsupported` accordingly.

All configured aliases are included by default, with no `gpt-` prefix rule.
`config.model_filters` snapshots `include`, `exclude` and `hide_dated`; a per-call
mapping replaces it, `{}` is unfiltered, `include: []` includes none, and exclude
wins. Globs are case-insensitive and OS-independent; returned names are unchanged.
`hide_dated` only hides a YYYY-MM-DD suffix when its undated alias is selected.
Zero ModelInfo limits mean unknown; no capability is inferred from a deployment
name. Explicit inference selections and authentication remain unchanged.

### Session media compatibility

`get_media_request_support(*, model, request_options=None)` is available on both
the setup object and mounted runtime provider. It returns version 1 with
`positions: []`: **no Azure deployment image position is qualified on this
route yet**, including user attachments and tool-result images. The PNG/JPEG
MIME list describes the media contract, not an image-support claim.

Azure deployment aliases, even `gpt-4o` or another public model's name, do not
establish the actual deployed model or supported wire format. Configuration and
request-option metadata cannot opt into vision. This override deliberately does
not inherit the OpenAI provider's public-model support declaration.

A compatible host's session-media adapter must freeze explicit
`[Image unavailable: ...; no pixels supplied]` text at the original user/tool
position **before** budgeting/counting, preserving text order and tool-call
identity. It must not relocate pixels to a synthetic user message, change the
selected deployment, or interpret a serialized ref dictionary as pixels.
The support method alone does not prepare requests or retrofit older hosts.

Azure continues to use its pre-existing OpenAI-provider runtime dependency for
transport, counting, retries and diagnostic copies. SDK retries remain disabled
(`max_retries=0`); provider-owned retries must revalidate the exact private
projection and effective options immediately before each actual attempt. HTTP
transport is nonredirecting: a redirect must not add a physical request
outside that per-attempt witness. Authentication/token refresh remains unchanged.
Media activation is held until the host's media-v1 adapter and that base provider's
validation/diagnostic safeguards are accepted together. A support lookup or
offline text-fallback test is not proof of vision, native counting, retry guards
or pixel-free diagnostics for arbitrary legacy inputs.

The optional `tests/test_media_projection_integration.py` runs against an
explicitly selected Foundation media-v1 test installation (including its media
extra); it skips when that API is absent. Foundation is not an Azure runtime
dependency. These synthetic fixtures remove inference credentials and fence
real sockets, exercising original-position fallback through the real OpenAI SDK
with an in-memory HTTP transport. They do not perform live Azure inference.

`tests/test_media_inherited.py` additionally requires an explicitly selected
OpenAI media-base candidate. It exercises automatic validation, unavailable
native count/compaction, SSE completion, owned retry/revocation, unknown-send
fences and diagnostic-copy isolation through Azure's actual subclass. Its strict
regressions are qualification gates, not assertions that any chosen candidate
already passes them. Legacy inline-image diagnostic fixtures qualify omission
only, not real Azure image support or permission to hydrate unqualified refs.

## Features

- **Azure OpenAI Service Integration**: Connect to your Azure-hosted OpenAI deployments
- **Responses API Compatibility**: Routes requests through Azure's Responses API endpoint
- **Deployment Name Mapping**: Map model names to Azure deployment names
- **Multiple Authentication Methods**: Support for API keys, Azure AD tokens, and Managed Identity
- **Tool Calling Support**: Full support for function calling/tools
- **Managed Identity Support**: Seamless authentication in Azure environments

### Tool Calling

The provider recognises Responses API `function_call` / `tool_call`
payloads, decodes any JSON-encoded arguments, and forwards standard
`ToolCall` objects to Amplifier. No additional configuration is needed—tools
declared in your bundle configuration run as soon as the model
requests them.

### Graceful Error Recovery

This provider inherits graceful degradation from the OpenAI base provider:

- **Automatic repair** of missing tool results in conversation history
- **Visible failures** via `[SYSTEM ERROR]` messages in synthetic results
- **Session continuity** even when context bugs occur
- **Observability** via `provider:tool_sequence_repaired` events

See [OpenAI provider documentation](https://github.com/microsoft/amplifier-module-provider-openai#graceful-error-recovery) for detailed explanation of the graceful degradation pattern.

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

The Azure OpenAI provider is automatically installed when you run `amplifier init`:

```bash
amplifier init
```

Or install it explicitly:

```bash
amplifier provider install azure-openai
```

For Managed Identity authentication support, the `azure-identity` package is required. Install it separately:

```bash
uv pip install azure-identity
```

**Note**: When Amplifier runs `amplifier init`, it automatically installs all known provider modules including this one.

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
export AZURE_OPENAI_DEFAULT_MODEL="gpt-5.4"

# Start amplifier - no config file needed!
amplifier
```

### For API Key Authentication

```bash
# Set up authentication
export AZURE_OPENAI_ENDPOINT="https://myresource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key-here"

# Optional: Configure API version and defaults
export AZURE_OPENAI_API_VERSION="2024-10-01-preview"
export AZURE_OPENAI_DEFAULT_MODEL="gpt-5.4"

# Start amplifier
amplifier
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
default_model = "gpt-5.4"
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

# Optional: API version (defaults to 2024-10-01-preview)
api_version = "2024-10-01-preview"

# Optional: Map model names to Azure deployment names
[providers.config.deployment_mapping]
"gpt-5.4" = "my-gpt5-deployment"
"gpt-5.4-mini" = "my-gpt5-mini-deployment"
"gpt-5-mini" = "my-mini-deployment"

# Optional: Default deployment when no mapping matches
default_deployment = "my-default-deployment"

# Optional: Default model for requests
default_model = "gpt-5.4"

# Optional: Generation parameters
max_tokens = 4096
temperature = 0.7

# Optional: Debug configuration
debug = false      # Enable standard debug events
raw_debug = false  # Enable ultra-verbose raw API I/O logging
```

### Debug Configuration

**Standard Debug** (`debug: true`):

- Emits `llm:request:debug` and `llm:response:debug` events
- Contains request/response summaries with message counts, model info, usage stats
- Moderate log volume, suitable for development

**Raw Debug** (`debug: true, raw_debug: true`):

- Emits `llm:request:raw` and `llm:response:raw` events
- Contains complete, unmodified request params and response objects
- Extreme log volume, use only for deep provider integration debugging
- Captures the exact data sent to/from Azure OpenAI API before any processing

**Example**:

```yaml
providers:
  - module: provider-azure-openai
    config:
      debug: true # Enable debug events
      raw_debug: true # Enable raw API I/O capture
      azure_endpoint: https://myresource.openai.azure.com
      api_key: ${AZURE_OPENAI_API_KEY}
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
"gpt-5.4" = "production-gpt5"
"gpt-5-mini" = "fast-mini"

default_deployment = "fallback-deployment"
```

- Request for "gpt-5.4" → Uses "production-gpt5"
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
- `AZURE_OPENAI_API_VERSION` - API version (defaults to `2024-10-01-preview`)

### Managed Identity Configuration

- `AZURE_USE_MANAGED_IDENTITY` - Enable managed identity authentication (`true`, `1`, or `yes`)
- `AZURE_USE_DEFAULT_CREDENTIAL` - Enable DefaultAzureCredential authentication (`true`, `1`, or `yes`)
- `AZURE_MANAGED_IDENTITY_CLIENT_ID` - Client ID for user-assigned managed identity

### Deployment & Model Configuration

- `AZURE_OPENAI_DEFAULT_DEPLOYMENT` - Default deployment name to use when no mapping matches
- `AZURE_OPENAI_DEFAULT_MODEL` - Default model to use for requests (defaults to `gpt-5.4`)

### Generation Parameters

- `AZURE_OPENAI_MAX_OUTPUT_TOKENS` - Maximum output tokens (defaults to `4096`)
- `AZURE_OPENAI_TEMPERATURE` - Temperature for generation (defaults to `0.7`)

**Note**: Configuration file values take precedence over environment variables.

## Usage Example

Once configured, the Azure OpenAI provider works seamlessly with Amplifier:

```python
# In your Amplifier session
response = await session.send_message(
    "Hello, how are you?",
    provider="azure-openai",
    model="gpt-5.4"  # Will be mapped to your Azure deployment
)
```

## API Versions

The module defaults to API version `2024-10-01-preview`. You can override this:

```toml
[provider.config]
api_version = "2024-10-01-preview"  # Use a newer version
```

### Breaking Changes in Newer API Versions

**API versions 2024-08-01-preview and later** introduce parameter changes:

- Use `max_output_tokens` (Azure) / `max_completion_tokens` (OpenAI) instead of `max_tokens`
- The provider automatically handles this translation for you

**Model-Specific Restrictions**: Some models (e.g., GPT-5 and later) have specific parameter requirements:

- May only support default temperature values
- Check Azure OpenAI documentation for your specific model's capabilities

Example config for newer models:

```toml
[provider.config]
api_version = "2025-03-01-preview"
default_model = "gpt-5.4"
temperature = 1.0  # Use model's default temperature
```

Check [Azure OpenAI documentation](https://docs.microsoft.com/azure/cognitive-services/openai/) for available API versions and model capabilities.

## Tool Calling

The provider fully supports Responses API tool/function calling:

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

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

### Waiting for model work

The default `timeout` is `null`: model work waits for completion, explicit cancellation, or a provider/transport error. Set a numeric `timeout` in seconds to opt into a deadline. Existing cleanup limits are unchanged.

Azure inherits completion, streaming, and background polling wait behavior from `provider-openai`. Deploy with the corresponding OpenAI provider wait-policy update; an older parent can still impose its own deadline. Native compaction remains capability-gated and is not advertised for Azure endpoints.

### Automatic tool-result cache boundaries

With the OpenAI provider policy hook introduced in
[provider-openai #138](https://github.com/microsoft/amplifier-module-provider-openai/pull/138),
this adapter opts in only for recognized supported model names on non-PTU
deployments. `deployment_type: PTU` never receives automatic explicit boundaries.
Arbitrary deployment aliases keep implicit service behavior; an alias does not
prove its underlying model. Caller-supplied fields remain the caller's settings.
This follows [Azure's deployment-specific cache support](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/prompt-caching).
No longer retention policy or new remote cache resource is enabled.
