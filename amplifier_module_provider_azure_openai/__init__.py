"""
Azure OpenAI provider module for Amplifier.
Integrates with Azure OpenAI Service using Responses API.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderResponse
from amplifier_core import ToolCall
from amplifier_core.content_models import TextContent
from amplifier_core.content_models import ThinkingContent
from amplifier_core.content_models import ToolCallContent
from amplifier_core.message_models import ChatRequest
from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Message
from openai import AsyncOpenAI

# Try to import Azure Identity for managed identity support
try:
    from azure.identity import DefaultAzureCredential
    from azure.identity import ManagedIdentityCredential
    from azure.identity import get_bearer_token_provider

    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Azure OpenAI provider."""
    config = config or {}

    # Get Azure endpoint from config or environment
    azure_endpoint = (
        config.get("azure_endpoint")
        or os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("AZURE_OPENAI_BASE_URL")
    )

    if not azure_endpoint:
        logger.warning("No azure_endpoint found for Azure OpenAI provider")
        return

    # Azure Responses API is at /openai/v1/ path
    # Remove trailing slash if present, then add the v1 path
    azure_endpoint = azure_endpoint.rstrip("/")
    base_url = f"{azure_endpoint}/openai/v1/"

    # Get API version (not used in base_url approach, but keep for reference)
    api_version = config.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-10-01-preview"

    # Get authentication configuration
    api_key = config.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")

    # Get managed identity configuration
    use_managed_identity = config.get("use_managed_identity", False)
    if not use_managed_identity:
        env_val = os.environ.get("AZURE_USE_MANAGED_IDENTITY", "").lower()
        use_managed_identity = env_val in ("true", "1", "yes")

    use_default_credential = config.get("use_default_credential", False)
    if not use_default_credential:
        env_val = os.environ.get("AZURE_USE_DEFAULT_CREDENTIAL", "").lower()
        use_default_credential = env_val in ("true", "1", "yes")

    managed_identity_client_id = config.get("managed_identity_client_id") or os.environ.get(
        "AZURE_MANAGED_IDENTITY_CLIENT_ID"
    )

    # Priority-based authentication
    # Priority 1: API Key
    if api_key:
        provider = AzureOpenAIProvider(
            base_url=base_url,
            api_key=api_key,
            config=config,
            coordinator=coordinator,
        )
        logger.info("Using API key authentication for Azure OpenAI")
    # Priority 2: Managed Identity / Default Credential
    elif use_managed_identity or use_default_credential:
        if not AZURE_IDENTITY_AVAILABLE:
            logger.error(
                "Managed identity authentication requires 'azure-identity' package. "
                "Install with: pip install azure-identity"
            )
            return

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

            # Get token provider for OpenAI client
            sync_token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

            # Wrap sync token provider in async function for AsyncOpenAI
            async def async_token_provider():
                return sync_token_provider()

            provider = AzureOpenAIProvider(
                base_url=base_url,
                token_provider=async_token_provider,
                config=config,
                coordinator=coordinator,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure credential: {e}")
            return
    else:
        logger.warning("No authentication method configured for Azure OpenAI provider")
        return

    await coordinator.mount("providers", provider, name="azure-openai")
    logger.info(f"Mounted AzureOpenAIProvider (Responses API, endpoint: {base_url}, version: {api_version})")

    return


class AzureOpenAIProvider:
    """Azure OpenAI Service integration using Responses API (matches OpenAI provider)."""

    name = "azure-openai"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        token_provider: Any = None,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        """Initialize Azure OpenAI provider."""
        # Create client using base_url approach (for Responses API)
        if api_key:
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            logger.debug("Using API key authentication")
        elif token_provider:
            # Pass token_provider as api_key - SDK will call it for each request
            self.client = AsyncOpenAI(base_url=base_url, api_key=token_provider)
            logger.debug("Using Azure credential authentication")
        else:
            raise ValueError("No authentication method provided")

        self.config = config or {}
        self.coordinator = coordinator

        # Configuration (same as OpenAI provider)
        self.default_model = self.config.get("default_model") or self.config.get("default_deployment", "gpt-5")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", None)  # None = not sent
        self.reasoning = self.config.get("reasoning", None)  # None = not sent
        self.enable_state = self.config.get("enable_state", False)
        self.priority = self.config.get("priority", 100)
        self.debug = self.config.get("debug", False)  # Enable full request/response logging
        self.raw_debug = self.config.get("raw_debug", False)  # Enable ultra-verbose raw API I/O logging
        self.timeout = self.config.get("timeout", 300.0)  # API timeout in seconds (default 5 minutes)

        logger.debug(f"Configured with default_model: {self.default_model}")

    async def complete(self, messages: list[dict[str, Any]] | ChatRequest, **kwargs) -> ProviderResponse | ChatResponse:
        """Generate completion using Azure OpenAI Responses API."""

        # Handle ChatRequest format
        if isinstance(messages, ChatRequest):
            return await self._complete_chat_request(messages, **kwargs)

        # 1. Extract system instructions and convert messages to input
        instructions, remaining_messages = self._extract_system_instructions(messages)
        input_text = self._convert_messages_to_input(remaining_messages)

        # 2. Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "input": input_text,
        }

        # Add instructions if present
        if instructions:
            params["instructions"] = instructions

        # Add max output tokens
        if max_tokens := kwargs.get("max_tokens", self.max_tokens):
            params["max_output_tokens"] = max_tokens

        # Add temperature
        if temperature := kwargs.get("temperature", self.temperature):
            params["temperature"] = temperature

        # Add reasoning control
        if reasoning := kwargs.get("reasoning", self.reasoning):
            params["reasoning"] = {"effort": reasoning}

        # Add tools if provided
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = self._convert_tools(kwargs["tools"])

        # Add JSON schema if requested
        if json_schema := kwargs.get("json_schema"):
            params["text"] = {"format": {"type": "json_schema", "json_schema": json_schema}}

        # Handle stateful conversations if enabled
        if self.enable_state:
            params["store"] = True
            if previous_id := kwargs.get("previous_response_id"):
                params["previous_response_id"] = previous_id

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "data": {
                        "provider": "azure-openai",
                        "model": params["model"],
                        "message_count": len(remaining_messages),
                        "reasoning_enabled": params.get("reasoning") is not None,
                    }
                },
            )

            # DEBUG level: Full request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "data": {
                            "provider": "azure-openai",
                            "request": {
                                "model": params["model"],
                                "input": input_text,
                                "instructions": instructions,
                                "max_output_tokens": params.get("max_output_tokens"),
                                "temperature": params.get("temperature"),
                                "reasoning": params.get("reasoning"),
                            },
                        },
                    },
                )

        # RAW DEBUG: Complete request params sent to Azure OpenAI API (ultra-verbose)
        if self.coordinator and hasattr(self.coordinator, "hooks") and self.debug and self.raw_debug:
            await self.coordinator.hooks.emit(
                "llm:request:raw",
                {
                    "lvl": "DEBUG",
                    "data": {
                        "provider": "azure-openai",
                        "params": params,  # Complete params dict as-is
                    },
                },
            )

        start_time = time.time()
        try:
            # 3. Call Responses API with timeout
            try:
                response = await asyncio.wait_for(self.client.responses.create(**params), timeout=self.timeout)

                # RAW DEBUG: Complete raw response from Azure OpenAI API (ultra-verbose)
                if self.coordinator and hasattr(self.coordinator, "hooks") and self.debug and self.raw_debug:
                    await self.coordinator.hooks.emit(
                        "llm:response:raw",
                        {
                            "lvl": "DEBUG",
                            "data": {
                                "provider": "azure-openai",
                                "response": response,  # Complete response object as-is
                            },
                        },
                    )

                elapsed_ms = int((time.time() - start_time) * 1000)
            except TimeoutError:
                logger.error("Azure OpenAI Responses API timed out after 30s")
                if self.coordinator and hasattr(self.coordinator, "hooks"):
                    await self.coordinator.hooks.emit(
                        "llm:response",
                        {
                            "provider": "azure-openai",
                            "model": params["model"],
                            "status": "error",
                            "duration_ms": int((time.time() - start_time) * 1000),
                            "error": f"Timeout after {self.timeout} seconds",
                        },
                    )
                raise TimeoutError(f"Azure OpenAI API request timed out after {self.timeout} seconds")

            # 4. Parse response output (same as OpenAI)
            content, tool_calls, content_blocks = self._parse_response_output(response.output)

            # Check for reasoning/thinking blocks
            has_reasoning = False
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                for block in content_blocks or []:
                    if isinstance(block, ThinkingContent):
                        has_reasoning = True
                        await self.coordinator.hooks.emit("thinking:final", {"text": block.text})

            # Emit llm:response success event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "data": {
                            "provider": "azure-openai",
                            "model": params["model"],
                            "usage": {
                                "input": getattr(response.usage, "input_tokens", 0)
                                if hasattr(response, "usage")
                                else 0,
                                "output": getattr(response.usage, "output_tokens", 0)
                                if hasattr(response, "usage")
                                else 0,
                            },
                            "has_reasoning": has_reasoning,
                        },
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                # DEBUG level: Full response (if debug enabled)
                if self.debug:
                    content_preview = content[:500] + "..." if len(content) > 500 else content
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "data": {
                                "provider": "azure-openai",
                                "response": {
                                    "content_preview": content_preview,
                                    "tool_calls": [{"tool": tc.tool, "id": tc.id} for tc in tool_calls]
                                    if tool_calls
                                    else [],
                                },
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # 5. Return standardized response
            return ProviderResponse(
                content=content,
                raw=response,
                usage={
                    "input": getattr(response.usage, "input_tokens", 0) if hasattr(response, "usage") else 0,
                    "output": getattr(response.usage, "output_tokens", 0) if hasattr(response, "usage") else 0,
                    "total": getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") else 0,
                },
                tool_calls=tool_calls if tool_calls else None,
                content_blocks=content_blocks if content_blocks else None,
            )

        except Exception as e:
            logger.error(f"Azure OpenAI Responses API error: {e}")

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "azure-openai",
                        "model": params.get("model", self.default_model),
                        "status": "error",
                        "duration_ms": int((time.time() - start_time) * 1000),
                        "error": str(e),
                    },
                )

            raise

    async def _complete_chat_request(self, request: ChatRequest, **kwargs) -> ChatResponse:
        """Handle ChatRequest format with developer message conversion.

        Args:
            request: ChatRequest with messages
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content blocks
        """
        logger.info(f"[PROVIDER] Received ChatRequest with {len(request.messages)} messages")

        # Separate messages by role
        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [m for m in request.messages if m.role in ("user", "assistant", "tool")]

        # Combine system messages as instructions
        instructions = (
            "\n\n".join(m.content if isinstance(m.content, str) else "" for m in system_msgs) if system_msgs else None
        )

        # Convert developer messages to XML-wrapped user messages
        developer_text = ""
        if developer_msgs:
            for dev_msg in developer_msgs:
                content = dev_msg.content if isinstance(dev_msg.content, str) else ""
                developer_text += f"<context_file>\n{content}\n</context_file>\n\n"

        # Convert conversation messages to input text format
        conversation_dicts = [m.model_dump() for m in conversation]
        conversation_text = self._convert_messages_to_input(conversation_dicts)

        # Combine: developer context THEN conversation
        if developer_text:
            input_text = f"USER: {developer_text}\n\n{conversation_text}"
        else:
            input_text = conversation_text

        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "input": input_text,
        }

        if instructions:
            params["instructions"] = instructions

        if max_tokens := (request.max_output_tokens or kwargs.get("max_tokens", self.max_tokens)):
            params["max_output_tokens"] = max_tokens

        if temperature := (request.temperature or kwargs.get("temperature", self.temperature)):
            params["temperature"] = temperature

        # Add tools if provided
        if request.tools:
            params["tools"] = self._convert_tools_from_request(request.tools)

        logger.info(
            f"[PROVIDER] Azure OpenAI API call - model: {params['model']}, has_instructions: {bool(instructions)}"
        )

        # Emit llm:request event
        if self.coordinator and hasattr(self.coordinator, "hooks"):
            # INFO level: Summary only
            await self.coordinator.hooks.emit(
                "llm:request",
                {
                    "data": {
                        "provider": "azure-openai",
                        "model": params["model"],
                        "has_instructions": bool(instructions),
                        "has_developer_context": bool(developer_text),
                    }
                },
            )

            # DEBUG level: Full request payload (if debug enabled)
            if self.debug:
                await self.coordinator.hooks.emit(
                    "llm:request:debug",
                    {
                        "lvl": "DEBUG",
                        "data": {
                            "provider": "azure-openai",
                            "request": {
                                "model": params["model"],
                                "input": input_text,
                                "instructions": instructions,
                                "max_output_tokens": params.get("max_output_tokens"),
                                "temperature": params.get("temperature"),
                            },
                        },
                    },
                )

        start_time = time.time()

        # Call Azure OpenAI API
        try:
            response = await self.client.responses.create(**params)
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info("[PROVIDER] Received response from Azure OpenAI API")

            # Emit llm:response event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                # INFO level: Summary only
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "data": {
                            "provider": "azure-openai",
                            "model": params["model"],
                            "usage": {
                                "input": getattr(response.usage, "input_tokens", 0)
                                if hasattr(response, "usage")
                                else 0,
                                "output": getattr(response.usage, "output_tokens", 0)
                                if hasattr(response, "usage")
                                else 0,
                            },
                        },
                        "status": "ok",
                        "duration_ms": elapsed_ms,
                    },
                )

                # DEBUG level: Full response (if debug enabled)
                if self.debug:
                    content_preview = str(response.output)[:500] if response.output else ""
                    await self.coordinator.hooks.emit(
                        "llm:response:debug",
                        {
                            "lvl": "DEBUG",
                            "data": {
                                "provider": "azure-openai",
                                "response": {
                                    "content_preview": content_preview,
                                },
                            },
                            "status": "ok",
                            "duration_ms": elapsed_ms,
                        },
                    )

            # Convert to ChatResponse
            return self._convert_to_chat_response(response)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[PROVIDER] Azure OpenAI API error: {e}")

            # Emit error event
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "azure-openai",
                        "model": params["model"],
                        "status": "error",
                        "duration_ms": elapsed_ms,
                        "error": str(e),
                    },
                )
            raise

    def parse_tool_calls(self, response: ProviderResponse) -> list[ToolCall]:
        """Parse tool calls from provider response."""
        return response.tool_calls or []

    def _extract_system_instructions(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system messages as instructions."""
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]

        instructions = None
        if system_messages:
            instructions = "\n\n".join([m.get("content", "") for m in system_messages])

        return instructions, other_messages

    def _convert_messages_to_input(self, messages: list[dict[str, Any]]) -> str:
        """Convert message array to single input string."""
        formatted = []

        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")

            # Handle tool messages
            if role == "TOOL":
                tool_id = msg.get("tool_call_id", "unknown")
                formatted.append(f"TOOL RESULT [{tool_id}]: {content}")
            elif role == "ASSISTANT" and msg.get("tool_calls"):
                tool_call_desc = ", ".join([tc.get("tool", "") for tc in msg["tool_calls"]])
                if content:
                    formatted.append(f"{role}: {content} [Called tools: {tool_call_desc}]")
                else:
                    formatted.append(f"{role}: [Called tools: {tool_call_desc}]")
            else:
                formatted.append(f"{role}: {content}")

        return "\n\n".join(formatted)

    def _parse_response_output(self, output: list[Any]) -> tuple[str, list[ToolCall], list[Any]]:
        """Parse output blocks into content, tool calls, and content_blocks."""
        content_parts = []
        tool_calls = []
        content_blocks = []

        for block in output:
            # Handle both SDK objects and dictionaries
            if hasattr(block, "type"):
                block_type = block.type

                if block_type == "message":
                    block_content = getattr(block, "content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if hasattr(content_item, "type") and content_item.type == "output_text":
                                text = getattr(content_item, "text", "")
                                content_parts.append(text)
                                content_blocks.append(TextContent(text=text, raw=content_item))
                            elif hasattr(content_item, "get") and content_item.get("type") == "output_text":
                                text = content_item.get("text", "")
                                content_parts.append(text)
                                content_blocks.append(TextContent(text=text, raw=content_item))
                    elif isinstance(block_content, str):
                        content_parts.append(block_content)
                        content_blocks.append(TextContent(text=block_content, raw=block))

                elif block_type == "reasoning":
                    reasoning_text = getattr(block, "text", "")
                    if reasoning_text:
                        content_blocks.append(ThinkingContent(text=reasoning_text, raw=block))

                elif block_type in {"function_call", "tool_call"}:
                    arguments = getattr(block, "input", None)
                    if arguments is None and hasattr(block, "arguments"):
                        arguments = block.arguments
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            logger.debug("Failed to decode tool call arguments: %s", arguments)
                    if arguments is None:
                        arguments = {}
                    # Ensure arguments is dict for type safety
                    if not isinstance(arguments, dict):
                        arguments = {}

                    call_id = getattr(block, "call_id", "") or getattr(block, "id", "")
                    tool_name = getattr(block, "name", "")
                    tool_calls.append(
                        ToolCall(
                            tool=tool_name,
                            arguments=arguments,
                            id=call_id,
                        )
                    )
                    content_blocks.append(
                        ToolCallContent(
                            id=call_id,
                            name=tool_name,
                            arguments=arguments,
                            raw=block,
                        )
                    )
            else:
                # Dictionary format
                block_type = block.get("type")

                if block_type == "message":
                    block_content = block.get("content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if content_item.get("type") == "output_text":
                                text = content_item.get("text", "")
                                content_parts.append(text)
                                content_blocks.append(TextContent(text=text, raw=content_item))
                    elif isinstance(block_content, str):
                        content_parts.append(block_content)
                        content_blocks.append(TextContent(text=block_content, raw=block))

                elif block_type == "reasoning":
                    reasoning_text = block.get("text", "")
                    if reasoning_text:
                        content_blocks.append(ThinkingContent(text=reasoning_text, raw=block))

                elif block_type in {"function_call", "tool_call"}:
                    arguments = block.get("input")
                    if arguments is None:
                        arguments = block.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            logger.debug("Failed to decode tool call arguments: %s", arguments)
                    if arguments is None:
                        arguments = {}
                    # Ensure arguments is dict for type safety
                    if not isinstance(arguments, dict):
                        arguments = {}

                    call_id = block.get("call_id", "") or block.get("id", "")
                    tool_name = block.get("name", "")
                    tool_calls.append(
                        ToolCall(
                            tool=tool_name,
                            arguments=arguments,
                            id=call_id,
                        )
                    )
                    content_blocks.append(
                        ToolCallContent(
                            id=call_id,
                            name=tool_name,
                            arguments=arguments,
                            raw=block,
                        )
                    )

        content = "\n\n".join(content_parts) if content_parts else ""
        return content, tool_calls, content_blocks

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Responses API format."""
        responses_tools = []

        for tool in tools:
            input_schema = getattr(tool, "input_schema", {"type": "object", "properties": {}, "required": []})

            responses_tools.append(
                {"type": "function", "name": tool.name, "description": tool.description, "parameters": input_schema}
            )

        return responses_tools

    def _convert_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        """Convert ToolSpec objects from ChatRequest to Responses API format.

        Args:
            tools: List of ToolSpec objects

        Returns:
            List of Responses API-formatted tool definitions
        """
        responses_tools = []
        for tool in tools:
            responses_tools.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters,
                }
            )
        return responses_tools

    def _convert_to_chat_response(self, response: Any) -> ChatResponse:
        """Convert Azure OpenAI response to ChatResponse format.

        Args:
            response: Azure OpenAI API response

        Returns:
            ChatResponse with content blocks
        """
        from amplifier_core.message_models import TextBlock
        from amplifier_core.message_models import ThinkingBlock
        from amplifier_core.message_models import ToolCall
        from amplifier_core.message_models import ToolCallBlock
        from amplifier_core.message_models import Usage

        content_blocks = []
        tool_calls = []

        for block in response.output:
            if hasattr(block, "type"):
                block_type = block.type

                if block_type == "message":
                    block_content = getattr(block, "content", [])
                    if isinstance(block_content, list):
                        for content_item in block_content:
                            if hasattr(content_item, "type") and content_item.type == "output_text":
                                text = getattr(content_item, "text", "")
                                content_blocks.append(TextBlock(text=text))
                    elif isinstance(block_content, str):
                        content_blocks.append(TextBlock(text=block_content))

                elif block_type == "reasoning":
                    reasoning_text = getattr(block, "text", "")
                    content_blocks.append(ThinkingBlock(thinking=reasoning_text, signature=None))

                elif block_type in {"function_call", "tool_call"}:
                    call_id = getattr(block, "call_id", "") or getattr(block, "id", "")
                    name = getattr(block, "name", "")
                    input_data = getattr(block, "input", None)
                    if input_data is None and hasattr(block, "arguments"):
                        input_data = block.arguments
                    if isinstance(input_data, str):
                        try:
                            input_data = json.loads(input_data)
                        except json.JSONDecodeError:
                            logger.debug("Failed to decode tool call arguments: %s", input_data)
                    if input_data is None:
                        input_data = {}
                    # Ensure input_data is dict for type safety
                    if not isinstance(input_data, dict):
                        input_data = {}
                    content_blocks.append(ToolCallBlock(id=call_id, name=name, input=input_data))
                    tool_calls.append(ToolCall(id=call_id, name=name, arguments=input_data))

        usage = Usage(
            input_tokens=getattr(response.usage, "input_tokens", 0) if hasattr(response, "usage") else 0,
            output_tokens=getattr(response.usage, "output_tokens", 0) if hasattr(response, "usage") else 0,
            total_tokens=getattr(response.usage, "total_tokens", 0) if hasattr(response, "usage") else 0,
        )

        return ChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=None,  # Azure OpenAI Responses API doesn't provide finish_reason
        )
