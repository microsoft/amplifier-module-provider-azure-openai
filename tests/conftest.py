"""
Pytest configuration for module tests.

Behavioral tests use inheritance from amplifier-core base classes.
See tests/test_behavioral.py for the inherited tests.

The amplifier-core pytest plugin provides fixtures automatically:
- module_path: Detected path to this module
- module_type: Detected type (provider, tool, hook, etc.)
- provider_module, tool_module, etc.: Mounted module instances
"""

import pytest


class MockOpenAIProvider:
    """Minimal stand-in for OpenAIProvider to test Azure-specific overrides."""

    def __init__(self, *, api_key=None, config=None, coordinator=None, client=None):
        self._api_key = api_key
        self.config = config or {}
        self.coordinator = coordinator


@pytest.fixture
def mock_openai_provider_cls():
    """The MockOpenAIProvider class for use as a base_cls argument."""
    return MockOpenAIProvider
