"""SDK import/transport floor, independent of optional inference-provider imports.

Also run this file with the exact minimum SDK wheel first on PYTHONPATH. That
checks the real floor exports without installing over the user's environment.
"""

import os
from pathlib import Path
import subprocess
import sys
import tomllib


def test_declared_runtime_and_development_sdk_floors_match():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    runtime = [item for item in project["project"]["dependencies"] if item.startswith("openai")]
    development = [item for item in project["dependency-groups"]["dev"] if item.startswith("openai")]
    assert runtime == development == ["openai>=2.8.1"]


def test_sdk_setup_import_and_nonredirecting_factory_without_optional_dependencies():
    source = '''
import asyncio, importlib.abc, socket, sys

def forbidden(*args, **kwargs):
    raise AssertionError("SDK floor fixture attempted a socket")
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("amplifier_module_provider_openai", "azure.identity")):
            raise AssertionError("Optional inference dependency imported during setup: " + fullname)
sys.meta_path.insert(0, BlockOptional())
import openai
from amplifier_module_provider_azure_openai import AzureOpenAIProvider, _create_azure_provider

assert callable(openai.DefaultAsyncHttpxClient)
setup = AzureOpenAIProvider({"deployments": ["fixture-deployment"]})
assert setup.get_info().id == "azure-openai"
assert [model.id for model in asyncio.run(setup.list_models())] == ["fixture-deployment"]
assert setup.get_media_request_support(model="fixture-deployment")["positions"] == []

class Base:
    def __init__(self, *, api_key=None, config=None, **kwargs):
        self._api_key, self.config = api_key, config or {}

provider = _create_azure_provider(Base, base_url="https://azure.invalid/openai/v1/", api_key="fixture-only")
client = provider.client
try:
    assert client.max_retries == 0
    assert client._client.follow_redirects is False
finally:
    asyncio.run(provider.close())
assert client.is_closed()
print("sdk", openai.__version__, "setup import and nonredirecting factory PASS")
'''
    env = {name: value for name, value in os.environ.items() if not name.startswith(
        ("AZURE_", "OPENAI_", "ANTHROPIC_", "GOOGLE_", "GEMINI_")
    )}
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, env=env, timeout=20)
    assert result.returncode == 0, result.stderr
