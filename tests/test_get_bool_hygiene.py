"""Fail-before/pass-after regression test for the _get_bool string-coercion bug.

`_get_bool(config, key, env_value)` used to do `bool(config[key])` when the
key was present in config -- `bool("false")` is `True` in Python, so every
wizard-configured `use_managed_identity: "false"` was silently treated as
enabled (a live bug on the managed-identity auth path).
"""

from amplifier_module_provider_azure_openai import _get_bool


def test_string_false_is_false():
    assert _get_bool({"use_managed_identity": "false"}, "use_managed_identity", None) is False


def test_string_true_is_true():
    assert _get_bool({"use_managed_identity": "true"}, "use_managed_identity", None) is True


def test_real_bool_passthrough():
    assert _get_bool({"use_managed_identity": False}, "use_managed_identity", None) is False
    assert _get_bool({"use_managed_identity": True}, "use_managed_identity", None) is True


def test_falls_back_to_env_when_key_absent():
    assert _get_bool({}, "use_managed_identity", "true") is True
    assert _get_bool({}, "use_managed_identity", "false") is False


def test_falls_back_to_false_when_key_and_env_absent():
    assert _get_bool({}, "use_managed_identity", None) is False
