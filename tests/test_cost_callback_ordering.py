"""Tests for cost-callback ordering in Azure's _convert_to_chat_response override.

Regression coverage for amplifier-support#229.

The M2 cost-stamping change in provider-openai (commit 74ba9b6, "feat(m2):
provider-openai cost stamping (#32)") added `self._add_cost(cost)` inside the
parent's `_convert_to_chat_response`. Because Azure subclasses OpenAIProvider
and overrides `_convert_to_chat_response` to apply PTU short-circuit and
Azure-specific rates, the parent's `_add_cost` would fire with OpenAI-rate
cost BEFORE Azure's override corrects it. To keep the contributor accumulator
in sync with the final `llm:response` event payload, the Azure subclass:

  1. Hands the parent a no-op `add_cost=lambda _c: None` in `__init__`
  2. Stores the real callback as `self._azure_add_cost`
  3. Calls `self._azure_add_cost(cost)` from `_convert_to_chat_response` AFTER
     computing the Azure-corrected cost

These tests pin that contract so a future change to the parent's M2 wiring
cannot silently re-break Azure cost reporting.
"""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from amplifier_core.message_models import ChatResponse
from amplifier_core.message_models import Usage

from amplifier_module_provider_azure_openai import _create_azure_provider


def _make_response(*, model: str = "gpt-5.4", prompt_tokens: int = 1_000_000):
    """A minimal Responses-API shape that BOTH parent and Azure overrides can parse.

    Parent (`OpenAIProvider._convert_to_chat_response`) reads:
      response.output (iterable of blocks)
      response.usage.{input_tokens, output_tokens, input_tokens_details,
                      output_tokens_details, prompt_tokens, completion_tokens}
      response.model

    Azure's override reads:
      response.usage.{prompt_tokens, completion_tokens, prompt_tokens_details,
                      input_tokens_details}
      response.model

    `output=[]` short-circuits the parent's content-block parsing loop without
    affecting cost flow. Both `prompt_tokens`/`completion_tokens` and
    `input_tokens`/`output_tokens` are populated so each layer reads what it expects.
    """
    return SimpleNamespace(
        model=model,
        output=[],
        usage=SimpleNamespace(
            input_tokens=prompt_tokens,
            output_tokens=0,
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            prompt_tokens_details=None,
            input_tokens_details=None,
        ),
    )


def _make_parent_chat_response(parent_cost_usd: Decimal | None = Decimal("99.99")):
    """A ChatResponse stub representing what the PARENT _convert_to_chat_response returns.

    `parent_cost_usd` is deliberately wrong (OpenAI-rate placeholder). The Azure
    override is supposed to ignore it for accumulation and substitute its own
    Azure-rate cost.

    `cost_usd` is stored on Usage via the pydantic `extra="allow"` mechanism that
    the production code uses (`usage.model_copy(update={"cost_usd": cost})`).
    """
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=0,
        total_tokens=1_000_000,
    ).model_copy(update={"cost_usd": parent_cost_usd})
    return ChatResponse(
        content=[],
        usage=usage,
        finish_reason="stop",
    )


def test_parent_add_cost_is_a_no_op():
    """Parent's self._add_cost must be the no-op lambda Azure passes in __init__.

    If a future refactor accidentally forwards the real callback to the parent,
    cost would be double-counted (parent fires with OpenAI cost, then Azure
    fires with Azure cost — both into the same _totals dict).
    """
    from amplifier_module_provider_openai import OpenAIProvider

    captured: list = []

    provider = _create_azure_provider(
        OpenAIProvider,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config={"use_streaming": False},
        add_cost=captured.append,
    )

    # Parent's self._add_cost is the inherited attribute. It must NOT route to
    # the real callback (captured.append). If this assertion fails, the parent
    # is wired to the real callback and Azure's override would double-count.
    provider._add_cost(Decimal("99.99"))
    assert captured == [], (
        "Parent's self._add_cost must be the no-op; the real callback is "
        "_azure_add_cost, called from _convert_to_chat_response after "
        "the Azure cost is computed."
    )


def test_convert_to_chat_response_silences_parent_and_uses_azure_cost():
    """End-to-end regression guard for the M2 cost-ordering conflict.

    Drives the FULL conversion path — both the parent's body (which calls
    `self._add_cost(cost)` with the OpenAI-rate cost at the M2 stamping site)
    AND Azure's override (which calls `_azure_add_cost(cost)` with the
    Azure-corrected cost). Does NOT mock out `super()._convert_to_chat_response`,
    because doing so would skip the very line that causes the regression.

    Setup:
      - Patches `compute_cost` at BOTH module sites to force known, distinct
        rates: parent computes Decimal("99.99"), Azure computes Decimal("2.50").
      - The Azure subclass passes a no-op `lambda _c: None` to the parent's
        constructor; the real user callback (captured.append) is stored as
        `self._azure_add_cost` and invoked only from Azure's override.

    Verifies:
      1. Parent's compute_cost actually ran — the parent's M2 code path WAS
         executed, so the regression-causing `self._add_cost(99.99)` line fired.
      2. The user's accumulator received exactly [Decimal("2.50")] — proving the
         no-op silenced the parent's call and only Azure's value got through.
      3. The returned chat_response carries the Azure-corrected cost on usage.

    Failure modes this test catches:
      - If the no-op is removed (parent gets the real callback), `captured` will
        contain BOTH values: [Decimal("99.99"), Decimal("2.50")]. Assertion 2 fails.
      - If Azure's override stops calling `_azure_add_cost`, `captured` will be []
        and assertion 2 fails.
      - If Azure's compute_cost isn't called (override bypassed), the cost_usd
        on the result wouldn't be the Azure value and assertion 3 fails.
    """
    from amplifier_module_provider_openai import OpenAIProvider

    captured: list = []

    provider = _create_azure_provider(
        OpenAIProvider,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config={"use_streaming": False},
        add_cost=captured.append,
    )

    response = _make_response(model="gpt-5.4", prompt_tokens=1_000_000)

    # Force known, distinct costs at each compute_cost site. The parent module's
    # name resolves to the imported function inside provider-openai; Azure's
    # name resolves to its own imported function. Patching the module-level
    # name overrides what the call site inside each method resolves to.
    with (
        patch(
            "amplifier_module_provider_openai.compute_cost",
            return_value=Decimal("99.99"),
        ) as parent_compute_cost,
        patch(
            "amplifier_module_provider_azure_openai.compute_cost",
            return_value=Decimal("2.50"),
        ),
    ):
        result = provider._convert_to_chat_response(response)

    # 1. Prove the parent's body actually ran — if compute_cost was never called
    #    in the parent, the M2 self._add_cost line wasn't reached either, and the
    #    test isn't exercising the regression it claims to guard.
    assert parent_compute_cost.called, (
        "Parent's compute_cost was never invoked — the parent's "
        "_convert_to_chat_response body did not execute. This test cannot guard "
        "the M2 ordering regression without the parent body running. Check the "
        "response shape and patching."
    )

    # 2. The user's accumulator must contain ONLY the Azure-corrected value.
    #    If the no-op lambda were removed, this list would contain BOTH the
    #    parent's 99.99 AND Azure's 2.50 — double-counting the regression.
    assert captured == [Decimal("2.50")], (
        f"Expected user accumulator == [Decimal('2.50')] (Azure value only); "
        f"got {captured!r}. If captured contains the parent's OpenAI-rate "
        f"(99.99), the no-op lambda is missing: the Azure subclass is "
        f"forwarding the real callback to the parent's self._add_cost, causing "
        f"double-accumulation."
    )

    # 3. The returned chat_response carries the Azure-corrected cost on usage,
    #    not the parent's OpenAI value.
    assert result.usage.cost_usd == Decimal("2.50"), (
        f"Expected chat_response.usage.cost_usd == Decimal('2.50') (Azure rate); "
        f"got {result.usage.cost_usd!r}. Azure's override didn't replace the "
        f"parent's cost_usd."
    )


def test_ptu_short_circuit_does_not_accumulate():
    """PTU deployments produce cost=None; _azure_add_cost must be a no-op for None.

    Azure-corrected cost is None for PTU deployments (no marginal per-token cost).
    The mount-level _add_cost callback already handles None as a no-op, so PTU
    deployments contribute nothing to _totals. This test pins that contract.
    """
    from amplifier_module_provider_openai import OpenAIProvider

    captured: list = []

    provider = _create_azure_provider(
        OpenAIProvider,
        base_url="https://example.openai.azure.com/openai/v1/",
        api_key="test-key",
        config={"use_streaming": False, "deployment_type": "PTU"},
        add_cost=captured.append,
    )

    fake_parent_response = _make_parent_chat_response(parent_cost_usd=Decimal("99.99"))
    response = _make_response(model="gpt-5.4", prompt_tokens=1_000_000)

    with patch.object(
        OpenAIProvider,
        "_convert_to_chat_response",
        return_value=fake_parent_response,
    ):
        result = provider._convert_to_chat_response(response)

    # PTU: _azure_add_cost(None) is invoked but accumulates nothing
    assert captured == [None], (
        f"Expected _azure_add_cost called once with None (PTU short-circuit); "
        f"got {captured!r}."
    )
    # Returned cost_usd should be None for PTU
    assert result.usage.cost_usd is None, (
        f"Expected PTU chat_response.usage.cost_usd to be None; "
        f"got {result.usage.cost_usd!r}."
    )
