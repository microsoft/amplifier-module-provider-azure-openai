"""Azure OpenAI pricing rates and cost computation.

Verification date: 2026-05-06
Source: https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/
Rates: PAYG (pay-as-you-go) — same as OpenAI.com rates.

Only model IDs referenced in the Azure OpenAI provider source are included.
Azure model IDs often lack date stamps.
Unknown models return None — DO NOT default to $0.00.

Usage
-----
    from amplifier_module_provider_azure_openai._cost import compute_cost
    from decimal import Decimal

    cost = compute_cost(
        "gpt-5.4",
        prompt_tokens=1_000,
        completion_tokens=200,
        cached_tokens=100,
    )
    # Returns Decimal or None if the model is not recognised.

Notes
-----
- No cache write cost for Azure OpenAI PAYG (same as OpenAI.com).
- cached_tokens subtraction happens INSIDE compute_cost to prevent call-site double-charging.
- PTU short-circuit happens at the call site (_convert_to_chat_response), not here.
"""

from __future__ import annotations

from decimal import Decimal

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_PER_M = Decimal("1_000_000")

# _RATES maps model-id → {
#   "input_per_m":       Decimal,  # fresh input tokens, per 1M
#   "output_per_m":      Decimal,  # output/completion tokens, per 1M
#   "cache_read_per_m":  Decimal,  # cached input tokens, per 1M
# }
#
# Rates are in USD.
# Unknown models → return None (DO NOT default to $0.00).
#
# Only model IDs referenced in Azure OpenAI provider source are included.
# Azure model IDs often lack date stamps.
_RATES: dict[str, dict[str, Decimal]] = {
    # ------------------------------------------------------------------
    # GPT 5.4 (Azure default)  ($2.50 / $15.00, cache_read $0.25)
    # ------------------------------------------------------------------
    "gpt-5.4": {
        "input_per_m": Decimal("2.50"),
        "output_per_m": Decimal("15.00"),
        "cache_read_per_m": Decimal("0.25"),
    },
    # ------------------------------------------------------------------
    # GPT 5.5  ($5.00 / $30.00, cache_read $0.50)
    # ------------------------------------------------------------------
    "gpt-5.5": {
        "input_per_m": Decimal("5.00"),
        "output_per_m": Decimal("30.00"),
        "cache_read_per_m": Decimal("0.50"),
    },
}


def compute_cost(
    model: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
) -> Decimal | None:
    """Compute the cost of an Azure OpenAI API call in USD.

    Args:
        model: The model ID (e.g. 'gpt-5.4').
        prompt_tokens: Total prompt tokens (TOTAL, includes cached).
            This is response.usage.prompt_tokens.
        completion_tokens: Completion tokens used.
        cached_tokens: Number of prompt tokens served from cache.
            This is response.usage.prompt_tokens_details.cached_tokens.

    Returns:
        Decimal cost in USD, or None if the model is not in the pricing table.

    Note:
        cached_tokens subtraction happens inside this function to prevent
        call-site double-charging.  Callers pass the raw API fields directly.
        PTU short-circuit happens at the call site, not here.
    """
    rates = _RATES.get(model)
    if rates is None:
        return None
    # Subtract cached from total INSIDE the function to prevent call-site double-charging.
    # Clamp to 0: if caller passes only cached_tokens without matching prompt_tokens,
    # fresh_input should not go negative.
    fresh_input = max(0, prompt_tokens - cached_tokens)
    cost = Decimal(fresh_input) * rates["input_per_m"] / _PER_M
    cost += Decimal(completion_tokens) * rates["output_per_m"] / _PER_M
    if cached_tokens:
        cost += Decimal(cached_tokens) * rates["cache_read_per_m"] / _PER_M
    return cost
