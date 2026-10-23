"""USD per 1M tokens. Approximate and illustrative: check current published pricing before
trusting the cost numbers."""

COMPLETION_PRICE_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
DEFAULT_COMPLETION_PRICE = (0.50, 1.50)

EMBEDDING_PRICE_PER_MILLION: dict[str, float] = {
    "text-embedding-3-large": 0.13,
    "text-embedding-3-small": 0.02,
}
DEFAULT_EMBEDDING_PRICE = 0.13


def estimate_completion_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_rate, completion_rate = COMPLETION_PRICE_PER_MILLION.get(model, DEFAULT_COMPLETION_PRICE)
    return (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000


def estimate_embedding_cost_usd(model: str, tokens: int) -> float:
    if model.startswith("mock"):
        return 0.0
    return tokens * EMBEDDING_PRICE_PER_MILLION.get(model, DEFAULT_EMBEDDING_PRICE) / 1_000_000
