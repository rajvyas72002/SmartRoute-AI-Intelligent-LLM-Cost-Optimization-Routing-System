from core.schemas import ModelSpec


def cost_usd(model: ModelSpec, in_tokens: int, out_tokens: int, local_cost_per_mtok: float = 0.0) -> float:
    """cost = in*price_in + out*price_out (per million tokens). Local models use the configured ESTIMATE."""
    if model.kind == "local":
        return (in_tokens + out_tokens) * local_cost_per_mtok / 1_000_000
    return (in_tokens * model.input_price_per_mtok + out_tokens * model.output_price_per_mtok) / 1_000_000


def latency_s(model: ModelSpec, out_tokens: int) -> float:
    return model.ttft_s + out_tokens / model.tokens_per_sec
