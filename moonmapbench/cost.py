"""Token usage and cost of all cached answers of one body, model and effort (their responses.jsonl).

Cost = tokens x list price, halved for answers from the Message Batches API.
Thinking/reasoning tokens are billed as output tokens. The Anthropic and OpenAI APIs count them
in the output tokens; xAI counts them apart, and tokens() adds them. Only models in
PRICES_PER_MTOK get a price, except Grok models: xAI reports the cost of each answer
(usage.cost_in_usd_ticks), and that is used instead.

    python -m moonmapbench.cost --model claude-opus-5-5 --effort low
    python -m moonmapbench.cost --body mars --model claude-opus-5-5 --effort low
"""

import argparse

from moonmapbench.bodies import BODIES
from moonmapbench.query import load_responses, provider_for, run_dir

# USD per million (input, output) tokens, standard API list prices. Anthropic models only:
# cost() reads Anthropic usage records.
PRICES_PER_MTOK = {"claude-opus-5-5": (4.00, 20.00)}
# Price relative to the list price, by the usage record's service_tier.
TIER_PRICE_FACTOR = {"standard": 1.0, "batch": 0.5}
# Anthropic usage fields that cost() knows; any other field might change the price.
ANTHROPIC_USAGE_KEYS = {"cache_creation", "cache_creation_input_tokens", "cache_read_input_tokens", "inference_geo",
                        "input_tokens", "output_tokens", "output_tokens_details", "service_tier"}
# xAI's cost_in_usd_ticks is in units of 1e-10 USD (checked: it matches xAI's list prices for grok-4.7).
XAI_USD_PER_TICK = 1e-10


def tokens(model: str, usage: dict) -> tuple[int, int]:
    """(input, output) tokens of one answer, from the usage record of the model's API."""
    provider = provider_for(model)
    if provider == "anthropic":  # Messages
        return usage["input_tokens"], usage["output_tokens"]
    if provider == "xai":  # Chat Completions, but completion_tokens leaves out the reasoning tokens
        output = usage["completion_tokens"] + usage["completion_tokens_details"]["reasoning_tokens"]
        if usage["prompt_tokens"] + output != usage["total_tokens"]:
            raise ValueError(f"xAI token counts do not add up to total_tokens: {usage}")
        return usage["prompt_tokens"], output
    return usage["prompt_tokens"], usage["completion_tokens"]  # OpenAI Chat Completions


def cost(model: str, usage: dict) -> float:
    """USD for one answer of a model in PRICES_PER_MTOK. Raises for usage it cannot price."""
    unknown = usage.keys() - ANTHROPIC_USAGE_KEYS
    if unknown:
        raise ValueError(f"unknown usage fields {sorted(unknown)} might change the price: {usage}")
    # Cache tokens, US-only inference and other service tiers have other prices.
    if (usage["cache_creation_input_tokens"] or usage["cache_read_input_tokens"]
            or usage["inference_geo"] != "global" or usage["service_tier"] not in TIER_PRICE_FACTOR):
        raise ValueError(f"only uncached, global, standard or batch usage is priced: {usage}")
    input_price, output_price = PRICES_PER_MTOK[model]
    list_price = (usage["input_tokens"] * input_price + usage["output_tokens"] * output_price) / 1e6
    return list_price * TIER_PRICE_FACTOR[usage["service_tier"]]


def estimate_batch_cost(model: str, records: dict, n: int) -> float:
    """USD for n more answers through the Batches API, at the mean tokens per answer of `records`."""
    if model not in PRICES_PER_MTOK:
        raise RuntimeError(f"no list price for {model} in moonmapbench.cost.PRICES_PER_MTOK, so no estimate")
    if not records:
        raise RuntimeError(f"no answers of {model} to estimate the tokens per answer from. "
                           "Query a few points first, e.g. scripts/run.sh --model ... --points 20")
    counts = [tokens(model, rec["usage"]) for rec in records.values()]
    mean_input = sum(i for i, _ in counts) / len(counts)
    mean_output = sum(o for _, o in counts) / len(counts)
    input_price, output_price = PRICES_PER_MTOK[model]
    return n * (mean_input * input_price + mean_output * output_price) / 1e6 * TIER_PRICE_FACTOR["batch"]


def main(args: argparse.Namespace) -> None:
    body = BODIES[args.body]
    path = run_dir(body, args.model, args.effort) / "responses.jsonl"
    records = load_responses(path, body, args.model, args.effort)
    if not records:
        raise RuntimeError(f"no answers in {path}")
    counts = [tokens(args.model, rec["usage"]) for rec in records.values()]
    n = len(counts)
    input_tokens = sum(i for i, _ in counts)
    output_tokens = sum(o for _, o in counts)
    print(f"{body.title}, {args.model} ({args.effort}): {n} answers, {input_tokens:,} input tokens "
          f"({input_tokens / n:.0f}/answer), {output_tokens:,} output tokens ({output_tokens / n:.0f}/answer)")
    if provider_for(args.model) == "xai":
        cached = sum(rec["usage"]["prompt_tokens_details"]["cached_tokens"] for rec in records.values())
        total = sum(rec["usage"]["cost_in_usd_ticks"] for rec in records.values()) * XAI_USD_PER_TICK
        print(f"  {cached:,} of the input tokens were cached")
        print(f"cost: ${total:.4f} as reported by xAI (${total / n:.5f}/answer)")
        return
    if args.model not in PRICES_PER_MTOK:
        print(f"cost: unknown, no list price for {args.model} in moonmapbench.cost.PRICES_PER_MTOK")
        return
    by_tier: dict[str, list[float]] = {}
    for rec in records.values():
        by_tier.setdefault(rec["usage"]["service_tier"], []).append(cost(args.model, rec["usage"]))
    for tier, costs in sorted(by_tier.items()):
        print(f"  {tier}: {len(costs)} answers, ${sum(costs):.4f} (${sum(costs) / len(costs):.5f}/answer)")
    input_price, output_price = PRICES_PER_MTOK[args.model]
    total = sum(sum(costs) for costs in by_tier.values())
    print(f"cost: ${total:.4f} at ${input_price:g}/${output_price:g} per M input/output tokens, "
          f"x{TIER_PRICE_FACTOR['batch']:g} for batch")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", choices=BODIES, default="moon", help="which body the answers are about")
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--effort", default="low", help="reasoning effort")
    main(p.parse_args())


if __name__ == "__main__":
    cli()
