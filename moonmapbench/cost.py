"""Token usage and cost of all cached answers in results/<model>__<effort>/responses.jsonl.

Cost = tokens x list price. Thinking/reasoning tokens are billed as output tokens and are
part of the output counts of both APIs. Only models in PRICES_PER_MTOK get a price.

    python -m moonmapbench.cost --model claude-opus-5-5 --effort low
"""

import argparse

from moonmapbench.query import load_responses, run_dir

# USD per million (input, output) tokens, standard API list prices. Anthropic models only:
# tokens() does not handle OpenAI's discount for cached input tokens.
PRICES_PER_MTOK = {"claude-opus-5-5": (4.00, 20.00)}
# Anthropic usage fields that tokens() knows; any other field might change the price.
ANTHROPIC_USAGE_KEYS = {"cache_creation", "cache_creation_input_tokens", "cache_read_input_tokens", "inference_geo",
                        "input_tokens", "output_tokens", "output_tokens_details", "service_tier"}


def tokens(usage: dict) -> tuple[int, int]:
    """(input, output) tokens of one answer, from either API's usage record."""
    if "input_tokens" in usage:  # Anthropic Messages
        unknown = usage.keys() - ANTHROPIC_USAGE_KEYS
        if unknown:
            raise ValueError(f"unknown usage fields {sorted(unknown)} might change the price: {usage}")
        # Cache tokens, US-only inference and other service tiers have other prices.
        if (usage["cache_creation_input_tokens"] or usage["cache_read_input_tokens"]
                or usage["inference_geo"] != "global" or usage["service_tier"] != "standard"):
            raise ValueError(f"only uncached, global, standard-tier usage is priced: {usage}")
        return usage["input_tokens"], usage["output_tokens"]
    return usage["prompt_tokens"], usage["completion_tokens"]  # OpenAI Chat Completions


def main(args: argparse.Namespace) -> None:
    path = run_dir(args.model, args.effort) / "responses.jsonl"
    records = load_responses(path, args.model, args.effort)
    if not records:
        raise RuntimeError(f"no answers in {path}")
    counts = [tokens(rec["usage"]) for rec in records.values()]
    n = len(counts)
    input_tokens = sum(i for i, _ in counts)
    output_tokens = sum(o for _, o in counts)
    print(f"{args.model} ({args.effort}): {n} answers, {input_tokens:,} input tokens "
          f"({input_tokens / n:.0f}/answer), {output_tokens:,} output tokens ({output_tokens / n:.0f}/answer)")
    if args.model not in PRICES_PER_MTOK:
        print(f"cost: unknown, no list price for {args.model} in moonmapbench.cost.PRICES_PER_MTOK")
        return
    input_price, output_price = PRICES_PER_MTOK[args.model]
    cost = (input_tokens * input_price + output_tokens * output_price) / 1e6
    print(f"cost: ${cost:.4f} (${cost / n:.5f}/answer) at ${input_price:g}/${output_price:g} per M input/output tokens")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--effort", default="low", help="reasoning effort")
    main(p.parse_args())


if __name__ == "__main__":
    cli()
