"""Query a model for the elevation at every pixel centre of a grid.

Answers are appended to results/<model>__<effort>/responses.jsonl, one JSON
record per (lat, lon, sample). Points already present are skipped, so an
interrupted run can be restarted and a finer grid reuses any coinciding points.
An answer that is not a plain number stops the run before it is saved; rerunning
asks that point again. For Claude models, moonmapbench.batch sends the same requests through
the Message Batches API at half price; this module refuses to run while a batch is not collected.

Claude models (claude-*) go to the Anthropic Messages API with adaptive thinking and
output_config.effort; all other models go to the OpenAI Chat Completions API with
reasoning_effort. The API key (ANTHROPIC_API_KEY or OPENAI_API_KEY) comes from the
environment or from .env in the repository root; the environment wins.

The API rate-limits an account. The number of requests in flight adapts
(AIMD): it starts at --concurrency, drops by 20% when the API answers HTTP 429
or 529 (Anthropic: overloaded), at most once per 30 s,, and slowly grows back up to --concurrency.
Running out of quota stops the run.

    python -m moonmapbench.query --res 22.5 --samples 1
"""

import argparse
import asyncio
import fcntl
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv

from moonmapbench import REPO_ROOT, RESULTS_DIR
from moonmapbench.batchlog import BATCH_LOG, read_batch_log
from moonmapbench.grid import grid_points
from moonmapbench.prompt import build_prompt, parse_elevation

MAX_ATTEMPTS = 8  # with the backoff below, ~6 min of failures before giving up on a point
BASE_DELAY_S = 5
MAX_DELAY_S = 120
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}  # 529: Anthropic is overloaded
ANTHROPIC_MAX_TOKENS = 16000  # thinking + answer; a longer reply stops the run
API_KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


class AdaptiveLimit:
    """Concurrency limit that backs off on rate limiting (AIMD).

    +2 per `limit` successes (about +2 per round of requests), x0.8 on a rate-limit error.
    Requests already in flight when the backend starts limiting fail together, so at most
    one cut per `cooldown_s` (about one request latency).
    """

    def __init__(self, maximum: int, cooldown_s: float = 30.0):
        self.maximum = maximum
        self.limit = float(maximum)
        self.cooldown_s = cooldown_s
        self.active = 0
        self._last_cut = -float("inf")
        self._cond = asyncio.Condition()

    async def __aenter__(self) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: self.active < int(self.limit))
            self.active += 1

    async def __aexit__(self, *exc) -> None:
        async with self._cond:
            self.active -= 1
            self._cond.notify_all()

    def success(self) -> None:
        self.limit = min(self.maximum, self.limit + 2 / self.limit)

    def rate_limited(self) -> None:
        now = time.monotonic()
        if now - self._last_cut >= self.cooldown_s:
            self.limit = max(1.0, self.limit * 0.8)
            self._last_cut = now


class BadAnswer(RuntimeError):
    """The API call worked, but the answer cannot be used (wrong model, refusal, truncated, ...)."""


def provider_for(model: str) -> str:
    return "anthropic" if model.startswith("claude-") else "openai"


async def create_openai(client: openai.AsyncOpenAI, model: str, effort: str, prompt: str):
    return await client.chat.completions.create(
        model=model, reasoning_effort=effort, messages=[{"role": "user", "content": prompt}]
    )


def check_served_model(served: str, model: str) -> None:
    """Raises unless `served` is `model`, or `model` with a date suffix (OpenAI names snapshots that way)."""
    if served != model and not served.startswith(f"{model}-20"):
        raise BadAnswer(f"asked for {model}, but {served} answered")


def answer_openai(completion, model: str, lat: float, lon: float) -> tuple[str, dict]:
    """(answer text, usage). Raises unless the model finished normally with a non-empty answer."""
    check_served_model(completion.model, model)
    if len(completion.choices) != 1:
        raise BadAnswer(f"expected 1 choice, got {len(completion.choices)}: {completion}")
    choice = completion.choices[0]
    content = choice.message.content
    if (choice.finish_reason != "stop" or not isinstance(content, str) or not content.strip()
            or completion.usage is None):
        raise BadAnswer(f"bad completion for ({lat}, {lon}): {completion}")
    return content, completion.usage.model_dump(exclude_none=True)


def anthropic_params(model: str, effort: str, prompt: str) -> dict:
    """Messages API request, the same for single requests and batches.

    No refusal fallback: an answer from another model would be scored as this model's.
    """
    return {
        "model": model, "max_tokens": ANTHROPIC_MAX_TOKENS, "thinking": {"type": "adaptive"},
        "output_config": {"effort": effort}, "messages": [{"role": "user", "content": prompt}],
    }


async def create_anthropic(client: anthropic.AsyncAnthropic, model: str, effort: str, prompt: str):
    return await client.messages.create(**anthropic_params(model, effort, prompt))


def answer_anthropic(message, model: str, lat: float, lon: float) -> tuple[str, dict]:
    """(answer text, usage). Raises unless the model finished normally with one non-empty text block."""
    check_served_model(message.model, model)
    texts = [block.text for block in message.content if block.type == "text"]
    other = {block.type for block in message.content} - {"text", "thinking", "redacted_thinking"}
    if message.stop_reason != "end_turn" or len(texts) != 1 or not texts[0].strip() or other:
        raise BadAnswer(f"bad message for ({lat}, {lon}): {message}")
    return texts[0], message.usage.model_dump(exclude_none=True)


BACKENDS = {"anthropic": (create_anthropic, answer_anthropic), "openai": (create_openai, answer_openai)}


def load_api_key(provider: str) -> None:
    """Puts the provider's API key from .env into the environment, unless the environment already has it."""
    load_dotenv(REPO_ROOT / ".env")  # does not override variables already set
    key_var = API_KEY_VARS[provider]
    if not os.environ.get(key_var):
        raise RuntimeError(f"{key_var} is unset or empty. Set it in the environment or in {REPO_ROOT / '.env'} "
                           "(a variable in the environment, even an empty one, hides the one in .env)")


def make_client(provider: str, timeout: float) -> anthropic.AsyncAnthropic | openai.AsyncOpenAI:
    """Client with SDK retries off: retries happen in query_one, so they can lower the concurrency limit."""
    load_api_key(provider)
    if provider == "anthropic":
        return anthropic.AsyncAnthropic(timeout=timeout, max_retries=0)
    return openai.AsyncOpenAI(timeout=timeout, max_retries=0)


def run_dir(model: str, effort: str) -> Path:
    return RESULTS_DIR / f"{model}__{effort}"


def lock_run_dir(out_dir: Path):
    """Exclusive lock on a results directory; hold the returned file open while writing to it.

    Two concurrent queries on the same responses.jsonl would write duplicate records.
    """
    lock = open(out_dir / ".query.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"another query is already writing to {out_dir}") from None
    return lock


def select_points(res: float, n_points: int | None) -> list[tuple[float, float]]:
    """The grid's pixel centres, or n_points of them for a cheap test."""
    points = grid_points(res)
    if n_points is None:
        return points
    if n_points > len(points):
        raise ValueError(f"--points {n_points} is more than the {len(points)} points of the {res}° grid")
    # A fixed random order: spread over the whole grid, and --points 2N contains the points of --points N.
    order = list(range(len(points)))
    random.Random(0).shuffle(order)
    return [points[i] for i in sorted(order[:n_points])]


def load_responses(path: Path, model: str, effort: str) -> dict[tuple[float, float, int], dict]:
    """Existing records keyed by (lat, lon, sample). Raises on duplicates or mismatched prompt/model/effort."""
    records: dict[tuple[float, float, int], dict] = {}
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            rec = json.loads(line)
            key = (rec["lat"], rec["lon"], rec["sample"])
            if key in records:
                raise RuntimeError(f"{path}:{line_no}: duplicate record for {key}")
            if rec["model"] != model or rec["reasoning_effort"] != effort:
                raise RuntimeError(f"{path}:{line_no}: record is for {rec['model']}/{rec['reasoning_effort']}")
            if rec["prompt"] != build_prompt(rec["lat"], rec["lon"]):
                raise RuntimeError(
                    f"{path}:{line_no}: stored prompt differs from the current PROMPT_TEMPLATE. "
                    "Move the old results away before querying with a new prompt."
                )
            records[key] = rec
    return records


async def query_one(client: anthropic.AsyncAnthropic | openai.AsyncOpenAI, limit: AdaptiveLimit, model: str,
                    effort: str, lat: float, lon: float, sample: int) -> dict:
    create, answer = BACKENDS[provider_for(model)]
    prompt = build_prompt(lat, lon)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        async with limit:
            start = time.monotonic()
            try:
                response = await create(client, model, effort, prompt)
            except (openai.APIConnectionError, anthropic.APIConnectionError) as e:  # includes timeouts
                error = f"{type(e).__name__}: {e}"
            except (openai.APIStatusError, anthropic.APIStatusError) as e:
                # A 429 for an exhausted OpenAI quota does not go away by waiting.
                # (Anthropic answers an exhausted credit balance with HTTP 400.)
                if (e.status_code not in RETRY_STATUS
                        or isinstance(e, openai.APIStatusError) and e.code == "insufficient_quota"):
                    raise RuntimeError(f"HTTP {e.status_code} for ({lat}, {lon}): {e.message[:1000]}") from e
                error = f"HTTP {e.status_code}: {e.message[:300]}"
                if e.status_code in (429, 529):
                    limit.rate_limited()
            else:
                limit.success()
                break
        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(f"giving up on ({lat}, {lon}) after {MAX_ATTEMPTS} attempts: {error}")
        # Random jitter so requests that failed together do not retry together.
        delay = min(MAX_DELAY_S, BASE_DELAY_S * 2 ** (attempt - 1)) * random.uniform(0.5, 1.5)
        print(f"  retry {attempt}/{MAX_ATTEMPTS - 1} for ({lat}, {lon}) in {delay:.0f}s "
              f"(limit {int(limit.limit)}): {error}", flush=True)
        await asyncio.sleep(delay)

    content, usage = answer(response, model, lat, lon)
    parse_elevation(content)  # raises ValueError on anything but a plain number
    return {
        "lat": lat,
        "lon": lon,
        "sample": sample,
        "model": model,
        "reasoning_effort": effort,
        "prompt": prompt,
        "response": content,
        "usage": usage,
        "latency_s": round(time.monotonic() - start, 2),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


async def main(args: argparse.Namespace) -> None:
    out_dir = run_dir(args.model, args.effort)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "responses.jsonl"
    lock = lock_run_dir(out_dir)  # held until the process exits
    existing = load_responses(path, args.model, args.effort)

    points = select_points(args.res, args.points)
    todo = [(lat, lon, s) for lat, lon in points for s in range(args.samples) if (lat, lon, s) not in existing]
    total = len(points) * args.samples
    print(f"{args.model} ({args.effort}), {args.res}°/px: {len(points)} points x {args.samples} samples = {total}; "
          f"{total - len(todo)} cached, {len(todo)} to query -> {path}", flush=True)
    if not todo:
        return
    batches = read_batch_log(out_dir / BATCH_LOG)
    if batches.pending or batches.unresolved is not None:
        raise RuntimeError(f"{out_dir / BATCH_LOG} lists batches that are not collected yet. Collect them first "
                           "with moonmapbench.batch (scripts/run.sh --batch ...), so that no point is paid for twice")

    limit = AdaptiveLimit(args.concurrency)
    done = 0
    t0 = time.monotonic()
    async with make_client(provider_for(args.model), args.timeout) as client:
        with open(path, "a", encoding="utf-8") as f:

            async def worker(lat: float, lon: float, sample: int) -> None:
                nonlocal done
                rec = await query_one(client, limit, args.model, args.effort, lat, lon, sample)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                elapsed = time.monotonic() - t0
                print(f"[{done}/{len(todo)} {elapsed:6.0f}s {done / elapsed:4.1f}/s limit {int(limit.limit):3d}] "
                      f"{rec['prompt'].rsplit('. ', 1)[-1]:>24} -> {rec['response']!r}", flush=True)

            async with asyncio.TaskGroup() as tg:
                for lat, lon, sample in todo:
                    tg.create_task(worker(lat, lon, sample))


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=float, default=22.5, help="degrees per pixel (must divide 180)")
    p.add_argument("--samples", type=int, default=1, help="answers per pixel")
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--effort", default="low", help="reasoning effort")
    p.add_argument("--concurrency", type=int, default=256, help="maximum parallel requests (adapts down on rate limits)")
    p.add_argument("--timeout", type=float, default=600, help="per-request timeout in seconds")
    p.add_argument("--points", type=int, help="query only this many grid points, picked at random with a fixed seed (a cheap test)")
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be >= 1")
    if args.concurrency < 1:
        p.error("--concurrency must be >= 1")
    if not args.timeout > 0:
        p.error("--timeout must be > 0")
    if args.points is not None and args.points < 1:
        p.error("--points must be >= 1")
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
