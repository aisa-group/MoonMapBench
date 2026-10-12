"""Query a Claude model at every pixel centre through the Message Batches API, at half price.

The requests and saved records are the same as in moonmapbench.query (records also get the
batch_id), but all missing points go out as one batch. Results arrive within 24 h, usually
within 1 h. Before submitting, the cost of the batch is estimated from the mean tokens per
answer already saved for this model and effort. Nothing is sent if that estimate plus the
cost of the saved answers is above --max-cost.

Each submission is logged in batches.jsonl next to responses.jsonl (see batchlog.py) just
before and just after it is sent. A rerun first waits for and collects every batch that is
not collected yet, so it never sends a point that is still in a batch, and an interrupted run
loses nothing. If a run died while sending, the rerun stops and lists the batches created
since then, so that they can be adopted or cancelled by hand. Each request's custom_id
encodes its (lat, lon, sample).

Answers that cannot be used (API errors, refusals, answers that are not a plain number) are
not saved. The good answers of a batch are saved, then the run stops with a list of the
failures; a rerun asks those points again.

    python -m moonmapbench.batch --res 2 --model claude-opus-5-5 --effort low --max-cost 100
    python -m moonmapbench.batch --body mars --res 2 --model claude-opus-5-5 --effort low --max-cost 100
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic

from moonmapbench.batchlog import BATCH_LOG, append_batch_log, read_batch_log
from moonmapbench.bodies import BODIES, Body
from moonmapbench.cost import cost, estimate_batch_cost
from moonmapbench.grid import grid_shape
from moonmapbench.prompt import build_prompt, parse_elevation
from moonmapbench.query import (BadAnswer, anthropic_params, answer_anthropic, load_api_key, load_responses,
                                lock_run_dir, provider_for, run_dir, select_points, stop_on_failures)

POLL_S = 60
REPORT_EVERY_S = 600  # print the batch status at least this often while waiting
MAX_BATCH_REQUESTS = 100_000  # API limit per batch
SDK_RETRIES = 8  # for the GET requests while polling for hours; never for creating a batch


def custom_id(lat: float, lon: float, sample: int) -> str:
    """E.g. (-11.25, 33.75, 0) -> '-11p25_33p75_0'. Custom IDs may only use [a-zA-Z0-9_-], up to 64."""
    cid = f"{lat!r}_{lon!r}_{sample}".replace(".", "p")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cid) or parse_custom_id(cid) != (lat, lon, sample):
        raise ValueError(f"cannot encode {(lat, lon, sample)} as a custom_id: {cid!r}")
    return cid


def parse_custom_id(cid: str) -> tuple[float, float, int]:
    lat, lon, sample = cid.split("_")
    return float(lat.replace("p", ".")), float(lon.replace("p", ".")), int(sample)


def params_sha256(body: Body, model: str, effort: str) -> str:
    """Fingerprint of the request settings and prompt template. Collecting checks that they did not change."""
    params = anthropic_params(model, effort, body.prompt_template)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def wait(client: anthropic.Anthropic, batch_id: str):
    """Polls until the batch has ended and returns it."""
    t0 = time.monotonic()
    last, last_print = None, -float("inf")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        if counts != last or time.monotonic() - last_print >= REPORT_EVERY_S:
            print(f"[{(time.monotonic() - t0) / 60:5.0f} min] {batch_id} {batch.processing_status}: "
                  f"{counts.processing} processing, {counts.succeeded} succeeded, {counts.errored} errored, "
                  f"{counts.canceled} canceled, {counts.expired} expired", flush=True)
            last, last_print = counts, time.monotonic()
        if batch.processing_status == "ended":
            return batch
        time.sleep(POLL_S)


def collect(client: anthropic.Anthropic, entry: dict, body: Body, model: str, effort: str, existing: dict,
            responses: Path, log: Path) -> list[str]:
    """Waits for a submitted batch, appends its good answers to `responses` and returns the failures."""
    batch_id = entry["batch_id"]
    if entry["params_sha256"] != params_sha256(body, model, effort):
        raise RuntimeError(f"{batch_id} was sent with other request settings or another prompt template "
                           "than the current code would use; its answers would be saved with the wrong prompt")
    batch = wait(client, batch_id)
    counts = batch.request_counts
    expected = counts.succeeded + counts.errored + counts.canceled + counts.expired
    if expected != entry["n_requests"]:
        raise RuntimeError(f"{batch_id} reports {expected} results, but {entry['n_requests']} requests were sent")
    seen: set[str] = set()
    saved, failures = 0, []
    with open(responses, "a", encoding="utf-8") as f:
        for item in client.messages.batches.results(batch_id):
            if item.custom_id in seen:
                raise RuntimeError(f"{batch_id} returned {item.custom_id} twice")
            seen.add(item.custom_id)
            lat, lon, sample = key = parse_custom_id(item.custom_id)
            if key in existing:
                # Saved by an earlier, interrupted collection of this same batch.
                if existing[key].get("batch_id") != batch_id:
                    raise RuntimeError(f"{batch_id} has an answer for {key}, which already has one from elsewhere")
                continue
            result = item.result
            if result.type != "succeeded":
                detail = f": {result.error.error.type}: {result.error.error.message}" if result.type == "errored" else ""
                failures.append(f"{key}: {result.type}{detail}")
                continue
            try:
                content, usage = answer_anthropic(result.message, model, lat, lon)
                parse_elevation(content)
            except (BadAnswer, ValueError) as e:  # saved points stay saved; a rerun asks this one again
                failures.append(f"{key}: {e}")
                continue
            rec = {
                "lat": lat,
                "lon": lon,
                "sample": sample,
                "model": model,
                "reasoning_effort": effort,
                "prompt": build_prompt(body, model, lat, lon),
                "response": content,
                "usage": usage,
                "batch_id": batch_id,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            existing[key] = rec
            saved += 1
    if len(seen) != entry["n_requests"]:
        raise RuntimeError(f"{batch_id}: got {len(seen)} results for {entry['n_requests']} requests")
    append_batch_log(log, {"event": "collected", "batch_id": batch_id, "saved": saved, "failed": len(failures)})
    print(f"{batch_id}: saved {saved} answers, {len(failures)} failed", flush=True)
    return failures


def stop_on_unresolved(client: anthropic.Anthropic, entry: dict, log: Path) -> None:
    """A submission without a recorded outcome: list the batches that might be it, then stop."""
    since = datetime.fromisoformat(entry["time"]) - timedelta(minutes=5)
    found = [f"{b.id}, created {b.created_at.isoformat(timespec='seconds')}, {b.processing_status}, "
             f"{sum(b.request_counts.to_dict().values())} requests"
             for b in client.messages.batches.list() if b.created_at >= since]
    submitted = json.dumps({"event": "submitted", "batch_id": "<id>", "n_requests": entry["n_requests"],
                            "params_sha256": entry["params_sha256"]})
    raise RuntimeError(
        f"the submission of {entry['n_requests']} requests at {entry['time']} has no recorded outcome "
        f"(the run died or the request failed while sending). Batches created since then:\n  "
        + ("\n  ".join(found) or "none")
        + f"\nIf one of them is this submission, append this line to {log} so that a rerun collects it "
        f"(and cancel any duplicates):\n  {submitted}\n"
        f"If none is, append:\n  {json.dumps({'event': 'submit_rejected', 'error': 'no batch was created'})}"
    )


def main(args: argparse.Namespace) -> None:
    if provider_for(args.model) != "anthropic":
        raise ValueError(f"batches are implemented for Claude models only, not {args.model}")
    body = BODIES[args.body]
    grid_shape(args.res, args.stride)  # raises on a bad grid before anything is created or collected
    out_dir = run_dir(body, args.model, args.effort)
    out_dir.mkdir(parents=True, exist_ok=True)
    responses = out_dir / "responses.jsonl"
    log = out_dir / BATCH_LOG
    lock = lock_run_dir(out_dir)  # held until the process exits
    existing = load_responses(responses, body, args.model, args.effort)
    load_api_key("anthropic")
    client = anthropic.Anthropic(max_retries=SDK_RETRIES)

    state = read_batch_log(log)
    if state.unresolved is not None:
        stop_on_unresolved(client, state.unresolved, log)
    for entry in state.pending:
        print(f"collecting earlier batch {entry['batch_id']}", flush=True)
        stop_on_failures(collect(client, entry, body, args.model, args.effort, existing, responses, log))

    points = select_points(args.res, args.stride, args.points)
    todo = [(lat, lon, s) for lat, lon in points for s in range(args.samples) if (lat, lon, s) not in existing]
    total = len(points) * args.samples
    print(f"{body.title}, {args.model} ({args.effort}), {args.res}°/px, stride {args.stride}: "
          f"{len(points)} points x {args.samples} samples "
          f"= {total}; {total - len(todo)} cached, {len(todo)} to query -> {responses}", flush=True)
    if not todo:
        return
    if len(todo) > MAX_BATCH_REQUESTS:
        raise ValueError(f"{len(todo)} requests is more than the {MAX_BATCH_REQUESTS} a batch may hold")
    estimate = estimate_batch_cost(args.model, existing, len(todo))
    spent = sum(cost(args.model, rec["usage"]) for rec in existing.values())
    print(f"estimated cost of this batch ${estimate:.2f} (mean tokens of {len(existing)} saved answers); "
          f"saved answers so far ${spent:.2f}; budget ${args.max_cost:g} for both", flush=True)
    if spent + estimate > args.max_cost:
        raise RuntimeError(f"${spent:.2f} spent + ${estimate:.2f} estimated is above --max-cost {args.max_cost:g}; "
                           "nothing was sent")

    requests = [
        {"custom_id": custom_id(lat, lon, s),
         "params": anthropic_params(args.model, args.effort, build_prompt(body, args.model, lat, lon))}
        for lat, lon, s in todo
    ]
    submission = {"n_requests": len(requests), "params_sha256": params_sha256(body, args.model, args.effort),
                  "estimated_cost_usd": round(estimate, 2)}
    append_batch_log(log, {"event": "submitting", **submission})
    try:
        # No SDK retries: creating a batch is not idempotent, and a retry after a lost response
        # would create a second batch that nobody collects.
        batch = client.with_options(max_retries=0).messages.batches.create(requests=requests)
    except anthropic.APIStatusError as e:
        if 400 <= e.status_code < 500 and e.status_code not in (408, 409):  # refused, so no batch exists
            append_batch_log(log, {"event": "submit_rejected", "error": f"HTTP {e.status_code}: {e.message[:300]}"})
        raise
    entry = {"event": "submitted", "batch_id": batch.id, **submission}
    append_batch_log(log, entry)
    print(f"submitted {batch.id} with {len(requests)} requests", flush=True)
    stop_on_failures(collect(client, entry, body, args.model, args.effort, existing, responses, log))


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", choices=BODIES, default="moon", help="which body to ask about")
    p.add_argument("--res", type=float, default=22.5, help="degrees per pixel (must divide 180)")
    p.add_argument("--stride", type=int, default=1, help="query every STRIDE-th row and column of the grid (see grid.py)")
    p.add_argument("--samples", type=int, default=1, help="answers per pixel")
    p.add_argument("--model", required=True, help="a Claude model, e.g. claude-opus-5-5")
    p.add_argument("--effort", default="low", help="reasoning effort")
    p.add_argument("--points", type=int, help="query only this many grid points, picked at random with a fixed seed")
    p.add_argument("--max-cost", type=float, required=True,
                   help="USD for the saved answers of this model and effort plus the estimated batch; "
                        "do not submit if that is higher")
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be >= 1")
    if args.points is not None and args.points < 1:
        p.error("--points must be >= 1")
    if not args.max_cost > 0:
        p.error("--max-cost must be > 0")
    main(args)


if __name__ == "__main__":
    cli()
