"""One shareable image with the maps of several models above the ground truth, on the same grid.

Writes results/compare_res<res>_n<samples>.png. Each model's heading carries its mean absolute
error; the subtitle gives the error of always answering 0 m for scale.

    python -m moonmapbench.compare --res 2 --runs claude-opus-5-5:low gpt-6.1-sol:low
"""

import argparse

import numpy as np

from moonmapbench import RESULTS_DIR
from moonmapbench.grid import grid_points, grid_shape
from moonmapbench.lola import elevation_at, load_ldem
from moonmapbench.query import load_responses, run_dir
from moonmapbench.report import MODEL_NAMES, collect_predictions, draw_stacked


def parse_run(text: str) -> tuple[str, str]:
    """'claude-opus-5-5:low' -> ('claude-opus-5-5', 'low')."""
    model, sep, effort = text.rpartition(":")
    if not sep or not model or not effort:
        raise argparse.ArgumentTypeError(f"expected MODEL:EFFORT, got {text!r}")
    return model, effort


def main(args: argparse.Namespace) -> None:
    points = grid_points(args.res)
    shape = grid_shape(args.res)
    truth = elevation_at(load_ldem(), np.array([lat for lat, _ in points]), np.array([lon for _, lon in points]))

    panels = []
    for model, effort in args.runs:
        records = load_responses(run_dir(model, effort) / "responses.jsonl", model, effort)
        pred, _ = collect_predictions(records, points, args.samples)
        panels.append((MODEL_NAMES.get(model, model), f"mean error {np.abs(pred - truth).mean():,.0f} m", pred.reshape(shape)))
    panels.append(("Actual map of the Moon", "NASA LRO laser altimeter", truth.reshape(shape)))

    efforts = {effort for _, effort in args.runs}
    effort_text = f", at {efforts.pop()} reasoning effort" if len(efforts) == 1 else ""
    n_runs = len(args.runs)
    who = "Each model was" if n_runs > 2 else ("Both models were" if n_runs == 2 else "The model was")
    samples_text = "" if args.samples == 1 else f" ({args.samples} answers averaged per point)"
    subtitle = (f"{who} asked for the elevation in meters at {len(points):,} points{effort_text}, "
                f"without tools{samples_text}. Answering 0 m everywhere would give a mean error of "
                f"{np.abs(truth).mean():,.0f} m.")
    out = RESULTS_DIR / f"compare_res{args.res:g}_n{args.samples}.png"
    draw_stacked(out, panels, args.title, subtitle)
    print(f"wrote {out}")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=float, default=2, help="degrees per pixel (must divide 180)")
    p.add_argument("--samples", type=int, default=1, help="answers per pixel to average")
    p.add_argument("--runs", type=parse_run, nargs="+", required=True, help="MODEL:EFFORT, top to bottom")
    p.add_argument("--title", default="How AI models see the Moon")
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be >= 1")
    main(args)


if __name__ == "__main__":
    cli()
