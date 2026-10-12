"""One shareable image with the maps of several models above the ground truth.

Writes <results>/compare_res<res>_n<samples>.png, where <results> is results/ for the Moon and
results/<body>/ for the other bodies. Each model's heading carries its mean absolute error; the
subtitle gives the error of always answering 0 m for scale.

With --difference, each model's map is its answer minus the ground truth instead (red: too high,
blue: too low), without the ground truth map: <results>/difference_res<res>_n<samples>.png.
--output writes to another file instead.

A run given as MODEL:EFFORT:STRIDE is shown on every STRIDE-th row and column of the grid
(see grid.py) and scored on those points; the ground truth map is always the whole grid.

    python -m moonmapbench.compare --res 2 --runs claude-opus-5-5:low gpt-6.1-sol:low
    python -m moonmapbench.compare --res 2 --runs claude-opus-5-5:low gpt-6.1-sol:low --difference
    python -m moonmapbench.compare --body mars --res 2 --runs gpt-6.1-sol:low
    python -m moonmapbench.compare --res 2 --runs claude-opus-5-5:low gpt-6.1-sol:low grok-4.7:low:2 \
        --output results/compare_res2_n1_grok.png
"""

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from moonmapbench.bodies import BODIES
from moonmapbench.dem import Dem, elevation_at, load_dem
from moonmapbench.grid import grid_points, grid_shape
from moonmapbench.query import load_responses, run_dir
from moonmapbench.report import METERS_PER_FOOT, MODEL_NAMES, collect_predictions, draw_stacked, elevation_cbar_label

# Diverging scale for answer minus truth: blue (too low) - neutral gray - red (too high). The red
# anchors have the OKLab lightness of the blue ones, so both arms look equally strong.
NBSP = "\u00a0"
ERROR_CMAP = LinearSegmentedColormap.from_list("error", ["#104281", "#2a78d6", "#f0efec", "#c74845", "#762221"])


def parse_run(text: str) -> tuple[str, str, int]:
    """'claude-opus-5-5:low' -> ('claude-opus-5-5', 'low', 1); 'grok-4.7:low:2' -> ('grok-4.7', 'low', 2)."""
    parts = text.split(":")
    if len(parts) not in (2, 3) or not all(parts) or (len(parts) == 3 and not parts[2].isdigit()):
        raise argparse.ArgumentTypeError(f"expected MODEL:EFFORT or MODEL:EFFORT:STRIDE, got {text!r}")
    return parts[0], parts[1], int(parts[2]) if len(parts) == 3 else 1


def points_text(runs: list[tuple[str, int, int]], res: float) -> str:
    """'16,200 points', or '16,200 points (Grok 4.7: 4,050, 4° apart)' when a run has fewer points."""
    most = max(n for _, n, _ in runs)
    # No-break spaces keep a model name together when the subtitle is wrapped.
    fewer = [f"{name.replace(' ', NBSP)}: {n:,}, {res * stride:g}° apart" for name, n, stride in runs if n != most]
    return f"{most:,} points" + (f" ({'; '.join(fewer)})" if fewer else "")


def truth_at(dem: Dem, points: list[tuple[float, float]]) -> np.ndarray:
    return elevation_at(dem, np.array([lat for lat, _ in points]), np.array([lon for _, lon in points]))


def main(args: argparse.Namespace) -> None:
    body = BODIES[args.body]
    dem = load_dem(body.dem)
    shape = grid_shape(args.res, 1)
    truth = truth_at(dem, grid_points(args.res, 1))

    preds = []  # (name, prediction, truth at its points, image shape, stride)
    for model, effort, stride in args.runs:
        points = grid_points(args.res, stride)
        records = load_responses(run_dir(body, model, effort) / "responses.jsonl", body, model, effort)
        preds.append((MODEL_NAMES.get(model, model), collect_predictions(records, points, args.samples)[0],
                      truth_at(dem, points), grid_shape(args.res, stride), stride))
    where = points_text([(name, len(pred), stride) for name, pred, _, _, stride in preds], args.res)

    if args.difference:
        panels = []
        for name, pred, run_truth, run_shape, _ in preds:
            mae = np.abs(pred - run_truth).mean()
            panels.append((name, f"mean error {mae:,.0f} m ({mae / METERS_PER_FOOT:,.0f} ft)",
                           (pred - run_truth).reshape(run_shape)))
        subtitle = (f"Guessed elevation minus ground truth at {where}.\n"
                    "Red: guessed too high. Blue: guessed too low.")
        out = args.output or body.results_dir / f"difference_res{args.res:g}_n{args.samples}.png"
        draw_stacked(out, panels, args.title or f"Where AI models get {body.name} wrong", subtitle, None,
                     cmap=ERROR_CMAP, vlim=(-body.error_lim, body.error_lim), ticks=body.error_ticks,
                     cbar_label="Guessed elevation minus ground truth, in meters")
        print(f"wrote {out}")
        return

    panels = [(name, f"mean error {np.abs(pred - run_truth).mean():,.0f} m", pred.reshape(run_shape))
              for name, pred, run_truth, run_shape, _ in preds]
    panels.append((f"Actual map of {body.name}", body.dem_source, truth.reshape(shape)))

    efforts = {effort for _, effort, _ in args.runs}
    effort_text = f", at {efforts.pop()} reasoning effort" if len(efforts) == 1 else ""
    n_runs = len(args.runs)
    who = "Each model was" if n_runs > 2 else ("Both models were" if n_runs == 2 else "The model was")
    samples_text = "" if args.samples == 1 else f" ({args.samples} answers averaged per point)"
    subtitle = (f"{who} asked for the elevation in meters at {where}{effort_text}, "
                f"without tools{samples_text}. Answering 0{NBSP}m everywhere would give a mean error of "
                f"{np.abs(truth).mean():,.0f} m.")
    out = args.output or body.results_dir / f"compare_res{args.res:g}_n{args.samples}.png"
    draw_stacked(out, panels, args.title or f"How AI models see {body.name}", subtitle, None,
                 vlim=body.vlim, ticks=body.ticks, cbar_label=elevation_cbar_label(body), cmap=body.cmap)
    print(f"wrote {out}")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", choices=BODIES, default="moon", help="which body the answers are about")
    p.add_argument("--res", type=float, default=2, help="degrees per pixel (must divide 180)")
    p.add_argument("--samples", type=int, default=1, help="answers per pixel to average")
    p.add_argument("--runs", type=parse_run, nargs="+", required=True,
                   help="MODEL:EFFORT or MODEL:EFFORT:STRIDE, top to bottom")
    p.add_argument("--difference", action="store_true", help="map each model's answer minus the ground truth")
    p.add_argument("--title", help="replaces the default title")
    p.add_argument("--output", type=Path, help="write the image here instead of the default name")
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be >= 1")
    main(args)


if __name__ == "__main__":
    cli()
