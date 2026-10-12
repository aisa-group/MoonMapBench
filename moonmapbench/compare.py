"""One shareable image with the maps of several models above the ground truth, on the same grid.

Writes <results>/compare_res<res>_n<samples>.png, where <results> is results/ for the Moon and
results/<body>/ for the other bodies. Each model's heading carries its mean absolute error; the
subtitle gives the error of always answering 0 m for scale.

With --difference, each model's map is its answer minus the ground truth instead (red: too high,
blue: too low), without the ground truth map: <results>/difference_res<res>_n<samples>.png.

    python -m moonmapbench.compare --res 2 --runs claude-opus-5-5:low gpt-6.1-sol:low
    python -m moonmapbench.compare --res 2 --runs claude-opus-5-5:low gpt-6.1-sol:low --difference
    python -m moonmapbench.compare --body mars --res 2 --runs gpt-6.1-sol:low
"""

import argparse

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from moonmapbench.bodies import BODIES
from moonmapbench.dem import elevation_at, load_dem
from moonmapbench.grid import grid_points, grid_shape
from moonmapbench.query import load_responses, run_dir
from moonmapbench.report import METERS_PER_FOOT, MODEL_NAMES, collect_predictions, draw_stacked, elevation_cbar_label

# Diverging scale for answer minus truth: blue (too low) - neutral gray - red (too high). The red
# anchors have the OKLab lightness of the blue ones, so both arms look equally strong.
ERROR_CMAP = LinearSegmentedColormap.from_list("error", ["#104281", "#2a78d6", "#f0efec", "#c74845", "#762221"])


def parse_run(text: str) -> tuple[str, str]:
    """'claude-opus-5-5:low' -> ('claude-opus-5-5', 'low')."""
    model, sep, effort = text.rpartition(":")
    if not sep or not model or not effort:
        raise argparse.ArgumentTypeError(f"expected MODEL:EFFORT, got {text!r}")
    return model, effort


def main(args: argparse.Namespace) -> None:
    body = BODIES[args.body]
    points = grid_points(args.res)
    shape = grid_shape(args.res)
    truth = elevation_at(load_dem(body.dem), np.array([lat for lat, _ in points]), np.array([lon for _, lon in points]))

    preds = []
    for model, effort in args.runs:
        records = load_responses(run_dir(body, model, effort) / "responses.jsonl", body, model, effort)
        preds.append((MODEL_NAMES.get(model, model), collect_predictions(records, points, args.samples)[0]))

    if args.difference:
        panels = []
        for name, pred in preds:
            mae = np.abs(pred - truth).mean()
            panels.append((name, f"mean error {mae:,.0f} m ({mae / METERS_PER_FOOT:,.0f} ft)", (pred - truth).reshape(shape)))
        subtitle = (f"Guessed elevation minus ground truth at {len(points):,} points.\n"
                    "Red: guessed too high. Blue: guessed too low.")
        out = body.results_dir / f"difference_res{args.res:g}_n{args.samples}.png"
        draw_stacked(out, panels, args.title or f"Where AI models get {body.name} wrong", subtitle,
                     cmap=ERROR_CMAP, vlim=(-body.error_lim, body.error_lim), ticks=body.error_ticks,
                     cbar_label="Guessed elevation minus ground truth, in meters")
        print(f"wrote {out}")
        return

    panels = [(name, f"mean error {np.abs(pred - truth).mean():,.0f} m", pred.reshape(shape)) for name, pred in preds]
    panels.append((f"Actual map of {body.name}", body.dem_source, truth.reshape(shape)))

    efforts = {effort for _, effort in args.runs}
    effort_text = f", at {efforts.pop()} reasoning effort" if len(efforts) == 1 else ""
    n_runs = len(args.runs)
    who = "Each model was" if n_runs > 2 else ("Both models were" if n_runs == 2 else "The model was")
    samples_text = "" if args.samples == 1 else f" ({args.samples} answers averaged per point)"
    subtitle = (f"{who} asked for the elevation in meters at {len(points):,} points{effort_text}, "
                f"without tools{samples_text}. Answering 0 m everywhere would give a mean error of "
                f"{np.abs(truth).mean():,.0f} m.")
    out = body.results_dir / f"compare_res{args.res:g}_n{args.samples}.png"
    draw_stacked(out, panels, args.title or f"How AI models see {body.name}", subtitle,
                 vlim=body.vlim, ticks=body.ticks, cbar_label=elevation_cbar_label(body), cmap=body.cmap)
    print(f"wrote {out}")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", choices=BODIES, default="moon", help="which body the answers are about")
    p.add_argument("--res", type=float, default=2, help="degrees per pixel (must divide 180)")
    p.add_argument("--samples", type=int, default=1, help="answers per pixel to average")
    p.add_argument("--runs", type=parse_run, nargs="+", required=True, help="MODEL:EFFORT, top to bottom")
    p.add_argument("--difference", action="store_true", help="map each model's answer minus the ground truth")
    p.add_argument("--title", help="replaces the default title")
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be >= 1")
    main(args)


if __name__ == "__main__":
    cli()
