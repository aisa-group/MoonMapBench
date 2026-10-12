"""Leaderboard: how much smaller each run's mean absolute error (MAE) is than always answering the mean.

Score = 1 - MAE / MAE of the baseline, in percent, higher is better. The baseline answers the average
true elevation of the scored points everywhere (a plain mean over the points, which over-weights the
poles as the MAE does); that constant uses the ground truth, which a blind model cannot. A run below
0% does worse than that constant. Each run is scored on its own points, as in moonmapbench.compare
(MODEL:EFFORT for the whole grid, MODEL:EFFORT:STRIDE for every STRIDE-th row and column), and so is
its baseline.

Draws a column chart to <results>/leaderboard_res<res>_n<samples>.png (or --output) and prints the
same ranking as a Markdown table.

    python -m moonmapbench.leaderboard --res 2 --runs claude-opus-5-5:low gpt-6-astra:low grok-4.7:low:2
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D

from moonmapbench.bodies import BODIES
from moonmapbench.compare import parse_run, truth_at
from moonmapbench.dem import load_dem
from moonmapbench.grid import grid_points
from moonmapbench.query import load_responses, run_dir
from moonmapbench.report import (MODEL_NAMES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, _text_width_in,
                                 _wrap_to_width, collect_predictions)

BAR = "#2a78d6"  # one series, one hue (the default palette's first slot)
TEXT_MUTED = "#898781"
DPI = 150
LINE_IN = 1.35 / 72  # line height per point of font size, in inches
BAR_FRACTION = 0.56  # of each column's slot; the rest is air
CORNER_IN = 6 / DPI  # rounded end of a column
LABEL_ROOM_IN = 0.5  # above the highest and below the lowest column, for its value
BASELINE_LABEL = "always answering the mean"
MINUS = "−"


def score_text(score: float) -> str:
    return f"+{score:.0%}" if score >= 0 else f"{MINUS}{-score:.0%}"


def column(x_left: float, width: float, height: float, r_x: float, r_y: float) -> PathPatch:
    """A column from y=0 up to `height`, square at y=0, with rounded corners at the far end (radii in data units)."""
    top = height - r_y
    vertices = [(x_left, 0), (x_left, top), (x_left, height), (x_left + r_x, height), (x_left + width - r_x, height),
                (x_left + width, height), (x_left + width, top), (x_left + width, 0), (x_left, 0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3,
             MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    return PathPatch(MplPath(vertices, codes), fc=BAR, ec="none", zorder=2)


def draw_leaderboard(path: Path, rows: list[tuple[float, str, float]], title: str, subtitle: str) -> None:
    """Columns, best (highest score) on the left. rows: (score, name, MAE in m)."""
    rows = sorted(rows, reverse=True)
    names = [name for _, name, _ in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"two columns would have the same name: {names}")
    width, margin, plot_h = 8.0, 0.4, 3.6  # inches
    title_pt, subtitle_pt, value_pt, name_pt, mae_pt, base_pt = 22, 13.5, 20, 14, 12, 11
    fig = plt.figure(figsize=(width, 1), dpi=DPI, facecolor=SURFACE)
    plot_w = width - 2 * margin
    slot_in = plot_w / len(rows)
    title = _wrap_to_width(fig, title, title_pt, plot_w, fontweight="bold")
    subtitle = _wrap_to_width(fig, subtitle, subtitle_pt, plot_w)
    wrapped = [_wrap_to_width(fig, name, name_pt, 0.95 * slot_in, fontweight="bold") for name in names]
    for line in "\n".join(wrapped).split("\n"):
        if _text_width_in(fig, line, name_pt, "bold") > slot_in:
            raise ValueError(f"the word {line!r} is wider than a column ({slot_in:.2f} in)")
    name_lines = max(name.count("\n") + 1 for name in wrapped)
    # Vertical layout, top to bottom, in inches.
    blocks = [
        ("margin", 0.3),
        ("title", (title.count("\n") + 1) * title_pt * LINE_IN),
        ("gap", 0.08),
        ("subtitle", (subtitle.count("\n") + 1) * subtitle_pt * LINE_IN),
        ("gap", 0.15),
        ("plot", plot_h),
        ("gap", 0.1),
        ("names", name_lines * name_pt * LINE_IN),
        ("mae", mae_pt * LINE_IN),
        ("margin", 0.3),
    ]
    height = sum(h for _, h in blocks)
    fig.set_size_inches(width, height)
    top, cursor = {}, 0.0
    for key, h in blocks:
        top[key] = cursor
        cursor += h

    def y(inches_from_top: float) -> float:
        return 1 - inches_from_top / height

    x0 = margin / width
    fig.text(x0, y(top["title"]), title, ha="left", va="top", fontsize=title_pt, fontweight="bold", color=TEXT_PRIMARY)
    fig.text(x0, y(top["subtitle"]), subtitle, ha="left", va="top", fontsize=subtitle_pt, color=TEXT_SECONDARY,
             linespacing=1.25)

    ax = fig.add_axes((x0, y(top["plot"] + plot_h), plot_w / width, plot_h / height))
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()
    scores = [score for score, _, _ in rows]
    lo, hi = min(0.0, *scores), max(0.0, *scores)
    y_per_in = (hi - lo) / (plot_h - 2 * LABEL_ROOM_IN)
    ax.set_xlim(-0.5, len(rows) - 0.5)
    ax.set_ylim(lo - LABEL_ROOM_IN * y_per_in, hi + LABEL_ROOM_IN * y_per_in)
    x_per_in = 1 / slot_in
    bar_w = BAR_FRACTION * slot_in * x_per_in
    gap = 0.08 * y_per_in  # between a column's end and its value
    for i, ((score, _, mae), name_text) in enumerate(zip(rows, wrapped)):
        patch = column(i - bar_w / 2, bar_w, abs(score), CORNER_IN * x_per_in, CORNER_IN * y_per_in)
        if score < 0:  # pointing down: mirror it at the zero line
            patch.set_transform(Affine2D().scale(1, -1) + ax.transData)
        ax.add_patch(patch)
        ax.text(i, score + (gap if score >= 0 else -gap), score_text(score), ha="center",
                va="bottom" if score >= 0 else "top", fontsize=value_pt, fontweight="bold", color=TEXT_PRIMARY)
        x_fig = (margin + (i + 0.5) * slot_in) / width
        fig.text(x_fig, y(top["names"]), name_text, ha="center", va="top", fontsize=name_pt, fontweight="bold",
                 color=TEXT_PRIMARY, linespacing=1.15)
        fig.text(x_fig, y(top["mae"]), f"mean error {mae:,.0f} m", ha="center", va="top", fontsize=mae_pt,
                 color=TEXT_MUTED)
    ax.axhline(0, color=TEXT_SECONDARY, lw=1.3, zorder=3)

    # The zero line's label: at the right end above the line, or else at the left end below it,
    # wherever no column is in the way.
    label_w = _text_width_in(fig, BASELINE_LABEL, base_pt) * x_per_in

    def blocked(x_from: float, x_to: float, above: bool) -> bool:
        return any(abs(i - (x_from + x_to) / 2) < bar_w / 2 + (x_to - x_from) / 2 and (score > 0) == above
                   for i, score in enumerate(scores))

    if not blocked(len(rows) - 0.5 - label_w, len(rows) - 0.5, above=True):
        ax.text(len(rows) - 0.5, gap, BASELINE_LABEL, ha="right", va="bottom", fontsize=base_pt,
                color=TEXT_SECONDARY, style="italic")
    elif not blocked(-0.5, -0.5 + label_w, above=False):
        ax.text(-0.5, -gap, BASELINE_LABEL, ha="left", va="top", fontsize=base_pt, color=TEXT_SECONDARY,
                style="italic")
    else:
        raise ValueError(f"no free place for the label {BASELINE_LABEL!r} on the zero line")
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def mean_baseline_mae(truth: np.ndarray) -> float:
    return float(np.abs(truth - truth.mean()).mean())


def main(args: argparse.Namespace) -> None:
    body = BODIES[args.body]
    dem = load_dem(body.dem)
    rows = []  # (score, name, MAE)
    for model, effort, stride in args.runs:
        points = grid_points(args.res, stride)
        records = load_responses(run_dir(body, model, effort) / "responses.jsonl", body, model, effort)
        pred = collect_predictions(records, points, args.samples)[0]
        truth = truth_at(dem, points)
        mae = float(np.abs(pred - truth).mean())
        rows.append((1 - mae / mean_baseline_mae(truth), MODEL_NAMES.get(model, model), mae))

    subtitle = ("How much smaller the mean elevation error is than always answering the average elevation. "
                "Higher is better.")
    out = args.output or body.results_dir / f"leaderboard_res{args.res:g}_n{args.samples}.png"
    draw_leaderboard(out, rows, args.title or f"Which AI model knows {body.name} best?", subtitle)
    print(f"wrote {out}\n")

    print("| # | Model | Better than always the mean | Mean error |")
    print("|---|---|---|---|")
    for rank, (score, name, mae) in enumerate(sorted(rows, reverse=True), 1):
        print(f"| {rank} | {name} | **{score_text(score)}** | {mae:,.0f} m |")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", choices=BODIES, default="moon", help="which body the answers are about")
    p.add_argument("--res", type=float, default=2, help="degrees per pixel (must divide 180)")
    p.add_argument("--samples", type=int, default=1, help="answers per pixel to average")
    p.add_argument("--runs", type=parse_run, nargs="+", required=True, help="MODEL:EFFORT or MODEL:EFFORT:STRIDE")
    p.add_argument("--title", help="replaces the default title")
    p.add_argument("--output", type=Path, help="write the image here instead of the default name")
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be >= 1")
    main(args)


if __name__ == "__main__":
    cli()
