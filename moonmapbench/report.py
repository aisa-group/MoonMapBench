"""Score a run against the body's ground-truth DEM (LOLA for the Moon) and draw the maps.

Metric: mean absolute error (MAE) in meters, averaged over all queried pixels of the grid.
For reference we also report the MAE of always answering 0 m (the reference surface),
and the MAE of answering each pixel's exact cell-average elevation: the truth is the
height at the pixel centre, so even a model that knew every cell's average would not
score 0. With --stride, the cell is the `res` cell of the queried point, not the block it stands for.
With several samples per pixel the prediction is their mean, as the Earth post
averaged its samples per pixel.

Writes <results>/<model>__<effort>/res<res>_n<samples>/{metrics.json, points.csv, maps.png, side_by_side*.png,
stacked.png}, where <results> is results/ for the Moon and results/<body>/ for the other bodies.
With --stride k > 1 the directory is res<res>_stride<k>_n<samples>.

    python -m moonmapbench.report --res 22.5 --samples 1
    python -m moonmapbench.report --res 2 --stride 2 --model grok-4.7 --effort low
    python -m moonmapbench.report --body mars --res 2
"""

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap

from moonmapbench.bodies import BODIES, Body
from moonmapbench.dem import Dem, cell_means, elevation_at, load_dem, overview_map
from moonmapbench.grid import grid_name, grid_points, grid_shape
from moonmapbench.prompt import parse_elevation
from moonmapbench.query import load_responses, run_dir

# Every map of a body has the body's fixed color range and colormap (Body.vlim, Body.cmap), so images are comparable.
METERS_PER_FOOT = 0.3048  # for US readers of the shareable image
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
AXIS = "#b9b8b3"
# The highlighted line of the shareable image (the mean error): dark red on a light red rounded box.
HIGHLIGHT_TEXT = "#9b2f2c"
HIGHLIGHT_FILL = "#fbe1de"
HIGHLIGHT_PAD = 0.35  # box padding, in units of the font size
# Display names for figure titles; other models are shown by their id.
MODEL_NAMES = {"gpt-6.1-sol": "GPT 6.1 Sol", "gpt-6-astra": "GPT 6 Astra", "claude-opus-5-5": "Claude Opus 5.5",
               "grok-4.7": "Grok 4.7"}


def collect_predictions(records: dict, points: list[tuple[float, float]], samples: int) -> tuple[np.ndarray, list[list[str]]]:
    """Mean of the parsed answers per point. Raises if any answer is missing."""
    missing = [(lat, lon, s) for lat, lon in points for s in range(samples) if (lat, lon, s) not in records]
    if missing:
        raise RuntimeError(f"{len(missing)} answers missing (first: {missing[:3]}). Run moonmapbench.query first.")
    answers = [[records[(lat, lon, s)]["response"] for s in range(samples)] for lat, lon in points]
    return np.array([np.mean([parse_elevation(t) for t in texts]) for texts in answers]), answers


def _style_map_axes(ax: plt.Axes) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks([-180, -90, 0, 90, 180], ["180°W", "90°W", "0°", "90°E", "180°E"])
    ax.set_yticks([-90, -45, 0, 45, 90], ["90°S", "45°S", "0°", "45°N", "90°N"])
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3, color=AXIS)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)


def _draw_map(ax: plt.Axes, image: np.ndarray, title: str, title_size: float, body: Body):
    ax.set_facecolor(SURFACE)
    im = ax.imshow(image, extent=(-180, 180, -90, 90), cmap=body.cmap, vmin=body.vlim[0], vmax=body.vlim[1],
                   interpolation="nearest", aspect="equal")
    ax.set_title(title, loc="left", fontsize=title_size, color=TEXT_PRIMARY, pad=6)
    _style_map_axes(ax)
    return im


def _add_colorbar(fig: plt.Figure, im, rect: tuple[float, float, float, float], label: str,
                  label_size: float = 9, tick_size: float = 8, ticks: tuple[int, ...] | None = None) -> None:
    cbar = fig.colorbar(im, cax=fig.add_axes(rect), orientation="horizontal", extend="both")
    if ticks is not None:
        cbar.set_ticks(ticks, labels=[f"{t:,}".replace("-", "\u2212") for t in ticks])
    cbar.set_label(label, color=TEXT_SECONDARY, fontsize=label_size)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=tick_size, color=AXIS)
    cbar.outline.set_edgecolor(AXIS)


def points_text(res: float, stride: int) -> str:
    """Where the grid was queried, for figure titles."""
    return f"{res:g}° pixel centres" if stride == 1 else f"{res:g}° pixel centres {res * stride:g}° apart"


def draw_maps(path, body: Body, dem: Dem, truth_grid: np.ndarray, pred_grid: np.ndarray, res: float, stride: int,
              model_label: str, mae: float, zero_mae: float, n_points: int) -> None:
    """Three stacked maps: the DEM at 0.5°, the DEM at the query points, model prediction."""
    panels = [
        (overview_map(dem, 0.5), f"{body.title} ({body.dem_short}, 0.5° pixels)"),
        (truth_grid, f"{body.title} ({body.dem_short} at {points_text(res, stride)})"),
        (pred_grid, f"{model_label}, blind ({res * stride:g}° pixels)"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 12.6), facecolor=SURFACE)
    for ax, (image, title) in zip(axes, panels):
        im = _draw_map(ax, image, title, 11, body)
    fig.suptitle(f"How does {model_label} see {body.name}?", x=0.06, ha="left", fontsize=15, color=TEXT_PRIMARY, y=0.985)
    fig.text(0.06, 0.957,
             f"Mean absolute error {mae:,.0f} m over {n_points} pixels  ·  always answering 0 m: {zero_mae:,.0f} m",
             ha="left", fontsize=10, color=TEXT_SECONDARY)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.09, hspace=0.22)
    _add_colorbar(fig, im, (0.25, 0.04, 0.5, 0.012), f"Elevation relative to {body.datum} (m)")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def draw_side_by_side(path, body: Body, truth_image: np.ndarray, truth_title: str, pred_grid: np.ndarray,
                      pixel_deg: float, model_label: str, mae: float, n_points: int) -> None:
    """The real body and the model's map next to each other."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 5.4), facecolor=SURFACE)
    _draw_map(left, truth_image, truth_title, 13, body)
    im = _draw_map(right, pred_grid, f"{model_label}, blind ({pixel_deg:g}° pixels)", 13, body)
    fig.suptitle(f"How does {model_label} see {body.name}?", x=0.04, ha="left", fontsize=17, color=TEXT_PRIMARY, y=0.97)
    fig.text(0.04, 0.885, f"Mean absolute error {mae:,.0f} m over {n_points} pixels",
             ha="left", fontsize=11, color=TEXT_SECONDARY)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.80, bottom=0.2, wspace=0.08)
    _add_colorbar(fig, im, (0.35, 0.075, 0.3, 0.03), f"Elevation relative to {body.datum} (m)")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _text_width_in(fig: plt.Figure, text: str, fontsize: float, fontweight: str = "normal") -> float:
    """Rendered width of `text` in inches."""
    t = fig.text(0, 0, text, fontsize=fontsize, fontweight=fontweight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def _wrap_to_width(fig: plt.Figure, text: str, fontsize: float, max_width_in: float,
                   fontweight: str = "normal") -> str:
    """Greedy word wrap of each line of `text`, measuring rendered widths. Breaks only at plain spaces,
    so a no-break space (\u00a0) keeps two words together."""

    def width_in(candidate: str) -> float:
        return _text_width_in(fig, candidate, fontsize, fontweight)

    out = []
    for paragraph in text.split("\n"):
        line = ""
        for word in (w for w in paragraph.split(" ") if w):
            candidate = f"{line} {word}".strip()
            if line and width_in(candidate) > max_width_in:
                out.append(line)
                line = word
            else:
                line = candidate
        out.append(line)
    return "\n".join(out)


def elevation_cbar_label(body: Body) -> str:
    return f"Elevation in meters, relative to {body.datum_long}"


def draw_stacked(path, panels: list[tuple[str | None, str | None, np.ndarray]], title: str, subtitle: str,
                 highlight: str | None, vlim: tuple[float, float], ticks: tuple[int, ...], cbar_label: str,
                 cmap: Colormap) -> None:
    """Maps stacked top to bottom, no axes. Large text so it reads when shared as an image.

    `highlight`, if given, goes below the subtitle in bold dark red on a light red box.

    Each panel is (heading, note, image): the heading is bold above the map's left edge, the note
    is right-aligned on the same line (same size, so both share a baseline). A first panel without
    a heading is introduced by the title. All maps share one color scale (cmap, vlim) and colorbar.
    """
    width, margin = 10.0, 0.35  # inches
    map_w = width - 2 * margin
    map_h = map_w / 2
    title_pt, subtitle_pt, heading_pt, cbar_label_pt, cbar_tick_pt = 28, 18, 24, 17, 15
    line_in = 1.35 / 72  # line height per point of font size, in inches
    fig = plt.figure(figsize=(width, 1), dpi=150, facecolor=SURFACE)  # height set below
    title = _wrap_to_width(fig, title, title_pt, map_w, fontweight="bold")
    n_title = title.count("\n") + 1
    subtitle = _wrap_to_width(fig, subtitle, subtitle_pt, map_w)
    n_sub = subtitle.count("\n") + 1
    # Vertical layout, top to bottom, in inches.
    blocks = [
        ("margin", 0.3),
        ("title", n_title * title_pt * line_in),
        ("gap", 0.08),
        ("subtitle", n_sub * subtitle_pt * line_in),
    ]
    pad_in = HIGHLIGHT_PAD * subtitle_pt / 72
    if highlight is not None:
        highlight = _wrap_to_width(fig, highlight, subtitle_pt, map_w - 2 * pad_in, fontweight="bold")
        blocks += [("gap", 0.06), ("highlight", (highlight.count("\n") + 1) * subtitle_pt * line_in + 2 * pad_in)]
    for i, (heading, _, _) in enumerate(panels):
        blocks.append(("gap", 0.3))
        if heading is not None:
            blocks += [(f"heading{i}", heading_pt * line_in), ("gap", 0.12)]
        blocks.append((f"map{i}", map_h))
    blocks += [
        ("gap", 0.35),
        ("cbar", 0.22),
        ("cbar_text", (cbar_tick_pt + cbar_label_pt) * line_in + 0.1),
        ("margin", 0.25),
    ]
    height = sum(h for _, h in blocks)
    fig.set_size_inches(width, height)
    top = {}
    cursor = 0.0
    for name, h in blocks:
        top[name] = cursor
        cursor += h

    def y(inches_from_top: float) -> float:
        return 1 - inches_from_top / height

    x0 = margin / width
    fig.text(x0, y(top["title"]), title, ha="left", va="top", fontsize=title_pt, fontweight="bold", color=TEXT_PRIMARY)
    fig.text(x0, y(top["subtitle"]), subtitle, ha="left", va="top", fontsize=subtitle_pt, color=TEXT_SECONDARY,
             linespacing=1.25)
    if highlight is not None:  # the box's left edge on the margin
        fig.text(x0 + pad_in / width, y(top["highlight"] + pad_in), highlight, ha="left", va="top",
                 fontsize=subtitle_pt, fontweight="bold", color=HIGHLIGHT_TEXT, linespacing=1.25,
                 bbox=dict(boxstyle=f"round,pad={HIGHLIGHT_PAD}", fc=HIGHLIGHT_FILL, ec="none"))
    for i, (heading, note, image) in enumerate(panels):
        if heading is not None:
            fig.text(x0, y(top[f"heading{i}"]), heading, ha="left", va="top", fontsize=heading_pt, fontweight="bold",
                     color=TEXT_PRIMARY)
            if note is not None:
                room = map_w - _text_width_in(fig, heading, heading_pt, "bold") - _text_width_in(fig, note, heading_pt)
                if room < 0.3:
                    raise ValueError(f"heading {heading!r} and note {note!r} do not fit on one line")
                fig.text(1 - x0, y(top[f"heading{i}"]), note, ha="right", va="top", fontsize=heading_pt,
                         color=TEXT_SECONDARY)
        elif note is not None:
            raise ValueError("a note needs a heading")
        ax = fig.add_axes((x0, y(top[f"map{i}"] + map_h), map_w / width, map_h / height))
        im = ax.imshow(image, extent=(-180, 180, -90, 90), cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                       interpolation="nearest", aspect="equal")
        ax.set_axis_off()
    cbar_w = map_w * 0.7
    _add_colorbar(fig, im, ((width - cbar_w) / 2 / width, y(top["cbar"] + 0.22), cbar_w / width, 0.22 / height),
                  cbar_label, label_size=cbar_label_pt, tick_size=cbar_tick_pt, ticks=ticks)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    body = BODIES[args.body]
    records = load_responses(run_dir(body, args.model, args.effort) / "responses.jsonl", body, args.model, args.effort)
    points = grid_points(args.res, args.stride)
    pred, answers = collect_predictions(records, points, args.samples)

    dem = load_dem(body.dem)
    lats = np.array([lat for lat, _ in points])
    lons = np.array([lon for _, lon in points])
    truth = elevation_at(dem, lats, lons)

    abs_err = np.abs(pred - truth)
    metrics = {
        "body": body.key,
        "model": args.model,
        "reasoning_effort": args.effort,
        "res_deg": args.res,
        "stride": args.stride,
        "samples_per_pixel": args.samples,
        "n_pixels": len(points),
        "mae_m": float(abs_err.mean()),
        "baseline_zero_mae_m": float(np.abs(truth).mean()),
        "reference_cell_mean_mae_m": float(np.abs(
            cell_means(dem, args.res)[::args.stride, ::args.stride].ravel() - truth).mean()),
    }

    out = run_dir(body, args.model, args.effort) / f"{grid_name(args.res, args.stride)}_n{args.samples}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with open(out / "points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "truth_m", "prediction_m", "abs_error_m", "answers"])
        for (lat, lon), t, p, e, texts in zip(points, truth, pred, abs_err, answers):
            w.writerow([lat, lon, round(float(t), 1), round(float(p), 1), round(float(e), 1), " | ".join(texts)])

    shape = grid_shape(args.res, args.stride)
    model_label = f"{args.model} ({args.effort})"
    pixel_deg = args.res * args.stride
    draw_maps(out / "maps.png", body, dem, truth.reshape(shape), pred.reshape(shape), args.res, args.stride,
              model_label, metrics["mae_m"], metrics["baseline_zero_mae_m"], len(points))
    draw_side_by_side(out / "side_by_side.png", body, overview_map(dem, 0.5), f"{body.real_title} ({body.dem_short})",
                      pred.reshape(shape), pixel_deg, model_label, metrics["mae_m"], len(points))
    draw_side_by_side(out / "side_by_side_same_res.png", body, truth.reshape(shape),
                      f"{body.real_title} ({body.dem_short} at {points_text(args.res, args.stride)})",
                      pred.reshape(shape), pixel_deg, model_label, metrics["mae_m"], len(points))

    draw_stacked(out / "stacked.png",
                 [(None, None, pred.reshape(shape)), (f"Actual map of {body.name}", None, truth.reshape(shape))],
                 f"How {MODEL_NAMES.get(args.model, args.model)} sees {body.name}",
                 f"Asked for elevation at {len(points):,} points.",
                 f"Mean error: {metrics['mae_m']:,.0f} m ({metrics['mae_m'] / METERS_PER_FOOT:,.0f} ft)",
                 vlim=body.vlim, ticks=body.ticks, cbar_label=elevation_cbar_label(body), cmap=body.cmap)

    print(json.dumps(metrics, indent=2))
    print(f"wrote {out}/metrics.json, points.csv, maps.png, side_by_side.png, side_by_side_same_res.png, stacked.png")


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body", choices=BODIES, default="moon", help="which body the answers are about")
    p.add_argument("--res", type=float, default=22.5, help="degrees per pixel (must divide 180)")
    p.add_argument("--stride", type=int, default=1, help="score every STRIDE-th row and column of the grid (see grid.py)")
    p.add_argument("--samples", type=int, default=1, help="answers per pixel to average")
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--effort", default="low", help="reasoning effort")
    args = p.parse_args()
    if args.samples < 1:
        p.error("--samples must be >= 1")
    main(args)


if __name__ == "__main__":
    cli()
