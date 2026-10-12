# MoonMapBench

How does a blind model see the Moon? A Moon version of
[How does a blind model see the Earth?](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth).

The Earth post asks a model "Land or Water?" at the centre of every pixel of an
equirectangular lat/lon grid and draws the answers as an image. Here we ask for the
**lunar elevation in meters** instead and compare it to the real Moon.
The same works for [Mars and Mercury](#mars-and-mercury).

## Method

- **Grid:** equirectangular, `res` degrees per pixel, one query at each pixel centre
  (same as the Earth post). Default `res = 22.5°` gives a 16 × 8 image = 128 queries.
  The Earth post mostly used 2° (180 × 90 = 16,200 queries).
  `--stride k` queries only every k-th row and column, starting with the north-west pixel.
  Each point then stands for a k × k block of pixels, but it is still the centre of a `res` pixel.
  `--res 2 --stride 2` gives a 90 × 45 image of 4° blocks (4,050 queries) at latitudes
  89°, 85°, …, −87° and longitudes −179°, −175°, …, 177°. These are 2° grid points, so a later
  run of the whole 2° grid reuses the answers, and runs of the whole 2° grid can be scored on
  exactly these points. The cell-average reference then uses the 2° cell of each point.
- **Prompt** (one user message, no system prompt, no tools):

  > Say the elevation of this location on the Moon in meters, relative to the lunar
  > reference radius of 1737.4 km. If you are unsure, give your best guess. Say only
  > the number. Do not say anything else. 11.25° S, 33.75° W

  Without the "best guess" sentence, gpt-6.1-sol refused ("I don't know.", "Unknown")
  for about half of the points in a pilot. The Earth post forces a Land/Water choice, so
  we force a number in the same way.
  Grok models get a more insistent sentence instead: "If you are unsure, give your best
  guess: an estimate is fine, but always answer with a number and never say that you don't
  know." With the standard sentence, grok-4.7 (low) answered "I don't know the elevation at
  that point." or similar for 34 of 99 points in a pilot. With this one it did so for 1 of 100
  points in the pilot and for 104 of 4,030 in the run below (2.6%, mostly near the north pole).
  As for every model, such points are asked again until the answer is a number.
- **Samples:** 1 answer per pixel by default (`--samples N` averages N answers, as the
  Earth post averaged its n=4 samples per pixel).
- **Ground truth:** LRO LOLA global DEM
  [LDEM_16](https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_16.LBL)
  (data set LRO-L-LOLA-4-GDR-V1.0, NASA Planetary Data System;
  16 pixels/degree, ~1.9 km/pixel, heights relative to the 1737.4 km sphere),
  bilinearly interpolated at each pixel centre.
- **Metric:** mean absolute error (MAE) over all pixels: how far off the estimates
  are on average, in meters. For context we also report
  - the MAE of always answering 0 m, and
  - the MAE of answering each pixel's exact cell-average elevation. The truth is the
    height at one point, so even a model that knew every cell's average would not score 0.

  An equirectangular grid has more pixels per area near the poles, and every pixel counts equally.

## Results

| Model | Effort | Grid | Samples/pixel | MAE | Always 0 m | Exact cell average |
|---|---|---|---|---|---|---|
| gpt-6.1-sol | low | 22.5° (16 × 8) | 1 | **1,514 m** | 1,925 m | 925 m |
| gpt-6.1-sol | low | 7.5° (48 × 24) | 1 | **1,483 m** | 1,754 m | 735 m |
| gpt-6.1-sol | low | 2° (180 × 90) | 1 | **1,496 m** | 1,788 m | 405 m |
| claude-opus-5-5 | low | 2° (180 × 90) | 1 | **1,192 m** | 1,788 m | 405 m |
| grok-4.7 | low | 2°, every 2nd row and column (90 × 45) | 1 | **1,695 m** | 1,781 m | 396 m |

Grok 4.7 was asked only on every 2nd row and column of the 2° grid (`--res 2 --stride 2`,
4,050 points), to save cost: about $19 instead of about $75, as xAI does not take batch requests
for grok-4.7. Its "always 0 m" and "exact cell average" columns are for these 4,050 points.

![gpt-6.1-sol's blind map at 2° above the ground truth at the same pixel centres](results/gpt-6.1-sol__low/res2_n1/stacked.png)

The ground truth is LOLA at the 2° pixel centres, the values the model is scored against.
Side by side: [with full-detail LOLA](results/gpt-6.1-sol__low/res2_n1/side_by_side.png),
[with LOLA at the same resolution](results/gpt-6.1-sol__low/res2_n1/side_by_side_same_res.png).

At 2° (the Earth post's usual resolution) the model's map shows the nearside maria in
their real outline and individual basins in roughly the right places: Orientale as a round
depression near 95°W 20°S, Crisium, Smythii, Humboldtianum and the South Pole–Aitken basin.
Near the poles it often repeats one value along a row, which shows up as horizontal streaks.
It underestimates the farside highlands (bright from 150°W to 180°W near the equator) and has
little of the crater-scale roughness of the real map.

The MAE stays at about 1.5 km at every resolution, while the exact-cell-average reference
drops from 925 m to 405 m. So the error comes from what the model does not know, not
from the coarse grid.

Claude Opus 5.5 at 2°: [with LOLA at the same resolution](results/claude-opus-5-5__low/res2_n1/side_by_side_same_res.png),
[stacked](results/claude-opus-5-5__low/res2_n1/stacked.png). Its answers came through the
Message Batches API (all but 220 test answers), which uses the same requests at half price.

Grok 4.7 next to the other two models: [maps](results/compare_res2_n1_grok.png),
[answer minus ground truth](results/difference_res2_n1_grok.png); its own figures are in
[`grok-4.7__low/res2_stride2_n1/`](results/grok-4.7__low/res2_stride2_n1/side_by_side_same_res.png).
Its map shows the nearside maria in rough outline and the South Pole–Aitken basin, but it
places the farside highlands far too low, and many single pixels are off by several km.
With an MAE of 1,695 m it is only a little better than always answering 0 m (1,781 m).
On the same 4,050 points, claude-opus-5-5 scores 1,186 m and gpt-6.1-sol 1,476 m, close to their
scores on the whole 2° grid, so the smaller grid does not favor or penalize Grok much.

Other figures: [`res2_n1/maps.png`](results/gpt-6.1-sol__low/res2_n1/maps.png) (with LOLA
sampled at the same points), [7.5°](results/gpt-6.1-sol__low/res7.5_n1/side_by_side.png),
[22.5°](results/gpt-6.1-sol__low/res22.5_n1/side_by_side.png).
The 7.5° grid contains every 22.5° pixel centre, so it reused those 128 answers.

## Mars and Mercury

`--body mars` and `--body mercury` ask about Mars or Mercury instead. The grid, samples and
metric are the same as for the Moon. The prompt differs only in the body and its zero level:

> Say the elevation of this location on Mars in meters, relative to the Martian areoid
> (the standard zero-elevation datum). If you are unsure, give your best guess. Say only
> the number. Do not say anything else. 11.25° S, 33.75° W

> Say the elevation of this location on Mercury in meters, relative to Mercury's reference
> radius of 2439.4 km. If you are unsure, give your best guess. Say only the number.
> Do not say anything else. 11.25° S, 33.75° W

Ground truth, bilinearly interpolated at each pixel centre as for the Moon:

- **Mars:** MGS MOLA gridded topography
  [MEGT90N000EB](https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/megt90n000eb.lbl)
  (data set MGS-M-MOLA-5-MEGDR-L3-V1.0, NASA Planetary Data System; 16 pixels/degree,
  ~3.7 km/pixel), heights above the areoid. Mars is flattened at the poles, so heights relative
  to a sphere would mostly show that flattening. The elevations that are usually quoted
  (Olympus Mons about +21 km, the floor of Hellas about −8 km) are heights above the areoid.
  Latitudes are planetocentric, as in all MOLA products.
- **Mercury:** USGS MESSENGER global DEM v2
  ([Becker et al. 2016](https://www.hou.usra.edu/meetings/lpsc2016/pdf/2959.pdf),
  [product page](https://astrogeology.usgs.gov/search/map/mercury_messenger_global_dem_665m)),
  made from MESSENGER camera stereo images; 64 pixels/degree, ~665 m/pixel, heights relative to
  a 2439.4 km sphere. MESSENGER's laser altimeter covered mostly the northern hemisphere,
  but this stereo DEM covers the whole planet.

Answers and figures go to `results/<body>/<model>__<effort>/`, comparison images to
`results/<body>/`. The Moon keeps `results/` as before.

## Running

Needs `uv` and an API key, either in the environment or in a `.env` file in the
repository root (the environment wins):

- OpenAI models: `OPENAI_API_KEY`. Queries go to the
  [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) with
  `reasoning_effort`, through the official `openai` Python package.
- Claude models (`claude-*`): `ANTHROPIC_API_KEY`. Queries go to the
  [Messages API](https://platform.claude.com/docs/en/api/messages) with adaptive thinking
  and `output_config.effort`, through the official `anthropic` Python package. A refusal is
  not saved and stops the run; no other model answers in its place.
- Grok models (`grok-*`): `XAI_API_KEY`. Queries go to xAI's OpenAI-compatible
  [Chat Completions API](https://docs.x.ai/developers/rest-api-reference/inference/chat) with
  `reasoning_effort`, through the `openai` package. xAI reports the cost of each answer,
  and the cost script adds these up. xAI counts about 1,300 input tokens for each request
  (mostly cached), much more than the prompt: it adds text of its own.

```
# .env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
XAI_API_KEY=...
```

```bash
scripts/run.sh                                   # gpt-6.1-sol, low effort, 22.5°/px, 1 sample
scripts/run.sh --res 10 --samples 4 --effort medium
scripts/run.sh --res 7.5 --concurrency 64                 # fewer parallel requests
scripts/run.sh --model claude-opus-5-5 --effort low       # Claude Opus 5.5
scripts/run.sh --model claude-opus-5-5 --points 20        # 20 queries only, as a cheap test
scripts/run.sh --model grok-4.7 --effort low --res 2 --points 100   # grok-4.7, 100 points of the 2° grid
scripts/run.sh --batch --max-cost 100 --model claude-opus-5-5 --res 2   # Batches API, half price
scripts/run.sh --body mars --res 2                        # Mars instead of the Moon (also: --body mercury)
scripts/run.sh --model grok-4.7 --res 2 --stride 2        # every 2nd row and column of the 2° grid
```

`--points N` queries only N grid points, picked at random with a fixed seed, and skips the
report, which needs the whole grid. At the end the script prints the tokens of all saved
answers of that model and effort, and their cost: at list price for models in
`PRICES_PER_MTOK` in `moonmapbench/cost.py`, as reported by xAI for Grok models. This is a
lower bound on what was spent: requests that failed, were refused or were cancelled are billed
too, but not saved.

`--batch` (Claude models only) sends all missing points as one request to the
[Message Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing),
which costs half the normal price, and waits for the results (usually within an hour, at
most 24 h). The requests are the same as without `--batch`. Before it sends anything, it
estimates the cost of the batch from the mean tokens per answer already saved for that model
and effort. It sends nothing if that estimate plus the cost of all saved answers of that body,
model and effort is above `--max-cost`. So query a few points first, e.g. with `--points 20`.
Each submission is logged in `batches.jsonl` next to the answers. A rerun first waits
for and collects every batch that is not collected yet, so an interrupted run loses nothing
and no point that is still in a batch is sent again; queries without `--batch` refuse to run
until then.

With or without `--batch`, answers that are not a plain number (such as "I don't know.") are
not saved; the run saves the rest and then stops with a list of them, and a rerun asks those
points again. A rerun asks again until the model gives a number, so the saved answer is the
first number the model gave. With `--batch`, all other unusable answers and errors are handled
in the same way. Without `--batch`, they stop the run at once, as they would likely hit every
request (e.g. another model answering).

Requests run in parallel, up to `--concurrency` (default 256) at a time. When the API
answers "Too Many Requests" (HTTP 429), the number of requests in flight drops by 20% and
slowly grows back. Retries wait a random, growing time. A run stops when the account is
out of quota.

The script queries the model, writes the report and prints the cost.
The Python environment lives in `~/.local/share/venvs/moonmapbench`.
Every answer is appended to `<results>/<model>__<effort>/responses.jsonl` as soon as it
arrives, where `<results>` is `results/` for the Moon and `results/<body>/` for Mars and Mercury. A rerun only queries missing (lat, lon, sample) entries, so an interrupted run
resumes, and a later grid reuses any pixel centres it shares with an earlier one.
The ground-truth DEM is downloaded to `data/` on first use and checked by sha256: LOLA for the
Moon (33 MB), MOLA for Mars (33 MB), the MESSENGER DEM for Mercury (531 MB; the report needs
about 2 GB of memory for it).

Outputs per run in `<results>/<model>__<effort>/res<res>_n<samples>/`
(`res<res>_stride<k>_n<samples>/` with `--stride k > 1`):
`metrics.json`, `points.csv` (truth, prediction and raw answers per pixel), `maps.png`,
`side_by_side.png` (the real body next to the model's map), `side_by_side_same_res.png`
(the same, with the real Moon at the grid's resolution) and `stacked.png` (the model's map
above the ground truth at the grid's resolution).

To put several models above the ground truth in one image, for sharing:
`uv run python -m moonmapbench.compare --res 2 --runs claude-opus-5-5:low gpt-6.1-sol:low`
writes `results/compare_res2_n1.png` (needs `UV_PROJECT_ENVIRONMENT` set as in `scripts/run.sh`;
add `--body mars` or `--body mercury` for the other bodies). A run given as `MODEL:EFFORT:k` is shown
and scored on every k-th row and column, e.g. `grok-4.7:low:2`; `--output` writes to another file.
With `--difference` it maps each model's answer minus the ground truth instead (red: too high,
blue: too low) to `results/difference_res2_n1.png`.

## License

Code, figures and results are under the MIT license (see `LICENSE`). The ground-truth
heights are in the public domain: the Moon and Mars from NASA's Planetary Data System,
Mercury from the USGS. If you use them, please cite the LOLA data set LRO-L-LOLA-4-GDR-V1.0,
the MOLA data set MGS-M-MOLA-5-MEGDR-L3-V1.0, and for Mercury Becker et al. (2016),
"First global digital elevation model of Mercury", 47th Lunar and Planetary Science Conference.
