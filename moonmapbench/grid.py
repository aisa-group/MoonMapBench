"""Equirectangular pixel grid, as in "How does a blind model see the Earth?".

Each pixel is `res` x `res` degrees and is queried at its centre.
E.g. res=22.5 gives a 16 x 8 image (128 points), res=2 gives 180 x 90 (16200 points).

With stride k > 1 only every k-th row and column is queried, starting with the north-west pixel,
and each point stands for the k x k block of pixels whose north-west pixel it is the centre of.
E.g. res=2, stride=2 gives a 90 x 45 image of 4° blocks (4050 points), queried at latitudes
89, 85, ..., -87 and longitudes -179, -175, ..., 177. These are points of the 2° grid, so a later
run of the whole 2° grid reuses their answers, and runs of the whole 2° grid can be scored on them.
"""

import math

import numpy as np


def grid_shape(res: float, stride: int) -> tuple[int, int]:
    """(rows, cols) of the image. `res` must divide 180 degrees evenly, `stride` the number of rows."""
    if not (math.isfinite(res) and res > 0):
        raise ValueError(f"resolution must be a positive finite number, got {res}")
    rows = round(180 / res)
    if abs(rows * res - 180) > 1e-9:
        raise ValueError(f"resolution {res} does not divide 180 degrees evenly")
    if type(stride) is not int or stride < 1:
        raise ValueError(f"stride must be a positive int, got {stride!r}")
    if rows % stride:
        raise ValueError(f"stride {stride} does not divide the {rows} rows of the {res}° grid")
    return rows // stride, 2 * rows // stride


def grid_axes(res: float, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Queried pixel-centre (lats, lons). lats run north -> south (image row order), lons west -> east."""
    rows, cols = grid_shape(res, 1)
    grid_shape(res, stride)  # checks the stride
    lats = np.array([round(90 - (j + 0.5) * res, 6) for j in range(0, rows, stride)])
    lons = np.array([round(-180 + (i + 0.5) * res, 6) for i in range(0, cols, stride)])
    return lats, lons


def grid_points(res: float, stride: int) -> list[tuple[float, float]]:
    """All queried (lat, lon) pixel centres, row-major in image order."""
    lats, lons = grid_axes(res, stride)
    return [(float(lat), float(lon)) for lat in lats for lon in lons]


def grid_name(res: float, stride: int) -> str:
    """Part of output names: 'res2' for the whole 2° grid, 'res2_stride2' for every 2nd row and column."""
    return f"res{res:g}" if stride == 1 else f"res{res:g}_stride{stride}"
