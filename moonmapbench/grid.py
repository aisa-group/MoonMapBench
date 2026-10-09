"""Equirectangular pixel grid, as in "How does a blind model see the Earth?".

Each pixel is `res` x `res` degrees and is queried at its centre.
E.g. res=22.5 gives a 16 x 8 image (128 points), res=2 gives 180 x 90 (16200 points).
"""

import math

import numpy as np


def grid_shape(res: float) -> tuple[int, int]:
    """(rows, cols) of the image. `res` must divide 180 degrees evenly."""
    if not (math.isfinite(res) and res > 0):
        raise ValueError(f"resolution must be a positive finite number, got {res}")
    rows = round(180 / res)
    if abs(rows * res - 180) > 1e-9:
        raise ValueError(f"resolution {res} does not divide 180 degrees evenly")
    return rows, 2 * rows


def grid_axes(res: float) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre (lats, lons). lats run north -> south (image row order), lons west -> east."""
    rows, cols = grid_shape(res)
    lats = np.array([round(90 - (j + 0.5) * res, 6) for j in range(rows)])
    lons = np.array([round(-180 + (i + 0.5) * res, 6) for i in range(cols)])
    return lats, lons


def grid_points(res: float) -> list[tuple[float, float]]:
    """All (lat, lon) pixel centres, row-major in image order."""
    lats, lons = grid_axes(res)
    return [(float(lat), float(lon)) for lat in lats for lon in lons]
