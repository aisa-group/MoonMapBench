"""Ground truth: LRO LOLA global DEM LDEM_16 (16 pixels/degree, ~1.9 km/pixel).

Heights are in meters relative to a 1737.4 km reference sphere.
Layout (from LDEM_16.LBL): 2880 lines x 5760 samples, little-endian int16,
height = DN * 0.5 m, simple cylindrical, pixel registered,
line 0 starts at 90°N, sample 0 starts at 0°E, longitude positive east.
"""

import hashlib
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from moonmapbench import DATA_DIR
from moonmapbench.grid import grid_shape

LDEM_URL = "https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_16.IMG"
LDEM_SHA256 = "a511e40d7a3ea3275945b4da2a1df377133264fab0be94b7434b1cf8907254cb"
LDEM_PATH = DATA_DIR / "LDEM_16.IMG"
PPD = 16
LINES = 180 * PPD
SAMPLES = 360 * PPD
SCALING_FACTOR = 0.5


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_ldem() -> Path:
    if not LDEM_PATH.exists():
        LDEM_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {LDEM_URL} -> {LDEM_PATH}")
        tmp = LDEM_PATH.with_suffix(".part")
        with urllib.request.urlopen(LDEM_URL, timeout=60) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        digest = _sha256(tmp)
        if digest != LDEM_SHA256:
            raise RuntimeError(f"downloaded {tmp} has sha256 {digest}, expected {LDEM_SHA256}")
        tmp.rename(LDEM_PATH)
    digest = _sha256(LDEM_PATH)
    if digest != LDEM_SHA256:
        raise RuntimeError(f"{LDEM_PATH} has sha256 {digest}, expected {LDEM_SHA256}. Delete it and rerun.")
    return LDEM_PATH


def load_ldem() -> np.ndarray:
    """Heights in meters, shape (LINES, SAMPLES), rows north -> south, columns 0°E -> 360°E."""
    raw = np.fromfile(ensure_ldem(), dtype="<i2")
    if raw.size != LINES * SAMPLES:
        raise RuntimeError(f"LDEM_16 has {raw.size} samples, expected {LINES * SAMPLES}")
    return raw.reshape(LINES, SAMPLES).astype(np.float32) * SCALING_FACTOR


def elevation_at(dem: np.ndarray, lat, lon) -> np.ndarray:
    """Bilinear interpolation of the DEM at (lat, lon) in degrees, lon in any convention."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        raise ValueError("non-finite latitude or longitude")
    if np.any(np.abs(lat) > 90):
        raise ValueError("latitude out of [-90, 90]")
    # Continuous pixel coordinates where integer values are pixel centres.
    x = np.mod(lon, 360.0) * PPD - 0.5
    y = (90.0 - lat) * PPD - 0.5
    # Within half a pixel of a pole there is no row beyond: clamp to the edge row.
    y = np.clip(y, 0.0, LINES - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.minimum(np.floor(y).astype(np.int64), LINES - 2)
    fx = x - x0
    fy = y - y0
    x0 = np.mod(x0, SAMPLES)
    x1 = np.mod(x0 + 1, SAMPLES)
    y1 = y0 + 1
    top = dem[y0, x0] * (1 - fx) + dem[y0, x1] * fx
    bottom = dem[y1, x0] * (1 - fx) + dem[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def overview_map(dem: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsampled DEM with columns re-ordered to run from -180°E to 180°E."""
    if LINES % factor or SAMPLES % factor:
        raise ValueError(f"factor {factor} does not divide the DEM shape")
    small = dem.reshape(LINES // factor, factor, SAMPLES // factor, factor).mean(axis=(1, 3))
    return np.roll(small, small.shape[1] // 2, axis=1)


def cell_means(dem: np.ndarray, res: float) -> np.ndarray:
    """Area-weighted mean height of each `res` x `res` grid cell, in grid image order.

    Cell edges must fall on DEM pixel edges (res a multiple of 1/16 degree).
    """
    rows, cols = grid_shape(res)
    k = res * PPD
    if abs(k - round(k)) > 1e-9:
        raise ValueError(f"resolution {res} is not a multiple of 1/{PPD} degree")
    k = round(k)
    weights = np.cos(np.radians(90.0 - (np.arange(LINES) + 0.5) / PPD))
    rolled = np.roll(dem, SAMPLES // 2, axis=1)  # columns from -180°E, like the grid
    sums = (rolled * weights[:, None]).reshape(rows, k, cols, k).sum(axis=(1, 3))
    return sums / (weights.reshape(rows, k).sum(axis=1) * k)[:, None]
