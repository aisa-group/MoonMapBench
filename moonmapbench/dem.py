"""Global elevation models (DEMs) in simple cylindrical layout, and sampling them on the query grid.

Every DEM here is pixel registered: `ppd` pixels per degree, line 0 starts at 90°N, sample 0 at
0°E, longitude positive east. Heights are in meters relative to the body's reference surface.
Each file is downloaded to data/ on first use and checked by sha256.
"""

import hashlib
import shutil
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import certifi
import numpy as np

from moonmapbench import DATA_DIR
from moonmapbench.grid import grid_shape


@dataclass(frozen=True)
class DemFile:
    url: str
    sha256: str
    ppd: int  # pixels per degree
    dtype: str  # numpy dtype of the raw samples, e.g. "<i2"
    offset: int  # bytes before the first sample; the samples run to the end of the file
    scale: float  # meters per raw unit
    nodata: int | None = None  # raw value of a pixel without data; loading raises if there is one

    @property
    def path(self) -> Path:
        return DATA_DIR / self.url.rsplit("/", 1)[1]


@dataclass(frozen=True)
class Dem:
    heights: np.ndarray  # meters, float32, shape (180 * ppd, 360 * ppd), rows north -> south, columns 0°E -> 360°E
    ppd: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_file(spec: DemFile) -> Path:
    path = spec.path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {spec.url} -> {path}")
        tmp = path.with_suffix(path.suffix + ".part")
        # certifi's CA list: some servers (PDS Geosciences) chain to roots that older systems lack.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(spec.url, timeout=60, context=context) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        digest = _sha256(tmp)
        if digest != spec.sha256:
            raise RuntimeError(f"downloaded {tmp} has sha256 {digest}, expected {spec.sha256}")
        tmp.rename(path)
    digest = _sha256(path)
    if digest != spec.sha256:
        raise RuntimeError(f"{path} has sha256 {digest}, expected {spec.sha256}. Delete it and rerun.")
    return path


def load_dem(spec: DemFile) -> Dem:
    path = ensure_file(spec)
    lines, samples = 180 * spec.ppd, 360 * spec.ppd
    itemsize = np.dtype(spec.dtype).itemsize
    if path.stat().st_size != spec.offset + lines * samples * itemsize:
        raise RuntimeError(f"{path} has {path.stat().st_size} bytes, expected {spec.offset} + "
                           f"{lines} x {samples} samples of {itemsize} bytes")
    raw = np.fromfile(path, dtype=spec.dtype, offset=spec.offset).reshape(lines, samples)
    if spec.nodata is not None and np.any(raw == spec.nodata):
        raise RuntimeError(f"{path} has {np.count_nonzero(raw == spec.nodata)} pixels without data")
    return Dem(np.multiply(raw, spec.scale, dtype=np.float32), spec.ppd)


def elevation_at(dem: Dem, lat, lon) -> np.ndarray:
    """Bilinear interpolation of the DEM at (lat, lon) in degrees, lon in any convention."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        raise ValueError("non-finite latitude or longitude")
    if np.any(np.abs(lat) > 90):
        raise ValueError("latitude out of [-90, 90]")
    lines, samples = dem.heights.shape
    # Continuous pixel coordinates where integer values are pixel centres.
    x = np.mod(lon, 360.0) * dem.ppd - 0.5
    y = (90.0 - lat) * dem.ppd - 0.5
    # Within half a pixel of a pole there is no row beyond: clamp to the edge row.
    y = np.clip(y, 0.0, lines - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.minimum(np.floor(y).astype(np.int64), lines - 2)
    fx = x - x0
    fy = y - y0
    x0 = np.mod(x0, samples)
    x1 = np.mod(x0 + 1, samples)
    y1 = y0 + 1
    h = dem.heights
    top = h[y0, x0] * (1 - fx) + h[y0, x1] * fx
    bottom = h[y1, x0] * (1 - fx) + h[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def overview_map(dem: Dem, deg: float) -> np.ndarray:
    """Block-mean downsampled DEM with `deg`-degree pixels, columns re-ordered to run from -180°E to 180°E."""
    factor = deg * dem.ppd
    if abs(factor - round(factor)) > 1e-9:
        raise ValueError(f"{deg}° is not a whole number of DEM pixels")
    factor = round(factor)
    lines, samples = dem.heights.shape
    if lines % factor or samples % factor:
        raise ValueError(f"{deg}° pixels do not divide the DEM shape")
    small = dem.heights.reshape(lines // factor, factor, samples // factor, factor).mean(axis=(1, 3))
    return np.roll(small, small.shape[1] // 2, axis=1)


def cell_means(dem: Dem, res: float) -> np.ndarray:
    """Area-weighted mean height of each `res` x `res` grid cell, in grid image order.

    Cell edges must fall on DEM pixel edges (res a multiple of 1/ppd degree).
    """
    rows, cols = grid_shape(res)
    k = res * dem.ppd
    if abs(k - round(k)) > 1e-9:
        raise ValueError(f"resolution {res} is not a multiple of 1/{dem.ppd} degree")
    k = round(k)
    lines, samples = dem.heights.shape
    weights = np.cos(np.radians(90.0 - (np.arange(lines) + 0.5) / dem.ppd))
    means = np.empty((rows, cols))
    for j in range(rows):  # one row of cells at a time, so that a 64 ppd DEM fits in memory
        band = np.roll(dem.heights[j * k:(j + 1) * k], samples // 2, axis=1)  # columns from -180°E, like the grid
        w = weights[j * k:(j + 1) * k]
        means[j] = (band * w[:, None]).reshape(k, cols, k).sum(axis=(0, 2)) / (w.sum() * k)
    return means
