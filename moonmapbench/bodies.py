"""The bodies a model can be asked about: prompt, ground-truth DEM and the labels of the figures.

Each prompt depends only on (lat, lon), so cached answers stay valid for any grid that contains
the same point. Changing a prompt template invalidates the cached answers of that body;
query.py refuses to mix prompts within one results file. The prompts mirror the Earth post
("If this location is over land, say 'Land'. ... Do not say anything else. x° S, y° W"),
asking for a number instead of Land/Water.
"""

from dataclasses import dataclass
from pathlib import Path

from moonmapbench import RESULTS_DIR
from moonmapbench.dem import DemFile

KM_PER_MILE = 1.609344  # for US readers of the shareable images


def _prompt(question: str) -> str:
    return f"{question} If you are unsure, give your best guess. Say only the number. Do not say anything else. {{location}}"


@dataclass(frozen=True)
class Body:
    key: str  # --body value
    name: str  # in a sentence: "How GPT 6.1 Sol sees the Moon"
    title: str  # at the start of a figure title: "The Moon (LOLA, 0.5° pixels)"
    real_title: str  # "The real Moon (LOLA)"
    prompt_template: str
    dem: DemFile
    dem_short: str  # in figure titles
    dem_source: str  # next to the actual map in the shareable images
    datum: str  # colorbar of the detailed figures: "Elevation relative to {datum} (m)"
    datum_long: str  # colorbar of the shareable images: "Elevation in meters, relative to {datum_long}"
    vlim: tuple[int, int]  # color range of every elevation map, the same for every run
    ticks: tuple[int, ...]  # colorbar ticks of the shareable images
    error_lim: int  # color range of the difference maps: +-error_lim
    error_ticks: tuple[int, ...]

    @property
    def results_dir(self) -> Path:
        """results/ for the Moon (the original benchmark), results/<key>/ for the others."""
        return RESULTS_DIR if self.key == "moon" else RESULTS_DIR / self.key


MOON = Body(
    key="moon",
    name="the Moon",
    title="The Moon",
    real_title="The real Moon",
    prompt_template=_prompt("Say the elevation of this location on the Moon in meters, "
                            "relative to the lunar reference radius of 1737.4 km."),
    # LRO LOLA LDEM_16, data set LRO-L-LOLA-4-GDR-V1.0 (from LDEM_16.LBL): 16 pixels/degree (~1.9 km),
    # 2880 lines x 5760 samples, little-endian int16, height = DN * 0.5 m relative to a 1737.4 km sphere.
    dem=DemFile(url="https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_16.IMG",
                sha256="a511e40d7a3ea3275945b4da2a1df377133264fab0be94b7434b1cf8907254cb",
                ppd=16, dtype="<i2", offset=0, scale=0.5),
    dem_short="LOLA",
    dem_source="NASA LRO laser altimeter",
    datum="1737.4 km",
    datum_long=f"a {1737.4:,} km ({1737.4 / KM_PER_MILE:,.1f} mi) radius",
    # About the 0.5th..99.5th area percentile of LOLA heights is -5.7..+6.8 km.
    vlim=(-7000, 7000),
    ticks=(-6000, -3000, 0, 3000, 6000),
    error_lim=5000,  # about the 1st..99th percentile of both models' errors at 2°; beyond it the color saturates
    error_ticks=(-4000, -2000, 0, 2000, 4000),
)

MARS = Body(
    key="mars",
    name="Mars",
    title="Mars",
    real_title="The real Mars",
    prompt_template=_prompt("Say the elevation of this location on Mars in meters, "
                            "relative to the Martian areoid (the standard zero-elevation datum)."),
    # MGS MOLA MEGDR MEGT90N000EB, data set MGS-M-MOLA-5-MEGDR-L3-V1.0 (from MEGT90N000EB.LBL):
    # 16 pixels/degree (~3.7 km), 2880 lines x 5760 samples, big-endian int16 in meters,
    # planetocentric latitude, height = planetary radius minus the radius of the GMM3 areoid.
    dem=DemFile(url="https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x/meg016/megt90n000eb.img",
                sha256="d18d9b9ab8c5516d02e157dd2cde0f1d0d160c21940e953ba22391269a545e7b",
                ppd=16, dtype=">i2", offset=0, scale=1.0),
    dem_short="MOLA",
    dem_source="NASA MGS laser altimeter",
    datum="the areoid",
    datum_long="the Martian areoid",
    # The 0.5th..99.5th area percentile of MOLA heights is -6.6..+7.4 km; the big volcanoes saturate.
    vlim=(-8000, 8000),
    ticks=(-8000, -4000, 0, 4000, 8000),
    error_lim=6000,
    error_ticks=(-6000, -3000, 0, 3000, 6000),
)

MERCURY = Body(
    key="mercury",
    name="Mercury",
    title="Mercury",
    real_title="The real Mercury",
    prompt_template=_prompt("Say the elevation of this location on Mercury in meters, "
                            "relative to Mercury's reference radius of 2439.4 km."),
    # USGS MESSENGER global DEM v2 (Becker et al. 2016), from MDIS stereo images (from its ISIS3 label):
    # 64 pixels/degree (~665 m), 11520 lines x 23040 samples, little-endian int16 starting at byte 92981
    # of the GeoTIFF, height = DN * 0.5 m relative to a 2439.4 km sphere, -32768 = no data.
    dem=DemFile(url="https://planetarymaps.usgs.gov/mosaic/Mercury_Messenger_USGS_DEM_Global_665m_v2.tif",
                sha256="defce776241dcaf44cb0f081ee508c17dbea28aa5a22880bf7a5e8c25f96cbea",
                ppd=64, dtype="<i2", offset=92981, scale=0.5, nodata=-32768),
    dem_short="MESSENGER",
    dem_source="MESSENGER stereo images",
    datum="2439.4 km",
    datum_long=f"a {2439.4:,} km ({2439.4 / KM_PER_MILE:,.1f} mi) radius",
    # The 0.5th..99.5th area percentile of the DEM's heights is -2.7..+2.8 km.
    vlim=(-3000, 3000),
    ticks=(-3000, -1500, 0, 1500, 3000),
    error_lim=3000,
    error_ticks=(-3000, -1500, 0, 1500, 3000),
)

BODIES = {body.key: body for body in (MOON, MARS, MERCURY)}
