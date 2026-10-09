"""Prompt construction and answer parsing.

The prompt mirrors the Earth post ("If this location is over land, say 'Land'. ...
Do not say anything else. x° S, y° W"), asking for a number instead of Land/Water.
It depends only on (lat, lon), so cached answers stay valid for any grid that
contains the same point. Changing PROMPT_TEMPLATE invalidates cached answers;
query.py refuses to mix prompts within one results file.
"""

import re

PROMPT_TEMPLATE = (
    "Say the elevation of this location on the Moon in meters, "
    "relative to the lunar reference radius of 1737.4 km. "
    "If you are unsure, give your best guess. "
    "Say only the number. Do not say anything else. {location}"
)


def _format_degrees(value: float) -> str:
    text = repr(abs(float(value)))
    return text[:-2] if text.endswith(".0") else text


def format_location(lat: float, lon: float) -> str:
    """E.g. (-11.25, 33.75) -> '11.25° S, 33.75° E'."""
    if not -90 <= lat <= 90:
        raise ValueError(f"latitude {lat} not in [-90, 90]")
    if not -180 <= lon <= 180:
        raise ValueError(f"longitude {lon} not in [-180, 180]")
    lat_text = "0°" if lat == 0 else f"{_format_degrees(lat)}° {'N' if lat > 0 else 'S'}"
    lon_text = "0°" if lon == 0 else f"{_format_degrees(lon)}° {'E' if lon > 0 else 'W'}"
    return f"{lat_text}, {lon_text}"


def build_prompt(lat: float, lon: float) -> str:
    return PROMPT_TEMPLATE.format(location=format_location(lat, lon))


_ANSWER = re.compile(r"^\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?:m|meters|metres)?\s*\.?\s*$")


def parse_elevation(text: str) -> float:
    """Strictly parse a bare number answer. Accepts a unicode minus, thousands commas and a trailing unit."""
    match = _ANSWER.match(text.replace("−", "-"))
    if match is None:
        raise ValueError(f"cannot parse elevation from answer {text!r}")
    return float(match.group(1).replace(",", ""))
