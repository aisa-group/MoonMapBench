"""Prompt construction and answer parsing. The prompt templates are in bodies.py.

Grok models get a more insistent best-guess sentence: with the standard one, grok-4.7 (low) answered
"I don't know the elevation at that point." for about a third of the points.
"""

import re

from moonmapbench.bodies import Body


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


BEST_GUESS = "If you are unsure, give your best guess."
EXPLICIT_BEST_GUESS = ("If you are unsure, give your best guess: an estimate is fine, but always answer with a number "
                       "and never say that you don't know.")


def build_prompt(body: Body, model: str, lat: float, lon: float) -> str:
    """The question to `model` about (lat, lon) on `body`."""
    template = body.prompt_template
    if model.startswith("grok-"):
        if template.count(BEST_GUESS) != 1:
            raise ValueError(f"expected the best-guess sentence once in the {body.key} prompt: {template!r}")
        template = template.replace(BEST_GUESS, EXPLICIT_BEST_GUESS)
    return template.format(location=format_location(lat, lon))


_ANSWER = re.compile(r"^\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(?:m|meters|metres)?\s*\.?\s*$")


def parse_elevation(text: str) -> float:
    """Strictly parse a bare number answer. Accepts a unicode minus, thousands commas and a trailing unit."""
    match = _ANSWER.match(text.replace("−", "-"))
    if match is None:
        raise ValueError(f"cannot parse elevation from answer {text!r}")
    return float(match.group(1).replace(",", ""))
