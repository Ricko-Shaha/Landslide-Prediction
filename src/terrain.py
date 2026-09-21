"""Terrain factors at a coordinate, without any credentials.

WHY THIS EXISTS. Live coordinate lookup is the point of the project, and the original build made
it unreachable: `ee.Initialize()` ran at import, so the whole application refused to start without
an authenticated Google Earth Engine account. That is a hard dependency on a private credential
for the one feature a reader most wants to try.

SRTM 30m, the same elevation model the thesis used (`USGS/SRTMGL1_003`), is served without
authentication by OpenTopoData. Elevation is all that is needed for three of the eight factors,
and those three carry most of the model's weight:

    SLOPE 0.716 + ELEVATION 0.070 + ASPECT 0.033 = 0.819 of total feature importance

Slope and aspect are computed from a 3x3 elevation window by Horn's method, which is what GDAL and
Earth Engine both use, so the numbers are comparable to the thesis pipeline rather than a different
approximation.

The remaining factors are handled honestly rather than invented:
    NDVI, LANDUSE   Earth Engine if authenticated, otherwise the study-area median, flagged
    RAINFALL        a stated input, defaulting to the thesis value
    TWI, SPI        need catchment flow accumulation, which a point query cannot give; derived
                    from slope with the flow term held at the study-area median, and flagged
"""
from __future__ import annotations

import json
import math
import ssl
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "landslide-thesis/1.0 (research)"}
_SSL = ssl.create_default_context()

OPENTOPO = "https://api.opentopodata.org/v1/srtm30m"


def _get(url: str, tries: int = 3):
    last = None
    for a in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=30, context=_SSL) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except Exception as exc:                       # network, rate limit, upstream outage
            last = exc
            time.sleep(1.2 * (a + 1))
    raise RuntimeError("elevation service unreachable: %s: %s"
                       % (type(last).__name__, str(last)[:120]))


def elevation_grid(lat: float, lon: float, step_m: float = 90.0):
    """A 3x3 window of elevations centred on the point, in one request.

    `step_m` is the spacing between samples. 90 m is three SRTM cells, wide enough that the slope
    estimate is not dominated by the DEM's own vertical noise, which is roughly +/- 5 m.
    """
    dlat = step_m / 111_320.0
    dlon = step_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    pts = []
    for iy in (1, 0, -1):              # north row first, matching Horn's convention
        for ix in (-1, 0, 1):
            pts.append((lat + iy * dlat, lon + ix * dlon))
    locs = "|".join("%.6f,%.6f" % p for p in pts)
    d = _get(OPENTOPO + "?locations=" + urllib.parse.quote(locs))
    if d.get("status") != "OK":
        raise RuntimeError("elevation service returned %s" % d.get("status"))
    z = []
    for r in d["results"]:
        e = r.get("elevation")
        z.append(0.0 if e is None else float(e))       # sea returns null, which is 0 m
    if len(z) != 9:
        raise RuntimeError("expected 9 elevation samples, got %d" % len(z))
    return z, step_m


def slope_aspect(z, step_m: float):
    """Horn's method, the standard used by GDAL and Earth Engine.

    z is the 3x3 window in row-major order starting at the north-west corner:
        z0 z1 z2      a b c
        z3 z4 z5  ->  d e f
        z6 z7 z8      g h i
    """
    a, b, c, d, e, f, g, h, i = z
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * step_m)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * step_m)
    slope_deg = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
    aspect = math.degrees(math.atan2(dzdy, -dzdx))
    if aspect < 0:
        aspect += 360.0
    return slope_deg, aspect, e


def sample_point(lat: float, lon: float, step_m: float = 90.0):
    """Raw physical terrain factors at a coordinate. No credentials required."""
    z, step = elevation_grid(lat, lon, step_m)
    slope_deg, aspect_deg, elev = slope_aspect(z, step)
    return {
        "ELEVATION_m": float(elev),
        "SLOPE_deg": float(slope_deg),
        "ASPECT_deg": float(aspect_deg),
        "source": "SRTM 30m via OpenTopoData (no authentication required)",
        "window_m": step,
        "neighbourhood_m": [float(v) for v in z],
    }
