"""Rainfall at a coordinate, from a coarse cached grid.

WHY THIS EXISTS. Rainfall was the one factor the original application hardcoded: `rainfall = 28`
sat as a literal in the request handler, so every coordinate in Rangamati was told it received
identical rain. For landslides that is the wrong constant to pick, because monsoon rainfall is the
usual trigger. The restructured version at least made it an explicit input, but it was still held
at one class for the whole district, which is the same thing with better manners.

WHY A COARSE GRID AND NOT A LOOKUP PER POINT. Slope changes completely over 90 m; rainfall does
not. Annual precipitation is a smooth regional field, so sampling it every 13 km and interpolating
loses almost nothing, while a request per map cell would mean thousands of calls to a free service
for values that barely differ. The grid is fetched once and cached.

    elevation   fetched per point       varies on the scale of the hillslope
    rainfall    fetched on a 0.12 deg grid, interpolated    varies on the scale of the district

SOURCE. NASA POWER's climatology endpoint, which needs no account and returns a long-term monthly
mean in one request per point. Its agroclimatology community is the right product here: it is
built for exactly this question, and it reports the multi-year normal rather than one year's
weather, which is what a susceptibility factor should be keyed to.

Open-Meteo's ERA5 archive was the first choice and is kept below as a fallback, but it needs five
requests per point (one per year) and was returning 500s and timeouts when this was built. NASA
POWER answers the same question in one request and about a second.

HONESTY. If the cache is absent or the service is unreachable, `rainfall_class` returns None and
the caller falls back to the stated constant and flags the factor as held. Nothing is invented.
"""
from __future__ import annotations

import json
import math
import ssl
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "rainfall_grid.json"

POWER = "https://power.larc.nasa.gov/api/temporal/climatology/point"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"      # fallback, see the module docstring
UA = {"User-Agent": "landslide-thesis/1.0 (research)"}
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

#: enough years to average out a single freak monsoon, few enough to fetch in one sitting
YEARS = (2018, 2019, 2020, 2021, 2022)

#: the six RAINFALL class ratings, ascending, from the training inventory
RATINGS = [0, 5, 10, 21, 29, 35]

#: Class boundaries in mm of mean annual precipitation, EQUAL INTERVAL over the range actually
#: observed across the district. They are computed when the cache is built and stored with it, so
#: they are visible rather than hidden in code; the list below is only the fallback.
#:
#: Equal interval, not quantiles, and for the reason recorded in src/reclassify.py: the 392-row
#: inventory is balanced 196/196 by design and is therefore not a random sample of Rangamati, so
#: matching quantiles between the two distributions misassigns the classes. Equal-interval
#: reclassification is a standard GIS operation and does not assume the two populations match.
CUTS_MM = [2600.0, 2900.0, 3200.0, 3500.0, 3800.0]


def cuts():
    """The stored cut points if the cache has them, otherwise the fallback."""
    g = grid()
    if g and g.get("cuts_mm"):
        return [float(v) for v in g["cuts_mm"]]
    return CUTS_MM

_GRID = None


def _get(url, tries=3, timeout=90):
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except Exception as exc:
            last = exc
            time.sleep(2.0 * (a + 1))
    raise RuntimeError("%s: %s" % (type(last).__name__, str(last)[:120]))


def annual_mm_live(lat: float, lon: float) -> float:
    """Mean annual precipitation at a point, from NASA POWER's climatology.

    POWER reports PRECTOTCORR as a mean rate in mm/day, including an annual figure under `ANN`,
    so the yearly total is that rate over a year. Falls back to summing the twelve monthly means
    if `ANN` is missing.
    """
    d = _get("%s?parameters=PRECTOTCORR&community=AG&longitude=%.4f&latitude=%.4f&format=JSON"
             % (POWER, lon, lat))
    p = d["properties"]["parameter"]["PRECTOTCORR"]
    if "ANN" in p and p["ANN"] is not None and float(p["ANN"]) > -900:
        return float(p["ANN"]) * 365.25
    days = {"JAN": 31, "FEB": 28.25, "MAR": 31, "APR": 30, "MAY": 31, "JUN": 30,
            "JUL": 31, "AUG": 31, "SEP": 30, "OCT": 31, "NOV": 30, "DEC": 31}
    vals = [(p[m], n) for m, n in days.items() if m in p and float(p[m]) > -900]
    if not vals:
        raise RuntimeError("no precipitation returned for %.4f,%.4f" % (lat, lon))
    return float(sum(v * n for v, n in vals))


def annual_mm_openmeteo(lat: float, lon: float) -> float:
    """Fallback source: Open-Meteo ERA5, one request per year. See the module docstring."""
    total = 0.0
    for y in YEARS:
        d = _get("%s?latitude=%.4f&longitude=%.4f&start_date=%d-01-01&end_date=%d-12-31"
                 "&daily=precipitation_sum&timezone=UTC" % (ARCHIVE, lat, lon, y, y))
        vals = [v for v in d["daily"]["precipitation_sum"] if v is not None]
        total += sum(vals)
    return total / float(len(YEARS))


def grid():
    global _GRID
    if _GRID is None:
        if not CACHE.exists():
            return None
        _GRID = json.loads(CACHE.read_text(encoding="utf-8"))
    return _GRID


def annual_mm(lat: float, lon: float):
    """Interpolated mean annual precipitation, or None when the cache is not built.

    Inverse distance over the three nearest grid points. The field is smooth at this spacing, so
    this is a genuine interpolation rather than a guess dressed as one.
    """
    g = grid()
    if not g or not g.get("points"):
        return None
    pts = g["points"]
    scored = []
    for p in pts:
        dy = (p["lat"] - lat) * 111.32
        dx = (p["lon"] - lon) * 111.32 * math.cos(math.radians(lat))
        scored.append((math.hypot(dx, dy), p["mm"]))
    scored.sort(key=lambda t: t[0])
    near = scored[:3]
    if near[0][0] < 0.5:
        return float(near[0][1])
    wsum = sum(1.0 / d for d, _ in near)
    return float(sum((1.0 / d) * mm for d, mm in near) / wsum)


def rainfall_class(lat: float, lon: float):
    """The class rating for this coordinate, or None if rainfall cannot be measured here."""
    mm = annual_mm(lat, lon)
    if mm is None:
        return None
    cp = cuts()
    i = 0
    while i < len(cp) and mm >= cp[i]:
        i += 1
    return int(RATINGS[min(i, len(RATINGS) - 1)]), float(mm)


# --------------------------------------------------------------------------------------------
# cache builder

def build(step: float = 0.10, pause: float = 0.35):
    import study_area

    la0, lo0, la1, lo1 = study_area.bbox()
    targets = []
    lat = la0 - step
    while lat <= la1 + step:
        lon = lo0 - step
        while lon <= lo1 + step:
            if study_area.contains(lat, lon) or any(
                    study_area.contains(lat + dy * step, lon + dx * step)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                targets.append((round(lat, 4), round(lon, 4)))
            lon += step
        lat += step

    print("rainfall grid at %.3f deg (~%.0f km): %d points, 1 request each, about %.0f s"
          % (step, step * 111.32, len(targets), len(targets) * (pause + 1.1)))

    pts = []
    for i, (lat, lon) in enumerate(targets, 1):
        try:
            mm = annual_mm_live(lat, lon)
        except Exception as exc:
            print("  %d/%d skipped %.3f,%.3f: %s" % (i, len(targets), lat, lon, str(exc)[:70]))
            continue
        pts.append({"lat": lat, "lon": lon, "mm": round(mm, 1)})
        if i % 5 == 0 or i == len(targets):
            print("  %d/%d  last %.0f mm" % (i, len(targets), mm))
        time.sleep(pause)

    if not pts:
        raise RuntimeError("no rainfall points fetched")
    mms = sorted(p["mm"] for p in pts)
    # equal-interval cut points across the district's observed range, five of them for six classes
    lo, hi = mms[0], mms[-1]
    width = (hi - lo) / 6.0
    cut_points = [round(lo + width * (k + 1), 1) for k in range(5)]

    CACHE.write_text(json.dumps({
        "source": "NASA POWER climatology (PRECTOTCORR), mean annual precipitation",
        "step_deg": step,
        "cuts_mm": cut_points,
        "cuts_method": ("equal interval across the observed district range; see the module "
                        "docstring for why not quantiles"),
        "points": pts,
        "summary": {"n": len(pts), "min_mm": mms[0], "max_mm": mms[-1],
                    "median_mm": mms[len(mms) // 2]},
    }, indent=1), encoding="utf-8")
    print("class cut points (mm): %s" % cut_points)
    print("\nwrote %s: %d points, %.0f to %.0f mm (median %.0f)"
          % (CACHE.name, len(pts), mms[0], mms[-1], mms[len(mms) // 2]))
    return mms


if __name__ == "__main__":
    import sys

    if CACHE.exists() and "--rebuild" not in sys.argv:
        g = grid()
        s = g["summary"]
        print("cached: %d points, %.0f to %.0f mm (median %.0f)"
              % (s["n"], s["min_mm"], s["max_mm"], s["median_mm"]))
        for name, lat, lon in [("Sajek", 23.38, 92.29), ("Rangamati town", 22.65, 92.17),
                               ("Bilaichhari", 22.25, 92.32)]:
            rc = rainfall_class(lat, lon)
            print("  %-15s %.0f mm -> class %d" % (name, rc[1], rc[0]))
    else:
        build()
