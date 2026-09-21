"""The study area: the Rangamati district boundary, and the terrain distribution inside it.

WHERE THE POLYGON CAME FROM. The first version of this project carried a file named `coordinates_3`: 11,774
bare `lon,lat` pairs with no header and no documentation, plus an empty `coordinates2`. The points
turn out to be a single closed ring, consecutive vertices about 12 m apart, enclosing 5,792 km2.
Rangamati district is about 6,116 km2, so this is the district boundary traced off a map, not a
list of sampled sites. That matters: it means there is no hidden coordinate-to-label table hiding
in the repository, and the 392 inventory rows really are the only labelled data.

WHAT IT IS USED FOR HERE.

1. A mask. The model knows one district's terrain. A click in Dhaka should be refused, not scored.
2. Breakpoints. The training columns are reclassified class weights and the conversion table from
   raw measurement to class was never recorded. The class COUNT and ORDER are recoverable from the
   inventory (see src/reclassify.py); what is missing is where one class stops and the next
   begins. Those cut points are estimated from the actual distribution of terrain inside this
   polygon, which is what a quantile reclassification in a GIS would have done.

Run this module directly to rebuild the cached sample:

    python src/study_area.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RING_JSON = ROOT / "data" / "study_area.json"
TERRAIN_JSON = ROOT / "data" / "study_area_terrain.json"

_RING = None


def ring():
    """The boundary as a list of (lat, lon), loaded once."""
    global _RING
    if _RING is None:
        _RING = [tuple(p) for p in json.loads(RING_JSON.read_text(encoding="utf-8"))["ring"]]
    return _RING


def bbox():
    r = ring()
    la = [p[0] for p in r]
    lo = [p[1] for p in r]
    return min(la), min(lo), max(la), max(lo)


def contains(lat: float, lon: float) -> bool:
    """Ray casting. The ring is a simple closed polygon, so this is sufficient."""
    r = ring()
    inside = False
    n = len(r)
    j = n - 1
    for i in range(n):
        yi, xi = r[i]
        yj, xj = r[j]
        if (yi > lat) != (yj > lat):
            x_at = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_at:
                inside = not inside
        j = i
    return inside


def area_km2() -> float:
    r = ring()
    a = 0.0
    for i in range(len(r)):
        y1, x1 = r[i]
        y2, x2 = r[(i + 1) % len(r)]
        a += x1 * y2 - x2 * y1
    lat0 = sum(p[0] for p in r) / len(r)
    return abs(a) / 2.0 * (111.32 ** 2) * math.cos(math.radians(lat0))


def terrain_reference():
    """Cached elevation, slope and aspect sampled across the district."""
    return json.loads(TERRAIN_JSON.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# rebuild helpers, run as a script

def _simplify(pts, keep):
    """Every nth vertex. The ring is far denser than a mask or a map outline needs."""
    if len(pts) <= keep:
        return pts
    step = len(pts) / float(keep)
    out = [pts[int(i * step)] for i in range(keep)]
    if out[-1] != out[0]:
        out.append(out[0])
    return out


def _build_ring(src: Path, keep: int = 1400):
    pts = []
    for line in src.read_text(encoding="utf-8", errors="ignore").replace(" ", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        lon, lat = line.split(",")
        pts.append((round(float(lat), 6), round(float(lon), 6)))
    small = _simplify(pts, keep)
    RING_JSON.write_text(json.dumps({
        "name": "Rangamati Hill District, Bangladesh",
        "source": ("traced boundary carried by the first version of this project as `coordinates_3`, "
                   "11774 vertices, reduced to %d for use as a mask and map outline" % len(small)),
        "vertices_original": len(pts),
        "ring": [list(p) for p in small]}, indent=1), encoding="utf-8")
    print("wrote %s: %d of %d vertices" % (RING_JSON.name, len(small), len(pts)))


def _sample_terrain(n: int = 260, seed: int = 0):
    """Elevation, slope and aspect at random points inside the district.

    This is the empirical distribution the class breakpoints are read off. It is fetched once and
    cached, because the elevation service is a shared public endpoint and there is no reason to
    ask it the same question twice.
    """
    from terrain import elevation_grid, slope_aspect      # local import: only needed to rebuild

    rnd = random.Random(seed)
    la0, lo0, la1, lo1 = bbox()
    rows = []
    tries = 0
    while len(rows) < n and tries < n * 40:
        tries += 1
        lat = rnd.uniform(la0, la1)
        lon = rnd.uniform(lo0, lo1)
        if not contains(lat, lon):
            continue
        try:
            z, step = elevation_grid(lat, lon)
            s, a, e = slope_aspect(z, step)
        except Exception as exc:
            print("  skip %.4f,%.4f: %s" % (lat, lon, str(exc)[:60]))
            continue
        rows.append({"lat": round(lat, 5), "lon": round(lon, 5),
                     "elev": round(e, 1), "slope": round(s, 3), "aspect": round(a, 1)})
        if len(rows) % 20 == 0:
            print("  %d/%d" % (len(rows), n))

    def q(vals, ps):
        v = sorted(vals)
        out = []
        for p in ps:
            i = p * (len(v) - 1)
            lo_i, hi_i = int(math.floor(i)), int(math.ceil(i))
            out.append(round(v[lo_i] + (v[hi_i] - v[lo_i]) * (i - lo_i), 3))
        return out

    elev = [r["elev"] for r in rows]
    slope = [r["slope"] for r in rows]
    ps = [i / 20.0 for i in range(21)]
    TERRAIN_JSON.write_text(json.dumps({
        "note": ("random points inside the district boundary, elevation from SRTM 30m via "
                 "OpenTopoData, slope and aspect by Horn's method on a 90 m window"),
        "samples": len(rows),
        "percentiles": ps,
        "elevation_m": q(elev, ps),
        "slope_deg": q(slope, ps),
        "points": rows}, indent=1), encoding="utf-8")
    print("wrote %s: %d samples" % (TERRAIN_JSON.name, len(rows)))
    print("  elevation m   min %.0f  median %.0f  p95 %.0f  max %.0f"
          % (min(elev), q(elev, [.5])[0], q(elev, [.95])[0], max(elev)))
    print("  slope deg     min %.1f  median %.1f  p95 %.1f  max %.1f"
          % (min(slope), q(slope, [.5])[0], q(slope, [.95])[0], max(slope)))


if __name__ == "__main__":
    import sys

    src = ROOT.parent / ".landslide-src" / "coordinates_3"
    if not RING_JSON.exists():
        if not src.exists():
            sys.exit("need %s to build the ring" % src)
        _build_ring(src)
    print("study area: %s, %.0f km2" % (
        json.loads(RING_JSON.read_text(encoding="utf-8"))["name"], area_km2()))
    if not TERRAIN_JSON.exists() or "--refetch" in sys.argv:
        _sample_terrain()
    else:
        t = terrain_reference()
        print("terrain reference already cached: %d samples" % t["samples"])
