"""Build the susceptibility map: score every cell of the district, not just the clicked one.

This is the artefact the thesis was always aiming at. A single coordinate answers "is this spot
dangerous"; a map answers "where in Rangamati is the danger", which is the question anyone
planning a road, a settlement or an evacuation route actually asks.

HOW IT IS BUILT.

    1. Lay a regular grid over the district and keep the cells whose centre falls inside the
       boundary polygon. Nothing outside is scored, because the model has no evidence there.
    2. For each kept cell, fetch a 3x3 elevation window at 90 m spacing from SRTM and compute
       slope and aspect by Horn's method. The window spacing matters: slope is scale dependent,
       and measuring it across a 1.7 km cell would flatten every hillside in the district. The
       cells are far enough apart that the windows do not overlap.
    3. Convert to the model's class ratings (src/reclassify.py) and score.

WHAT THE COLOURS MEAN, AND DO NOT MEAN. The inventory is balanced 196/196 by construction, so the
model's output is calibrated to a 50/50 world and not to real terrain, where landslides are rare.
The map is therefore a RANKING of cells against each other. A red cell is more susceptible than a
green one; it is not a cell with an 80 percent chance of failing. Five of the eight predictors are
held at fixed classes because a point query cannot measure them, so the pattern you see is driven
by terrain: slope first, then elevation and aspect.

Run it:

    python src/susceptibility_map.py              # build, skipping if already cached
    python src/susceptibility_map.py --rebuild    # force
    python src/susceptibility_map.py --step 0.02  # coarser and faster
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import reclassify
import study_area
import terrain
from predict import predict_one

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "susceptibility_grid.json"

#: OpenTopoData accepts 100 locations per request. A cell needs a 3x3 window, so 11 cells
#: (99 points) is the largest whole number of cells that fits in one call.
CELLS_PER_REQUEST = 11
WINDOW_M = 90.0


def _cells(step: float):
    """Cells inside the district, plus a one-cell margin around it.

    The margin exists for the renderer, not for the model. The map is drawn by interpolating
    between cell centres and then clipping to the boundary, so without values just outside the
    edge the interpolation would fade into transparency and leave a pale fringe inside the
    district. Margin cells are marked, and the summary statistics count only interior ones.
    """
    la0, lo0, la1, lo1 = study_area.bbox()
    out = []
    lat = la0 - step
    while lat <= la1 + step:
        lon = lo0 - step
        while lon <= lo1 + step:
            inside = study_area.contains(lat, lon)
            near = inside or any(study_area.contains(lat + dy * step, lon + dx * step)
                                 for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                                (1, 1), (1, -1), (-1, 1), (-1, -1)))
            if near:
                out.append((round(lat, 6), round(lon, 6), inside))
            lon += step
        lat += step
    return out


def _window_points(lat: float, lon: float):
    dlat = WINDOW_M / 111_320.0
    dlon = WINDOW_M / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    pts = []
    for iy in (1, 0, -1):
        for ix in (-1, 0, 1):
            pts.append((lat + iy * dlat, lon + ix * dlon))
    return pts


def _fetch(points, tries=4):
    locs = "|".join("%.6f,%.6f" % p for p in points)
    url = terrain.OPENTOPO + "?locations=" + urllib.parse.quote(locs)
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers=terrain.UA)
            with urllib.request.urlopen(req, timeout=60, context=terrain._SSL) as r:
                d = json.loads(r.read().decode("utf-8", "ignore"))
            if d.get("status") != "OK":
                raise RuntimeError("status %s" % d.get("status"))
            return [0.0 if x.get("elevation") is None else float(x["elevation"])
                    for x in d["results"]]
        except Exception as exc:
            last = exc
            time.sleep(2.0 * (a + 1))
    raise RuntimeError("elevation batch failed: %s: %s" % (type(last).__name__, str(last)[:120]))


def build(step: float = 0.015, pause: float = 1.05):
    cells = _cells(step)
    print("district cells at %.4f deg (~%.1f km): %d" % (step, step * 111.32, len(cells)))
    print("requests: %d, about %.1f minutes"
          % (math.ceil(len(cells) / CELLS_PER_REQUEST),
             math.ceil(len(cells) / CELLS_PER_REQUEST) * pause / 60.0))

    rows = []
    t0 = time.time()
    for i in range(0, len(cells), CELLS_PER_REQUEST):
        batch = cells[i:i + CELLS_PER_REQUEST]
        pts = []
        for lat, lon, _inside in batch:
            pts.extend(_window_points(lat, lon))
        try:
            elevs = _fetch(pts)
        except Exception as exc:
            print("  batch %d skipped: %s" % (i // CELLS_PER_REQUEST, str(exc)[:80]))
            time.sleep(pause)
            continue

        for k, (lat, lon, inside) in enumerate(batch):
            z = elevs[k * 9:(k + 1) * 9]
            if len(z) != 9:
                continue
            slope_deg, aspect_deg, elev = terrain.slope_aspect(z, WINDOW_M)
            conv = reclassify.reclassify({"ELEVATION_m": elev, "SLOPE_deg": slope_deg,
                                          "ASPECT_deg": aspect_deg}, lat=lat, lon=lon)
            p = predict_one(conv["features"])["probability"]
            rows.append([round(lat, 5), round(lon, 5), round(p, 4),
                         round(elev, 1), round(slope_deg, 2), 1 if inside else 0,
                         round(aspect_deg, 1)])

        done = i + len(batch)
        if (i // CELLS_PER_REQUEST) % 20 == 0 or done >= len(cells):
            el = time.time() - t0
            print("  %d/%d cells, %.0fs elapsed, %.0fs left"
                  % (done, len(cells), el, el / max(done, 1) * (len(cells) - done)))
        time.sleep(pause)

    # A rebuild that ran into a rate limit half way through would otherwise replace a complete
    # map with a holed one, and holes are hard to see on a coloured surface. Refuse to overwrite
    # unless nearly every cell came back.
    got = len(rows) / float(len(cells)) if cells else 0.0
    if got < 0.97:
        raise RuntimeError(
            "only %d of %d cells were scored (%.1f%%). Refusing to overwrite %s with an "
            "incomplete map. The elevation service is probably rate limiting; try again later."
            % (len(rows), len(cells), got * 100, OUT.name))

    ps = sorted(r[2] for r in rows if r[5])          # statistics describe the district only
    summary = {
        "cells": sum(1 for r in rows if r[5]),
        "cells_with_margin": len(rows),
        "step_deg": step,
        "cell_km": round(step * 111.32, 2),
        "median_probability": round(ps[len(ps) // 2], 4) if ps else None,
        "share_high": round(sum(1 for p in ps if p >= 0.6) / float(len(ps)), 3) if ps else None,
        "share_very_high": round(sum(1 for p in ps if p >= 0.8) / float(len(ps)), 3) if ps else None,
    }
    # Fingerprint the model this map was built from. The map is precomputed, so retraining
    # leaves it silently describing a model that no longer exists. Recording the fingerprint is
    # what lets the application say so instead of quietly serving a stale picture.
    model_path = ROOT / "models" / "landslide_model.joblib"
    h = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16] if model_path.exists() else None
    # The map also depends on how measurements become class ratings, and on whether rainfall is
    # measured at all. Fingerprint those too, or retraining is caught while a changed
    # reclassification is not, which is the more likely change of the two.
    rain = ROOT / "data" / "rainfall_grid.json"
    built_from = {
        "model_sha256_16": h,
        "reclassify_config": reclassify.config_fingerprint(),
        "rainfall_grid": (hashlib.sha256(rain.read_bytes()).hexdigest()[:16]
                          if rain.exists() else None),
        "model_mtime": model_path.stat().st_mtime if model_path.exists() else None,
        "selected_model": (json.loads((ROOT / "reports" / "metrics.json")
                                      .read_text(encoding="utf-8")).get("selected_model")
                           if (ROOT / "reports" / "metrics.json").exists() else None),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    OUT.write_text(json.dumps({
        "built_from": built_from,
        "note": ("susceptibility ranking for every grid cell inside the district. Elevation from "
                 "SRTM 30m, slope and aspect by Horn's method on a 90 m window, converted to the "
                 "model's class ratings by the reconstruction in src/reclassify.py"),
        "calibration_warning": ("the inventory is balanced 196/196 by construction, so these are "
                                "relative rankings and not probabilities that a cell will fail"),
        "held_fixed": ["TWI", "SPI", "NDVI", "RAINFALL", "LANDUSE"],
        "summary": summary,
        "columns": ["lat", "lon", "probability", "elevation_m", "slope_deg", "inside_district",
                    "aspect_deg"],
        "grid": rows,
    }, separators=(",", ":")), encoding="utf-8")
    print("\nwrote %s" % OUT.name)
    print("  %d district cells (%d with margin) | median %.3f | %.0f%% at or above 0.6 | "
          "%.0f%% at or above 0.8"
          % (summary["cells"], summary["cells_with_margin"], summary["median_probability"],
             summary["share_high"] * 100, summary["share_very_high"] * 100))


if __name__ == "__main__":
    step = 0.015
    if "--step" in sys.argv:
        step = float(sys.argv[sys.argv.index("--step") + 1])
    if OUT.exists() and "--rebuild" not in sys.argv:
        d = json.loads(OUT.read_text(encoding="utf-8"))
        print("already built: %d cells at %.4f deg. Use --rebuild to redo it."
              % (d["summary"]["cells"], d["summary"]["step_deg"]))
    else:
        build(step)
