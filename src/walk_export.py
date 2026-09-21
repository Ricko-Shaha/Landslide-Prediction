"""Export the district grid in the form the walkable map needs.

`web/static/walk.js` draws the district as a solid you can send a figure across. It needs the
same numbers the susceptibility map is drawn from, but it needs them in the browser rather than
behind an endpoint, and it needs them small: the map view can afford a request per pan, a scene
running at sixty frames a second cannot.

So this reindexes `data/susceptibility_grid.json` onto the dense lattice it was sampled from,
quantises three fields to a byte each, and base64s them. 260 KB of JSON floats becomes about
70 KB that decodes in one pass. Nothing is computed here that is not already in that file.

  elev   0..255 over [elev_min, elev_max] metres. 884 m across 256 levels is a 3.5 m step,
         comfortably inside SRTM's own vertical error, so no real detail is lost.
  prob   0..255 over [0, 1]. The class breaks the map uses, .20/.40/.60/.80, land on 51, 102,
         153 and 204, so a cell never changes class in the rounding.
  slope  0..255 over [0, slope_max] degrees.
  mask   0 no data, 1 sampled but outside the district, 2 inside it.

The class breaks travel with the data because the walk has to answer a question the map never
asks: given a row of the inventory, which ground is it? Inventory rows carry factor ratings and
no coordinates, so the only honest answer is the ground whose elevation and slope fall in the
same classes, and the client needs the breaks to work that out.

    python src/walk_export.py
"""

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "susceptibility_grid.json"
OUT = ROOT / "web" / "static" / "walk_data.js"

STEP = 0.01

# Imported rather than restated, so a change to the classification cannot leave the walk
# disagreeing with the model about which class a piece of ground is in.
import sys
sys.path.insert(0, str(ROOT / "src"))
from reclassify import CUTS, RATINGS                                  # noqa: E402


def main():
    grid = json.loads(SRC.read_text(encoding="utf-8"))
    cols = grid["columns"]
    ilat, ilon = cols.index("lat"), cols.index("lon")
    iprob, ielev = cols.index("probability"), cols.index("elevation_m")
    islope, iinside = cols.index("slope_deg"), cols.index("inside_district")
    rows = grid["grid"]

    lats = [r[ilat] for r in rows]
    lons = [r[ilon] for r in rows]
    lat0, lon0 = min(lats), min(lons)
    nrow = int(round((max(lats) - lat0) / STEP)) + 1
    ncol = int(round((max(lons) - lon0) / STEP)) + 1

    elevs = [r[ielev] for r in rows]
    emin, emax = min(elevs), max(elevs)
    espan = float(emax - emin) or 1.0
    smax = max(r[islope] for r in rows)

    n = nrow * ncol
    elev = bytearray(n)
    prob = bytearray(n)
    slope = bytearray(n)
    mask = bytearray(n)

    inside = 0
    for r in rows:
        j = int(round((r[ilat] - lat0) / STEP))       # row, south to north
        i = int(round((r[ilon] - lon0) / STEP))       # column, west to east
        k = j * ncol + i
        elev[k] = int(round((r[ielev] - emin) / espan * 255))
        prob[k] = int(round(max(0.0, min(1.0, r[iprob])) * 255))
        slope[k] = int(round(min(1.0, r[islope] / smax) * 255))
        mask[k] = 2 if r[iinside] else 1
        inside += 1 if r[iinside] else 0

    def b64(b):
        return base64.b64encode(bytes(b)).decode("ascii")

    doc = {
        "note": ("Rangamati Hill District on a 0.01 degree lattice. Elevation is SRTM 30m, "
                 "slope is Horn's method over that DEM, and probability is the selected model. "
                 "The probability ranks ground within the district; it is not an annual chance "
                 "of failure."),
        "source": "data/susceptibility_grid.json",
        "built_from": grid.get("built_from", {}),
        "nrow": nrow, "ncol": ncol, "step_deg": STEP,
        "lat0": round(lat0, 5), "lon0": round(lon0, 5),
        "elev_min_m": emin, "elev_max_m": emax,
        "slope_max_deg": round(smax, 3),
        "cells_inside": inside,
        "bands": [[0.20, "Very low"], [0.40, "Low"], [0.60, "Moderate"],
                  [0.80, "High"], [1.01, "Very high"]],
        # what an inventory rating means in metres and degrees
        "breaks": {"ELEVATION": CUTS["ELEVATION"], "SLOPE": CUTS["SLOPE"]},
        "ratings": {"ELEVATION": RATINGS["ELEVATION"], "SLOPE": RATINGS["SLOPE"]},
        "elev": b64(elev), "prob": b64(prob), "slope": b64(slope), "mask": b64(mask),
    }

    # A script rather than JSON on purpose: this file is also copied into a static site that
    # has to work when it is opened straight off the disk, where a fetch of a local file is
    # blocked and a script tag is not.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.RANGAMATI=" + json.dumps(doc, separators=(",", ":")) + ";\n",
                   encoding="utf-8")

    print("%d x %d lattice, %d cells inside the district" % (nrow, ncol, inside))
    print("elevation %.0f to %.0f m, slope up to %.1f deg" % (emin, emax, smax))
    print("wrote %s (%.0f KB)" % (OUT.relative_to(ROOT), OUT.stat().st_size / 1024.0))


if __name__ == "__main__":
    main()
