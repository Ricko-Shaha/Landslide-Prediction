"""The missing bridge: raw physical measurements to the class weights the model was trained on.

THE PROBLEM THIS SOLVES. The model's training columns are not measurements. `SLOPE` is not an
angle, it is one of five integers {6, 13, 22, 28, 31}; `ELEVATION` is one of four {11, 13, 26, 50}.
They are class ratings from a weighted-overlay reclassification. The original web application
sampled Earth Engine and fed the raw numbers straight in, so it scored a slope of 0.94 degrees as
though it were the rating 0.94, which is below every class the model has ever seen. The answer
came back as a confident percentage. That is the defect at the centre of this project.

The conversion table was never written down. What follows is a reconstruction, and it is labelled
as one everywhere it surfaces.

WHAT IS RECOVERABLE, AND HOW.

1. THE NUMBER OF CLASSES. Each factor's distinct ratings sum to exactly 100:

       SLOPE      6 + 13 + 22 + 28 + 31          = 100   (5 classes)
       ELEVATION  11 + 13 + 26 + 50              = 100   (4 classes)
       TWI        0 + 9 + 16 + 30 + 45           = 100   (5 classes)
       SPI        0 + 6 + 17 + 23 + 24 + 30      = 100   (6 classes)
       NDVI       0 + 2 + 16 + 22 + 24 + 36      = 100   (6 classes)
       RAINFALL   0 + 5 + 10 + 21 + 29 + 35      = 100   (6 classes)
       LANDUSE    0 + 10 + 29 + 61               = 100   (4 classes)

   That is a normalisation, not a coincidence, and it confirms these are per-factor class ratings.
   ASPECT's observed ratings sum to 89, which says one class of weight 11 never occurs among the
   392 inventory sites: 8 observed plus 1 unobserved is the standard flat-plus-eight-octants
   scheme.

2. THE ORDER OF THE CLASSES. Within every factor the rating rises with the observed landslide rate
   in the inventory, monotonically for SLOPE (0.12, 0.62, 1.00, 1.00, 1.00) and TWI (0.00, 0.12,
   0.20, 0.46, 0.96). A higher rating means a more susceptible class. So the ratings are an
   ordering, and the ordering is known.

3. WHAT IS NOT RECOVERABLE: where one class stops and the next begins. The cut points below are
   the standard physical breaks used for slope and elevation reclassification in landslide
   susceptibility mapping, which is what the thesis would have applied in ArcGIS. They are the
   assumption to attack first if these numbers are ever used for anything real.

   An earlier attempt placed the cuts at quantiles of the district's own terrain instead, so that
   each class covered an equal share of the ground. That is defensible in the abstract and wrong
   here, for a reason worth recording: the 392-row inventory is balanced 196/196 BY DESIGN, so it
   is not a random sample of the district. Its slope class 22 has a 1.00 observed landslide rate.
   Mapping the district's median hillside into that class told the model that half of Rangamati
   was a certainty, and every coordinate came back above 0.96. Matching quantiles across two
   distributions only works when both describe the same population, and these two do not.

HONESTY RULES THIS FILE FOLLOWS.
  - Every returned class carries the raw value it came from, so nothing is untraceable.
  - Factors that cannot be measured from elevation alone return a stated default and are flagged
    `assumed`, never silently filled.
  - Nothing is invented inside an exception handler. That was the original bug.
  - A factor that cannot be measured is held fixed, not modelled from a proxy that happens to
    correlate. See the note on TWI below for what that rule caught.
"""
from __future__ import annotations

import bisect
from pathlib import Path

import rainfall as rainfall_src
import study_area

ROOT = Path(__file__).resolve().parents[1]

#: class ratings for every factor, ascending. Read directly off the training inventory; the
#: comment beside each is the observed landslide rate per class, which is what establishes that
#: the ordering runs from least to most susceptible.
RATINGS = {
    "ELEVATION": [11, 13, 26, 50],           # 0.24 0.88 0.68 0.82
    "SLOPE":     [6, 13, 22, 28, 31],        # 0.12 0.62 1.00 1.00 1.00
    "ASPECT":    [2, 7, 10, 11, 13, 14, 15, 17],   # flat lowest, then the eight octants
    "TWI":       [0, 9, 16, 30, 45],         # 0.00 0.12 0.20 0.46 0.96
    "SPI":       [0, 6, 17, 23, 24, 30],     # 0.00 0.14 0.36 0.69 0.49 0.56
    "NDVI":      [0, 2, 16, 22, 24, 36],     # 0.00 0.04 0.58 0.62 0.49 0.67
    "RAINFALL":  [0, 5, 10, 21, 29, 35],     # 0.00 0.20 0.30 0.59 0.70 0.65
    "LANDUSE":   [0, 10, 29, 61],            # 0.00 0.60 0.62 0.78
}

#: the eight compass octants plus flat. ASPECT is circular, so its ratings cannot be ordered by
#: the numeric aspect angle the way slope can. The assignment below is the one the regional
#: literature supports: slopes facing the southwest monsoon take the most rain and fail most
#: often, flat ground fails least. It is an assumption. ASPECT carries 0.033 of the model's total
#: feature importance, so the cost of getting it wrong is small, but it is still an assumption.
OCTANTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ASPECT_ORDER = ["N", "NE", "E", "NW", "SE", "W", "S", "SW"]   # least to most susceptible

#: factors a point query cannot measure. Each is HELD AT A FIXED CLASS and flagged, never quietly
#: substituted, so that any difference between two clicked points comes from terrain alone.
#:
#: TWI and SPI are here for a reason worth stating. Both are defined through flow accumulation,
#: which is a catchment property: you cannot get it from one point. An earlier version of this
#: file approximated them from slope, on the argument that TWI is high where water collects and
#: water collects on gentle ground. That is wrong in the way that matters. High TWI marks
#: convergent HOLLOWS on a hillslope, and slope alone cannot tell a hollow from a flat plateau, so
#: the approximation handed flat ground the most landslide-prone TWI class in the inventory (the
#: class with a 0.96 observed failure rate) and pushed lake shores to 0.91. Together TWI and SPI
#: carry 0.048 of the model's feature importance, so inventing them bought almost nothing and cost
#: the predictions their meaning. They are now held fixed.
DEFAULTS = {
    "TWI": {"value": 16, "why": ("needs catchment flow accumulation, which a single point cannot "
                                 "give; held at the middle class")},
    "SPI": {"value": 17, "why": ("needs catchment flow accumulation, which a single point cannot "
                                 "give; held at the middle class")},
    "NDVI": {"value": 16, "why": ("needs Landsat NDVI; held at the middle vegetation class")},
    "RAINFALL": {"value": 21, "why": ("no rainfall grid cached; run python src/rainfall.py to "
                                      "measure it per coordinate")},
    "LANDUSE": {"value": 29, "why": ("hill forest, the dominant cover of the district; needs "
                                     "MODIS land cover to measure")},
}

#: the physical class boundaries. Slope in degrees and elevation in metres, using the breaks that
#: are conventional in landslide susceptibility mapping and that suit this district: Rangamati
#: runs from about 30 m to about 840 m, so the elevation classes have to resolve the low hills
#: rather than spreading over a mountain range.
CUTS = {
    "SLOPE": [5.0, 15.0, 25.0, 35.0],        # 5 classes: flat, gentle, moderate, steep, very steep
    "ELEVATION": [100.0, 300.0, 500.0],      # 4 classes: valley, low hill, mid hill, high hill
}

#: out-of-fold feature importance of the selected model, read once from the metrics report. Used
#: only to state honestly how much of the model the live inputs actually cover.
def _importance():
    import json
    f = ROOT / "reports" / "metrics.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8")).get("feature_importance", {}) or {}
        except Exception:
            pass
    return {}


IMPORTANCE = _importance()

def config_fingerprint() -> str:
    """A hash of everything that changes what reclassify() returns, and nothing else.

    The district map is precomputed, so it has to know when the conversion underneath it has
    moved. Hashing this file was the obvious way to do that and the wrong one: editing a comment
    or a docstring invalidated a map whose numbers were identical, which trains the reader to
    ignore the staleness warning. This hashes the class ratings, the cut points, the aspect
    ordering and the held-constant values, so it changes exactly when the output can change.
    """
    import hashlib
    import json as _json
    payload = _json.dumps({
        "ratings": RATINGS,
        "cuts": CUTS,
        "aspect_order": ASPECT_ORDER,
        "octants": OCTANTS,
        "defaults": {k: v["value"] for k, v in DEFAULTS.items()},
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


_BREAKS = None


def breaks():
    """Class cut points, plus how much of the district actually falls in each class.

    The cut points are physical and fixed (see CUTS). The district sample is not used to place
    them; it is used to CHECK them, by reporting the share of sampled ground in each class. A
    class that no ground falls into is a class the live map can never produce, and that is
    something the reader should be able to see.
    """
    global _BREAKS
    if _BREAKS is not None:
        return _BREAKS
    ref = study_area.terrain_reference()
    pts = ref["points"]

    def coverage(key, cut_points, n_classes):
        counts = [0] * n_classes
        for r in pts:
            counts[min(bisect.bisect_right(cut_points, r[key]), n_classes - 1)] += 1
        return [round(c / float(len(pts)), 3) for c in counts]

    _BREAKS = {
        "ELEVATION": CUTS["ELEVATION"],
        "SLOPE": CUTS["SLOPE"],
        "samples": ref["samples"],
        "district_share": {
            "ELEVATION": coverage("elev", CUTS["ELEVATION"], len(RATINGS["ELEVATION"])),
            "SLOPE": coverage("slope", CUTS["SLOPE"], len(RATINGS["SLOPE"])),
        },
    }
    return _BREAKS


def _band(value, cut_points, ratings, reverse=False):
    i = bisect.bisect_right(cut_points, value)
    if reverse:
        i = len(ratings) - 1 - i
    return int(ratings[max(0, min(i, len(ratings) - 1))])


def aspect_class(aspect_deg: float, slope_deg: float) -> int:
    """Flat ground has no meaningful aspect, so it takes the lowest rating."""
    if slope_deg < 1.0:
        return int(RATINGS["ASPECT"][0])
    octant = OCTANTS[int(((aspect_deg % 360.0) + 22.5) // 45.0) % 8]
    # ratings[0] is reserved for flat, the remaining seven observed ratings span the octants
    ranked = RATINGS["ASPECT"][1:]
    pos = ASPECT_ORDER.index(octant)
    return int(ranked[min(pos, len(ranked) - 1)])


def reclassify(raw: dict, rainfall_class: int | None = None,
               lat: float | None = None, lon: float | None = None) -> dict:
    """Raw terrain to the eight model inputs.

    `raw` is what src/terrain.py returns: ELEVATION_m, SLOPE_deg, ASPECT_deg.

    `lat` and `lon` are optional and are used for one thing: looking up mean annual rainfall from
    the cached climatology grid (src/rainfall.py). Pass them and RAINFALL becomes a measured
    factor rather than a held constant. Leave them out, or build without the cache, and it falls
    back to the stated default and is flagged as held, as before.

    Returns the feature row, plus a per-feature account of where each number came from, so the
    interface can show which inputs were measured and which were assumed.
    """
    b = breaks()
    elev = float(raw["ELEVATION_m"])
    slope = float(raw["SLOPE_deg"])
    aspect = float(raw["ASPECT_deg"])

    features, provenance = {}, {}

    features["ELEVATION"] = _band(elev, b["ELEVATION"], RATINGS["ELEVATION"])
    provenance["ELEVATION"] = {"raw": round(elev, 1), "unit": "m", "measured": True,
                               "from": "SRTM 30m"}

    features["SLOPE"] = _band(slope, b["SLOPE"], RATINGS["SLOPE"])
    provenance["SLOPE"] = {"raw": round(slope, 2), "unit": "deg", "measured": True,
                           "from": "Horn's method on a 90 m SRTM window"}

    features["ASPECT"] = aspect_class(aspect, slope)
    provenance["ASPECT"] = {"raw": round(aspect, 1), "unit": "deg", "measured": True,
                            "from": "Horn's method", "octant_assumption": True}

    for name in ("TWI", "SPI", "NDVI", "RAINFALL", "LANDUSE"):
        d = DEFAULTS[name]
        features[name] = int(d["value"])
        provenance[name] = {"raw": None, "measured": False, "from": d["why"]}

    # Rainfall is the one held factor that can actually be measured from a coordinate, because it
    # is a smooth regional field rather than a catchment property. A cached climatology grid makes
    # it real; without the cache nothing changes and it stays flagged as held.
    if lat is not None and lon is not None:
        try:
            got = rainfall_src.rainfall_class(float(lat), float(lon))
        except Exception:
            got = None
        if got is not None:
            cls, mm = got
            features["RAINFALL"] = int(cls)
            provenance["RAINFALL"] = {
                "raw": round(mm), "unit": "mm/yr", "measured": True,
                "from": "NASA POWER climatology, interpolated from the cached district grid"}

    # An explicit figure from the caller always wins: it is the one number a user may know better
    # than any climatology, for instance the rainfall of the storm they are actually worried about.
    if rainfall_class is not None:
        features["RAINFALL"] = int(rainfall_class)
        provenance["RAINFALL"] = {"raw": None, "measured": False, "from": "supplied by the user"}

    measured = [k for k, v in provenance.items() if v.get("measured")]
    share = round(sum(IMPORTANCE.get(k, 0.0) for k in measured), 3)
    return {
        "features": features,
        "provenance": provenance,
        "measured": measured,
        "assumed": [k for k in provenance if k not in measured],
        "measured_importance": share,
        "reading": (
            "%d of the 8 predictors are measured at this coordinate (%s) and they carry %.3f of "
            "the model's feature importance. The rest are held at fixed classes, so the "
            "difference between two points is attributable to what was measured."
            % (len(measured), ", ".join(measured).lower(), share)),
        "reconstruction_note": (
            "The model was trained on reclassified class ratings, not measurements. The table "
            "converting one to the other was never recorded, so these class assignments are "
            "reconstructed: the class count and order come from the inventory, the cut points "
            "are the standard physical breaks for slope and elevation. Treat the result as a "
            "ranking of this site against others, not as a measured risk."),
    }


if __name__ == "__main__":
    import json

    b = breaks()
    print("class cut points, checked against %d sampled points inside the district\n"
          % b["samples"])
    for k in ("ELEVATION", "SLOPE"):
        print("  %-10s %d classes  cuts at %-26s district share %s"
              % (k, len(RATINGS[k]), b[k], b["district_share"][k]))
    print()
    for name, lat, lon in [("Rangamati town", 22.6533, 92.1750),
                           ("hill ridge east", 22.6600, 92.2100),
                           ("Kaptai lake shore", 22.4950, 92.2200)]:
        import terrain
        raw = terrain.sample_point(lat, lon)
        out = reclassify(raw)
        print("%-18s elev %5.0f m  slope %5.2f deg  ->  %s"
              % (name, raw["ELEVATION_m"], raw["SLOPE_deg"],
                 json.dumps(out["features"], separators=(",", ":"))))
