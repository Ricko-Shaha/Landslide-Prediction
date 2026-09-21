"""Flask application for the landslide susceptibility model.

RUNS WITHOUT CREDENTIALS. The original `app.py` called `ee.Initialize()` at import time, so the
whole application refused to start unless a Google Earth Engine account was already authenticated
on the machine. That makes the project impossible to demonstrate, review or grade without
credentials it cannot ship. Earth Engine is now optional and lazily initialised: the page works
immediately from the inventory's own sites or from manually entered terrain values, and the live
lookup switches itself on only when credentials happen to be present.

Three ways to obtain a feature vector:
  SITE    a real row from the study inventory, so the numbers are genuine field data
  MANUAL  terrain values typed or dragged, for exploring the model's response surface
  LIVE    click a point on the map; elevation comes from SRTM 30m over a public endpoint that
          needs no account, slope and aspect are computed from it, and the result is converted
          into the model's class ratings by src/reclassify.py

LIVE MODE IS THE POINT OF THE PROJECT, AND IT WAS THE PART THAT DID NOT WORK. Two separate faults
kept it from being usable, and both are fixed here:

  1. It could not start. `ee.Initialize()` ran at import, so a missing Google Earth Engine
     credential took the whole application down with it. Elevation now comes from SRTM through
     OpenTopoData, which is the same dataset the thesis used and needs no account at all.
  2. When it did run, it was wrong. The model is trained on reclassified class ratings, and the
     old handler fed it raw Earth Engine readings: 213 m of elevation into a column whose four
     legal values are 11, 13, 26 and 50. Every live answer was computed from inputs the model had
     never seen. src/reclassify.py now stands between the measurement and the model.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.predict import band_of, derive_twi_spi, feature_names, predict_one  # noqa: E402

import reclassify  # noqa: E402
import study_area  # noqa: E402
import terrain  # noqa: E402

app = Flask(__name__)
# Reload templates on change even with debug off. A cached template served a stale page during
# development and made a corrected file look unchanged.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

DATA = ROOT / "data" / "landslide_inventory.csv"
METRICS = ROOT / "reports" / "metrics.json"

_EE_STATE = {"tried": False, "ok": False, "error": ""}


@app.get("/healthz")
def health():
    """Check the local model without calling any external terrain service."""
    feature_names()
    return jsonify({"status": "ok"})


@app.get("/deployment-config.js")
def deployment_config():
    """Allow a host environment variable to override the optional static TOS link."""
    url = os.environ.get("PORTFOLIO_TOS_URL", "").strip()
    overrides = {"tosReferenceUrl": url} if url else {}
    response = app.response_class(
        "window.RANGAMATI_CONFIG = Object.assign({}, window.RANGAMATI_CONFIG, "
        + json.dumps(overrides) + ");\n",
        mimetype="application/javascript",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def ee_ready():
    """Initialise Earth Engine once, and remember the outcome rather than raising on import."""
    if _EE_STATE["tried"]:
        return _EE_STATE["ok"]
    _EE_STATE["tried"] = True
    try:
        import ee
        ee.Initialize()
        _EE_STATE["ok"] = True
    except Exception as exc:                      # no credentials, no network, no package
        _EE_STATE["ok"] = False
        _EE_STATE["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
    return _EE_STATE["ok"]


def load_metrics():
    if METRICS.exists():
        return json.loads(METRICS.read_text(encoding="utf-8"))
    return {}


def study_sites(n=60):
    """Real rows from the inventory, so the demo scores genuine field measurements."""
    df = pd.read_csv(DATA)
    feats = feature_names()
    keep = df[feats + ["Y"]].copy()
    pos = keep[keep["Y"] == 1].head(n // 2)
    neg = keep[keep["Y"] == 0].head(n // 2)
    out = []
    for i, (_, r) in enumerate(pd.concat([pos, neg]).iterrows()):
        out.append({"id": i,
                    "label": int(r["Y"]),
                    "features": {f: float(r[f]) for f in feats}})
    return out


@app.route("/")
def index():
    m = load_metrics()
    return render_template("index.html",
                           features=feature_names(),
                           metrics=m,
                           area=study_area.bbox(),
                           area_name=json.loads(
                               (ROOT / "data" / "study_area.json").read_text(encoding="utf-8")
                           )["name"],
                           ee_available=ee_ready(),
                           ee_error=_EE_STATE["error"])


@app.route("/api/schema")
def api_schema():
    """The real domain of every predictor, read from the training data.

    This exists because the predictors are NOT physical measurements. They are reclassified
    factor weights: ELEVATION takes four distinct integers between 11 and 50, ASPECT takes eight
    between 2 and 17, NDVI takes six between 0 and 36. Anything outside those sets is a value the
    model has never seen, so the interface offers the observed values rather than a plausible
    looking physical range.
    """
    df = pd.read_csv(DATA)
    out = {}
    for f in feature_names():
        vals = sorted(float(v) for v in df[f].unique())
        out[f] = {"values": vals, "min": vals[0], "max": vals[-1],
                  "n": len(vals), "mean": float(df[f].mean())}
    return jsonify(out)


@app.route("/api/sites")
def api_sites():
    return jsonify(study_sites())


@app.route("/api/metrics")
def api_metrics():
    return jsonify(load_metrics())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    feats = payload.get("features") or {}
    try:
        result = predict_one(feats)
    except Exception as exc:
        return jsonify({"error": "%s: %s" % (type(exc).__name__, exc)}), 400
    return jsonify(result)


@app.route("/notebook")
def notebook():
    """Serve the analysis notebook as a readable page.

    A .ipynb is a data format. Opening one from disk in a browser shows raw JSON, so the analysis
    is effectively unreadable unless the reader happens to have Jupyter or VS Code. The HTML is
    generated by src/nb_to_html.py and served here so the notebook is one click from the app.
    """
    f = ROOT / "notebooks" / "01_analysis.html"
    if not f.exists():
        return ("<p>The notebook has not been rendered yet. Run "
                "<code>python src/nb_to_html.py</code>.</p>"), 404
    return app.response_class(f.read_text(encoding="utf-8"), mimetype="text/html")


@app.route("/api/boundary")
def api_boundary():
    """The study area outline, for the map and for refusing clicks outside it.

    The polygon is the district boundary that shipped with the original repository as 11,774
    undocumented `lon,lat` pairs. It is a closed ring enclosing 5,782 km2, which identifies it as
    Rangamati district rather than as a list of sample sites.
    """
    la0, lo0, la1, lo1 = study_area.bbox()
    return jsonify({
        "name": json.loads((ROOT / "data" / "study_area.json").read_text(encoding="utf-8"))["name"],
        "ring": [[p[0], p[1]] for p in study_area.ring()],
        "bbox": {"south": la0, "west": lo0, "north": la1, "east": lo1},
        "area_km2": round(study_area.area_km2()),
    })


@app.route("/api/susceptibility")
def api_susceptibility():
    """The precomputed district map: every grid cell inside the boundary, scored.

    Built offline by src/susceptibility_map.py because each cell needs its own elevation window
    from a shared public endpoint, and doing that on request would take minutes and hammer a
    service that is free.
    """
    f = ROOT / "data" / "susceptibility_grid.json"
    if not f.exists():
        return jsonify({"error": "the district map has not been built yet",
                        "hint": "run: python src/susceptibility_map.py"}), 404

    payload = json.loads(f.read_text(encoding="utf-8"))

    # The map is precomputed, so it can fall behind the model without anything appearing wrong.
    # Compare the fingerprint recorded at build time against the model on disk now, and say so.
    model_path = ROOT / "models" / "landslide_model.joblib"
    stale, why = False, None
    built = payload.get("built_from") or {}
    if model_path.exists():
        current = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
        if not built.get("model_sha256_16"):
            stale, why = True, ("this map was built before the build fingerprint existed, so it "
                                "cannot be matched against the current model")
        elif built["model_sha256_16"] != current:
            stale, why = True, ("the model has been retrained since this map was built; rerun "
                                "python src/susceptibility_map.py --rebuild")

    if not stale:
        rain = ROOT / "data" / "rainfall_grid.json"
        rc_now = reclassify.config_fingerprint()
        rain_now = hashlib.sha256(rain.read_bytes()).hexdigest()[:16] if rain.exists() else None
        if built.get("reclassify_config") not in (None, rc_now):
            stale, why = True, ("the conversion from measurements to class ratings has changed "
                                "since this map was built; rerun "
                                "python src/susceptibility_map.py --rebuild")
        elif built.get("rainfall_grid") != rain_now and rain_now is not None:
            stale, why = True, ("rainfall is now measured per coordinate, which this map was "
                                "built without; rerun python src/susceptibility_map.py --rebuild")
    payload["stale"] = stale
    payload["stale_reason"] = why
    return app.response_class(json.dumps(payload), mimetype="application/json")


@app.route("/api/sample", methods=["POST"])
def api_sample():
    """Measure the terrain at a coordinate and score it.

    No credentials. Elevation comes from SRTM 30m, slope and aspect from Horn's method on a 90 m
    window, and src/reclassify.py converts those into the class ratings the model was trained on.

    The response separates what was MEASURED from what was ASSUMED, because only three of the
    eight predictors can be obtained from elevation. Those three carry 0.819 of the model's
    feature importance, and the other five are held at fixed classes, so the difference between
    two clicked points is attributable to terrain.
    """
    payload = request.get_json(force=True, silent=True) or {}
    try:
        lat = float(payload["lat"])
        lon = float(payload["lon"])
    except Exception:
        return jsonify({"error": "lat and lon are required"}), 400

    # Outside the district the model has no evidence at all, so nothing is scored. Returning a
    # number with a caveat attached invites the caveat to be ignored; refusing does not.
    if not study_area.contains(lat, lon):
        return jsonify({
            "refused": True,
            "lat": lat, "lon": lon,
            "reason": "outside_study_area",
            "message": ("This point is outside %s, the only ground this model was trained on. "
                        "No susceptibility is shown, because the model has no evidence about "
                        "terrain it has never seen and a number here would be invented."
                        % json.loads((ROOT / "data" / "study_area.json")
                                     .read_text(encoding="utf-8"))["name"]),
            "hint": "Click inside the outlined district.",
        }), 422

    inside = True
    try:
        raw = terrain.sample_point(lat, lon)
    except Exception as exc:
        return jsonify({"error": "elevation lookup failed",
                        "detail": "%s: %s" % (type(exc).__name__, str(exc)[:200]),
                        "hint": "the public elevation endpoint is rate limited; try again"}), 502

    rainfall = payload.get("rainfall")
    conv = reclassify.reclassify(raw, rainfall_class=rainfall, lat=lat, lon=lon)
    result = predict_one(conv["features"])

    return jsonify({
        "lat": lat, "lon": lon,
        "in_study_area": inside,
        "raw": raw,
        "features": conv["features"],
        "provenance": conv["provenance"],
        "measured": conv["measured"],
        "assumed": conv["assumed"],
        "measured_importance": conv["measured_importance"],
        "reading": conv["reading"],
        "reconstruction_note": conv["reconstruction_note"],
        "prediction": result,
    })


@app.route("/api/sample_ee", methods=["POST"])
def api_sample_ee():
    """The original Earth Engine path, kept for comparison when credentials happen to exist.

    It returns RAW physical readings, which is exactly what the old application fed to the model.
    It is retained so the defect can be reproduced rather than just described: compare these
    numbers against /api/schema and see that almost none of them fall inside a trained range.
    """
    if not ee_ready():
        return jsonify({"error": "Earth Engine is not initialised on this machine.",
                        "detail": _EE_STATE["error"],
                        "hint": "Live mode does not need it. This endpoint exists only to "
                                "reproduce the original behaviour."}), 503
    payload = request.get_json(force=True, silent=True) or {}
    try:
        lat = float(payload["lat"])
        lon = float(payload["lon"])
    except Exception:
        return jsonify({"error": "lat and lon are required"}), 400

    import ee
    try:
        poi = ee.Geometry.Point(lon, lat)
        scale = 1000

        elv_img = ee.Image("USGS/SRTMGL1_003")
        elevation = elv_img.sample(poi, scale).first().get("elevation").getInfo()
        slope = ee.Terrain.slope(elv_img).sample(poi, scale).first().get("slope").getInfo()
        aspect = ee.Terrain.aspect(elv_img).sample(poi, scale).first().get("aspect").getInfo()

        lc = ee.ImageCollection("MODIS/006/MCD12Q1")
        landuse = lc.first().sample(poi, scale).first().get("LC_Type1").getInfo()

        l8 = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA")
        img = ee.Image(l8.filterBounds(poi).filterDate("2021-01-01", "2021-03-31")
                       .sort("CLOUD_COVER").first())
        nir, red = img.select("B5"), img.select("B4")
        ndvi_img = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        ndvi = ndvi_img.reduceRegion(geometry=poi.buffer(scale), reducer=ee.Reducer.mean(),
                                     scale=30, maxPixels=int(1e9)).get("NDVI").getInfo()

        fa = ee.Image("MERIT/Hydro/v1_0_1").select("upa")
        fac = fa.sample(poi, scale).getInfo()["features"][0]["properties"]["upa"]

        twi, spi = derive_twi_spi(slope, fac)
        rainfall = float(payload.get("rainfall", 28))

        feats = {"ELEVATION": elevation, "SLOPE": slope, "ASPECT": aspect,
                 "TWI": twi, "SPI": spi, "NDVI": ndvi,
                 "RAINFALL": rainfall, "LANDUSE": landuse}
        from src.predict import out_of_domain
        return jsonify({"features": feats, "flow_accumulation": fac,
                        "raw_units": True,
                        "out_of_domain": out_of_domain(feats),
                        "warning": ("these are raw physical readings in physical units. The "
                                    "model was trained on class ratings, so feeding these "
                                    "straight in is the original defect.")})
    except Exception as exc:
        return jsonify({"error": "Earth Engine sampling failed",
                        "detail": "%s: %s" % (type(exc).__name__, str(exc)[:200])}), 502


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", "5000"))
    print("study area: %s, %.0f km2" % (
        json.loads((ROOT / "data" / "study_area.json").read_text(encoding="utf-8"))["name"],
        study_area.area_km2()))
    print("live coordinate mode: enabled, no credentials required")
    print("open http://127.0.0.1:%d" % port)
    # debug=True brings Flask's watchdog reloader, which on this machine walks the whole Python
    # installation and restarts the server every time anything under Lib/ is touched, dropping
    # live connections mid-request. Templates still reload on change (TEMPLATES_AUTO_RELOAD
    # above), which is the only part of the reloader this project needs. threaded=True matters
    # because /api/sample blocks on a network call, and without it one slow lookup stalls the
    # page for everyone.
    app.run(debug=False, use_reloader=False, threaded=True, port=port)
