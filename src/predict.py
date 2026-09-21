"""Inference for the landslide susceptibility model.

The original application rebuilt the feature vector as a positional list:

    fs = [[elv, slope, aspect, twi, spi, ndvi, rainfall, landcover]]
    model.predict_proba(fs)

Nothing checked that the order still matched the order used at training. Re-ordering the training
columns, or inserting a feature, would have produced confident nonsense with no error anywhere.
Here the saved artefact carries its feature names and the vector is assembled BY NAME, so a
mismatch raises instead of silently scoring the wrong thing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "landslide_model.joblib"

_BUNDLE = None


def load_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "%s not found. Run `python src/train.py` first." % MODEL_PATH)
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE


def feature_names():
    return list(load_bundle()["features"])


def predict_one(features: dict) -> dict:
    """Score a single site. `features` is keyed by feature name, not position."""
    b = load_bundle()
    names = b["features"]
    missing = [n for n in names if n not in features]
    if missing:
        raise ValueError("missing features: %s" % ", ".join(missing))
    row = pd.DataFrame([[float(features[n]) for n in names]], columns=names)
    proba = float(b["model"].predict_proba(row)[0][1])
    warnings = out_of_domain(features)
    return {
        "probability": proba,
        "warnings": warnings,
        "label": "Landslide-prone" if proba >= 0.5 else "Not landslide-prone",
        "band": band_of(proba),
        "features_used": {n: float(features[n]) for n in names},
    }


#: the observed domain of each predictor in the training inventory. Values outside it are
#: extrapolation, and the model gives no warning of its own when asked to extrapolate.
TRAIN_DOMAIN = {
    "ELEVATION": (11.0, 50.0), "SLOPE": (6.0, 31.0), "ASPECT": (2.0, 17.0),
    "TWI": (0.0, 45.0), "SPI": (0.0, 30.0), "NDVI": (0.0, 36.0),
    "RAINFALL": (0.0, 35.0), "LANDUSE": (0.0, 61.0),
}


def out_of_domain(features: dict) -> list:
    """Which supplied values the model has never seen.

    The original application fed raw Earth Engine readings straight in: elevation in metres
    (213), NDVI as a ratio (0.52), slope in degrees (0.94). The model was trained on reclassified
    factor weights, where elevation is one of four integers from 11 to 50 and NDVI one of six
    from 0 to 36. Nothing in scikit-learn objects to that, so the application returned confident
    probabilities computed from inputs it could not interpret. This function is what makes that
    visible instead of silent.
    """
    bad = []
    for k, v in features.items():
        rng = TRAIN_DOMAIN.get(k)
        if not rng:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            bad.append({"feature": k, "value": v, "reason": "not a number"})
            continue
        lo, hi = rng
        if x < lo or x > hi:
            bad.append({"feature": k, "value": x, "trained_range": [lo, hi],
                        "reason": "outside the range seen in training"})
    return bad


def band_of(p: float) -> str:
    """Coarse bands. Deliberately coarse: see the calibration note in the README."""
    if p < 0.20:
        return "Very low"
    if p < 0.40:
        return "Low"
    if p < 0.60:
        return "Moderate"
    if p < 0.80:
        return "High"
    return "Very high"


def derive_twi_spi(slope_deg: float, flow_accumulation: float) -> tuple[float, float]:
    """Topographic and stream power indices.

    The original computed these as

        twi = np.log(fac * 1000 / np.tan(slope))
        spi = fac * np.tan(slope)

    wrapped in a bare `except:` that substituted the constant 10 on any failure. On flat ground
    tan(slope) approaches zero, so TWI diverges and the except branch fired routinely, quietly
    feeding the model a fabricated 10 for a site it could not describe. A fabricated value that
    looks plausible is worse than an error, because nothing downstream can tell the difference.

    Here the slope is floored at a small positive angle, which is the standard treatment, and the
    substitution is explicit rather than hidden in an exception handler.
    """
    slope_rad = np.deg2rad(max(float(slope_deg), 0.1))   # floor: flat ground is not infinite TWI
    tan_s = float(np.tan(slope_rad))
    fac = max(float(flow_accumulation), 1e-6)
    twi = float(np.log((fac * 1000.0) / tan_s))
    spi = float(fac * tan_s)
    return twi, spi
