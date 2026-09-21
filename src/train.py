"""Train and honestly evaluate the landslide susceptibility model.

WHAT CHANGED FROM THE ORIGINAL THESIS CODE, and why.

1. ONE SPLIT BECAME REPEATED STRATIFIED CROSS-VALIDATION.
   The original reported a single number from one 75/25 split with the default seed: "Accuracy of
   Random Forest on test set: 0.88". With 392 rows, one split puts 98 samples in the test set, so
   a single percentage point is one sample. A different seed moves that number by several points,
   which means the original ranking of the nine models was partly noise. Here every model is run
   through 5-fold stratified CV repeated 6 times, and the spread is reported alongside the mean,
   so the reader can see which differences are real.

2. ACCURACY ALONE BECAME A SET OF METRICS.
   Accuracy is defensible here, because the inventory is balanced 196/196 by construction. But the
   deployed application reports a PROBABILITY, and accuracy says nothing about whether a predicted
   0.8 means anything. ROC AUC, average precision, Brier score and a calibration check are
   reported too.

3. THE BALANCE IS CALLED OUT INSTEAD OF BEING USED SILENTLY.
   A real hillslope is not 50% landslide. The 196/196 inventory is a balanced SAMPLE, so the
   model's output is a likelihood ratio, not a real-world probability of failure. Section
   `prevalence_note` in the metrics file states this, because presenting an uncalibrated 0.8 to
   someone making an evacuation decision is the kind of error this whole analysis exists to avoid.

4. FEATURE NAMES TRAVEL WITH THE MODEL.
   The original pickled a bare estimator, and the Flask app rebuilt the feature vector by hand as
   a positional list. Nothing checked that the order still matched. The saved artefact here
   carries its feature names, and the serving code looks them up by name.

5. THE MODEL IS SELECTED BY TOS, NOT BY A SINGLE HEADLINE NUMBER.
   This is the problem that started the metric work. Once every candidate sits within one standard
   deviation of every other, "pick the highest accuracy" is picking on noise, and it also hides
   the fact that the two errors are not equally expensive: in landslide susceptibility a missed
   landslide costs incomparably more than a false alarm. TOS (Shaha et al., IEEE RAAICON 2021)
   makes both axes explicit. It scores each candidate against the others actually on the table,
   on a named accuracy axis and a named error axis, so the trade-off being made is written down
   instead of assumed. See src/tos.py, including the known limitation that the score depends on
   the pool it was computed over.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
import joblib

from tos import rank, pool_sensitivity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "landslide_inventory.csv"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

#: the eight predictors the deployed application can actually obtain for an arbitrary
#: coordinate from Earth Engine. The thesis also explored PROFILE, PLAN, FAULTLINES and others,
#: but those were digitised by hand for the study area and cannot be served live, so training on
#: them would produce a model the application could never feed.
FEATURES = ["ELEVATION", "SLOPE", "ASPECT", "TWI", "SPI", "NDVI", "RAINFALL", "LANDUSE"]
TARGET = "Y"
SEED = 0

#: a reference line, not a candidate. See the selection block in main().
BASELINE = "Baseline (most frequent)"


def build_models():
    """Every model the thesis compared, plus a baseline it did not have."""
    return {
        BASELINE: DummyClassifier(strategy="most_frequent"),
        "Logistic regression": Pipeline([("s", StandardScaler()),
                                         ("m", LogisticRegression(max_iter=2000))]),
        "k-nearest neighbours": Pipeline([("s", StandardScaler()),
                                          ("m", KNeighborsClassifier(n_neighbors=5))]),
        "Decision tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "Support vector machine": Pipeline([("s", StandardScaler()),
                                            ("m", SVC(probability=True, random_state=SEED))]),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "Gradient boosting": GradientBoostingClassifier(random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=300, max_depth=5,
                                                random_state=SEED, n_jobs=-1),
    }


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df[TARGET].astype(int)
    pos = int(y.sum())
    print("rows %d | features %d | positives %d | negatives %d"
          % (len(df), len(FEATURES), pos, len(y) - pos))

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=SEED)
    scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc",
               "average_precision": "average_precision", "f1": "f1",
               "precision": "precision", "recall": "recall"}

    rows = []
    for name, model in build_models().items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for k in scoring:
            v = res["test_" + k]
            row[k] = float(np.mean(v))
            row[k + "_sd"] = float(np.std(v))
        rows.append(row)
        print("  %-24s acc %.3f +/- %.3f   auc %.3f   f1 %.3f"
              % (name, row["accuracy"], row["accuracy_sd"], row["roc_auc"], row["f1"]))

    table = pd.DataFrame(rows)
    table["fnr"] = 1.0 - table["recall"]

    # ---- selection by TOS -----------------------------------------------------------
    # Every candidate above the baseline sits within about one standard deviation of every
    # other on accuracy, so "highest accuracy wins" is a coin toss dressed as a decision. TOS
    # asks instead for the two axes to be named. The accuracy axis is cross-validated
    # accuracy. The error axis is the FALSE NEGATIVE RATE, because the expensive mistake here
    # is the missed landslide, and naming that is what makes the selection mean something.
    #
    # The baseline is excluded from the pool. It is a reference line showing that 0.88 is a
    # real result rather than a property of the data; it was never a deployable candidate,
    # and because TOS is computed across the pool, leaving it in would let a model nobody
    # would ship move the scores of the models actually under consideration. That is not a
    # workaround, it is the metric's documented behaviour, and `iia_demonstration` below
    # measures exactly how much it matters.
    pool = table[table["model"] != BASELINE]
    pool_names = list(pool["model"])
    pool_acc = [float(v) for v in pool["accuracy"]]
    pool_fnr = [float(v) for v in pool["fnr"]]

    tos_rank = rank(pool_names, pool_acc, pool_fnr)
    best_name = tos_rank[0][0]
    best = build_models()[best_name]

    all_names = [r["model"] for r in rows]
    all_acc = [float(r["accuracy"]) for r in rows]
    all_fnr = [1.0 - float(r["recall"]) for r in rows]
    order_without = [n for n, _, _, _ in tos_rank]
    order_with = [n for n, _, _, _ in rank(all_names, all_acc, all_fnr) if n != BASELINE]
    iia_demonstration = {
        "question": ("does adding a candidate that cannot win still reorder the candidates "
                     "above it? If it does, the ranking is a property of the pool as well "
                     "as of the models."),
        "irrelevant_alternative": BASELINE,
        "order_without_it": order_without,
        "order_with_it_in_the_pool": order_with,
        "order_changed": bool(order_with != order_without),
        "winner_changed": bool(order_with[0] != order_without[0]),
    }
    iia_demonstration["moved"] = [
        {"model": n, "rank_without_it": order_without.index(n) + 1,
         "rank_with_it": order_with.index(n) + 1}
        for n in order_without if order_without.index(n) != order_with.index(n)]
    sensitivity = pool_sensitivity(pool_names, pool_acc, pool_fnr)

    by_auc = str(table.sort_values("roc_auc", ascending=False).iloc[0]["model"])
    by_acc = str(table.sort_values("accuracy", ascending=False).iloc[0]["model"])

    tos_by_name = {n: sc for n, sc, _, _ in tos_rank}
    table["tos"] = table["model"].map(tos_by_name)
    table = table.sort_values("tos", ascending=False,
                              na_position="last").reset_index(drop=True)

    print("\nselection by TOS (accuracy axis: accuracy, error axis: false negative rate)")
    for n, sc, a, e in tos_rank:
        print("  %-24s TOS %+.4f   acc %.4f   fnr %.4f" % (n, sc, a, e))
    print("  selected: %s" % best_name)
    print("  ROC AUC would choose: %s | accuracy would choose: %s" % (by_auc, by_acc))
    print("  winner survives dropping the weakest candidates: %s"
          % sensitivity["winner_stable_under_pool_change"])
    print("  adding the baseline to the pool reorders the rest: %s"
          % iia_demonstration["order_changed"])

    # out-of-fold predictions give an honest confusion matrix and calibration curve
    proba = cross_val_predict(best, X, y, cv=cv.split(X, y).__class__ and
                              RepeatedStratifiedKFold(n_splits=5, n_repeats=1,
                                                      random_state=SEED),
                              method="predict_proba", n_jobs=-1)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=8, strategy="quantile")

    best.fit(X, y)
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump({"model": best, "features": FEATURES, "target": TARGET,
                 "trained_on": str(DATA.name), "sklearn_note": "load with joblib"},
                MODELS / "landslide_model.joblib")

    importances = {}
    est = best.named_steps["m"] if hasattr(best, "named_steps") else best
    if hasattr(est, "feature_importances_"):
        importances = {f: float(v) for f, v in
                       sorted(zip(FEATURES, est.feature_importances_),
                              key=lambda kv: -kv[1])}

    metrics = {
        "dataset": {"rows": int(len(df)), "positives": pos, "negatives": int(len(y) - pos),
                    "features": FEATURES},
        "protocol": "5-fold stratified CV repeated 6 times (30 fits per model), seed 0",
        "leaderboard": [{k: (None if isinstance(v, float) and np.isnan(v) else v)
                         for k, v in r.items()}
                        for r in table.to_dict(orient="records")],
        "selected_model": best_name,
        "selection": {
            "rule": "TOS, the relative selection metric of Shaha et al., IEEE RAAICON 2021",
            "formula": ("TOS_i = 0.5 * (tanh(z_acc_i) - tanh(z_err_i)), z-scores taken across "
                        "the candidate pool with the population standard deviation"),
            "why": ("Every candidate above the baseline lies within roughly one standard "
                    "deviation of every other on accuracy, so a single headline number cannot "
                    "separate them, and picking the largest one silently treats a missed "
                    "landslide and a false alarm as equally expensive. TOS requires both axes "
                    "to be named."),
            "accuracy_axis": "cross-validated accuracy, mean of 30 fits",
            "error_axis": ("false negative rate, 1 - recall: in landslide susceptibility the "
                           "missed landslide is the expensive error"),
            "candidate_pool": pool_names,
            "excluded_from_pool": {
                BASELINE: ("a reference line, not a deployable candidate; TOS is computed "
                           "across the pool, so including it would move the scores of the "
                           "models actually under consideration")},
            "ranking": [{"model": n, "tos": round(sc, 4), "accuracy": round(a, 4),
                         "fnr": round(e, 4)} for n, sc, a, e in tos_rank],
            "selected": best_name,
            "would_be_selected_by_roc_auc": by_auc,
            "would_be_selected_by_accuracy": by_acc,
            "agrees_with_roc_auc": bool(best_name == by_auc),
            "agrees_with_accuracy": bool(best_name == by_acc),
            "pool_sensitivity": sensitivity,
            "iia_demonstration": iia_demonstration,
            "limitation": ("TOS is pool-relative by construction, so it is not independent of "
                           "irrelevant alternatives: a candidate that never wins can still "
                           "reorder the candidates above it. The ranking is therefore reported "
                           "together with the pool it was computed over."),
        },
        "out_of_fold_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "out_of_fold": {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "average_precision": float(average_precision_score(y, proba)),
            "brier": float(brier_score_loss(y, proba)),
        },
        "calibration": {"predicted": [float(x) for x in mean_pred],
                        "observed": [float(x) for x in frac_pos]},
        "feature_importance": importances,
        "prevalence_note": (
            "The inventory is balanced by construction, 196 landslide and 196 non-landslide "
            "points. Real terrain is nowhere near 50 percent landslide, so the number this model "
            "returns is the probability under a balanced sample, not the probability that a given "
            "hillside will fail. To read it as a real-world risk it must be rescaled by the true "
            "base rate, which this dataset does not record. The application states this next to "
            "every number it shows."),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("out-of-fold: acc %.3f  auc %.3f  recall %.3f  brier %.3f"
          % (metrics["out_of_fold"]["accuracy"], metrics["out_of_fold"]["roc_auc"],
             metrics["out_of_fold"]["recall"], metrics["out_of_fold"]["brier"]))
    print("wrote models/landslide_model.joblib and reports/metrics.json")


if __name__ == "__main__":
    main()
