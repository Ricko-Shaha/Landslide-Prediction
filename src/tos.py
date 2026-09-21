"""TOS: the relative model-selection metric from Shaha et al., IEEE RAAICON 2021.

    TOS_i = 0.5 * ( tanh(z_acc,i) - tanh(z_err,i) )

where z_acc and z_err are z-scores taken ACROSS THE CANDIDATE POOL, using the population standard
deviation. The metric is relative by design: it scores a model against the other models actually
on the table, rather than against an absolute scale.

WHY IT IS USED HERE. This thesis produced nine candidate models whose accuracies sit within one
standard deviation of each other (0.869 to 0.893, sd about 0.03). Picking by accuracy alone means
picking on noise, and picking by accuracy while quietly caring more about missed landslides than
about false alarms means the stated criterion is not the real one. TOS was written for exactly
this situation: it asks for the accuracy axis and the error axis to be named separately, and then
scores the trade-off between them.

For landslide susceptibility the expensive error is the MISSED LANDSLIDE, so the error axis here
is the false negative rate. That choice is the whole point: it is stated, and it changes the
answer.

A KNOWN LIMITATION, stated because the same author later proved it. Because the z-scores are taken
across the pool, TOS is not independent of irrelevant alternatives: adding a candidate that never
wins can still reorder the candidates above it. The selection below is therefore reported together
with the pool it was computed over, and `tos_pool_sensitivity` records what happens when the
weakest candidates are dropped.
"""
from __future__ import annotations

import numpy as np


def zscores(values) -> np.ndarray:
    """Population z-scores (ddof=0), matching the original formulation."""
    v = np.asarray(values, dtype=float)
    sd = v.std(ddof=0)
    if sd == 0.0:
        # every candidate identical on this axis: the axis carries no information, and
        # returning zeros is the only honest answer. It is not an error.
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def tos_scores(accuracy, error) -> np.ndarray:
    """TOS for every candidate in the pool.

    accuracy : higher is better (e.g. accuracy, balanced accuracy)
    error    : lower is better  (e.g. false negative rate)
    """
    za = zscores(accuracy)
    ze = zscores(error)
    return 0.5 * (np.tanh(za) - np.tanh(ze))


def rank(names, accuracy, error):
    """Return [(name, tos, accuracy, error)] sorted best first."""
    s = tos_scores(accuracy, error)
    order = np.argsort(-s)
    return [(names[i], float(s[i]), float(accuracy[i]), float(error[i])) for i in order]


def pool_sensitivity(names, accuracy, error, drop_worst=(0, 1, 2, 3)):
    """Does the winner survive removing the weakest candidates?

    This is the IIA check. If dropping a candidate that never wins changes which model is
    selected, the selection was a property of the pool rather than of the models, and that is
    worth knowing before the result is quoted.
    """
    names = list(names)
    acc = list(map(float, accuracy))
    err = list(map(float, error))
    out = []
    for k in drop_worst:
        if k >= len(names) - 2:
            break
        keep = sorted(range(len(names)), key=lambda i: -acc[i])[: len(names) - k]
        sub_names = [names[i] for i in keep]
        sub = rank(sub_names, [acc[i] for i in keep], [err[i] for i in keep])
        out.append({"dropped_weakest": k, "pool_size": len(keep), "winner": sub[0][0],
                    "tos": round(sub[0][1], 4)})
    stable = len({o["winner"] for o in out}) == 1
    return {"trials": out, "winner_stable_under_pool_change": stable}
