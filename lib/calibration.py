"""Calibration primitives — reliability bins, ECE/MCE/Brier, and temperature scaling.

numpy only, no torch: the model's job is to emit logits (done on the Jetson); everything
here is post-hoc analysis of those logits + labels, so it runs anywhere and is reusable by
XP07 (calibration under drift). Nothing here retrains or touches model weights.

Vocabulary used throughout:
  confidence  a number in [0, 1] the model attaches to a claim ("class 3 is here")
  correct     1 if the claim was true, else 0
  calibrated  confidence C means the claim is right a fraction C of the time
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Group predictions into confidence bins; per bin, mean confidence vs actual accuracy.

    This is the reliability diagram in table form: a perfectly calibrated model has
    mean_conf == accuracy in every bin (points on the diagonal).
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        n = int(m.sum())
        out.append({
            "lo": round(float(lo), 3), "hi": round(float(hi), 3), "count": n,
            "conf_mean": round(float(conf[m].mean()), 4) if n else None,
            "accuracy": round(float(correct[m].mean()), 4) if n else None,
        })
    return out


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: average |confidence - accuracy|, weighted by bin count."""
    bins = reliability_bins(conf, correct, n_bins)
    n = len(conf)
    return round(sum(b["count"] / n * abs(b["conf_mean"] - b["accuracy"])
                     for b in bins if b["count"]), 4) if n else 0.0


def mce(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Maximum Calibration Error: the worst bin's |confidence - accuracy|."""
    bins = reliability_bins(conf, correct, n_bins)
    gaps = [abs(b["conf_mean"] - b["accuracy"]) for b in bins if b["count"]]
    return round(max(gaps), 4) if gaps else 0.0


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    """Brier score: mean squared error between confidence and outcome. Lower is better."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    return round(float(np.mean((conf - correct) ** 2)), 4) if len(conf) else 0.0


def _nll(logits: np.ndarray, labels: np.ndarray, t: float) -> float:
    p = np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    grid=(0.05, 20.0), iters: int = 60) -> float:
    """Fit a single temperature T>0 minimising NLL of sigmoid(logit/T) vs labels.

    Temperature scaling is the standard post-hoc calibrator: it rescales confidence
    without changing which class ranks highest, so accuracy/AUROC are untouched — only the
    probabilities move. T>1 means the model was over-confident (softens); T<1 sharpens.
    Solved by golden-section search on a 1-D convex-ish objective — no scipy needed.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(logits) < 2 or labels.min() == labels.max():
        return 1.0                                        # nothing to calibrate against
    lo, hi = grid
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if _nll(logits, labels, c) < _nll(logits, labels, d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return round((a + b) / 2, 4)


def summary(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> dict:
    """ECE + MCE + Brier + mean confidence + base rate — the calibration scorecard."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    return {
        "n": int(len(conf)),
        "mean_confidence": round(float(conf.mean()), 4) if len(conf) else None,
        "accuracy": round(float(correct.mean()), 4) if len(conf) else None,
        "ece": ece(conf, correct, n_bins),
        "mce": mce(conf, correct, n_bins),
        "brier": brier(conf, correct),
    }
