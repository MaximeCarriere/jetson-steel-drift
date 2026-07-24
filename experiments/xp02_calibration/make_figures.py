"""XP02 figures — reliability diagrams and ECE, before vs after temperature scaling.

    python experiments/xp02_calibration/make_figures.py

Reads results/xp02_calibration.json + results/raw/xp02_reliability.npz (produced by
calibrate.py). Plots from those, so no model is needed here.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib import calibration as cal                        # noqa: E402
from lib.severstal import CLASS_IDS, ROOT                 # noqa: E402

FIG_DIR = os.path.join(ROOT, "results", "figures")
JSON = os.path.join(ROOT, "results", "xp02_calibration.json")
NPZ = os.path.join(ROOT, "results", "raw", "xp02_reliability.npz")

COLORS = {"1": "#E69F00", "2": "#56B4E9", "3": "#009E73", "4": "#D55E00"}
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "font.size": 10, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 120,
    "savefig.bbox": "tight",
})


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=130)
    plt.close(fig)
    print(f"  wrote results/figures/{name}")


def _reliability_curve(ax, conf, correct, color, label):
    bins = cal.reliability_bins(conf, correct, n_bins=10)
    xs = [b["conf_mean"] for b in bins if b["count"]]
    ys = [b["accuracy"] for b in bins if b["count"]]
    ax.plot(xs, ys, marker="o", ms=5, lw=2, color=color, label=label)


def fig_reliability():
    """One panel per class: predicted certainty vs actual correctness, raw + calibrated.

    The dashed diagonal is perfect calibration. A curve BELOW it means over-confident
    (says 80%, right less often); above means under-confident.
    """
    res = json.load(open(JSON))["prediction_level"]
    npz = np.load(NPZ)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, c in zip(axes, CLASS_IDS):
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
        _reliability_curve(ax, npz[f"pred_{c}_conf_raw"], npz[f"pred_{c}_correct"],
                           COLORS[c], "raw")
        _reliability_curve(ax, npz[f"pred_{c}_conf_cal"], npz[f"pred_{c}_correct"],
                           INK, f"calibrated (T={res[c]['temperature']:.2f})")
        r = res[c]
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="predicted certainty",
               ylabel="actually correct" if c == "1" else "",
               title=f"class {c}   ECE {r['raw']['ece']:.02f} → {r['calibrated']['ece']:.02f}")
        ax.title.set_color(COLORS[c])
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("XP02 — reliability: when the model flags a class at certainty C, is it "
                 "right C of the time?  (below the diagonal = over-confident)",
                 fontsize=12, y=1.03)
    _save(fig, "xp02_reliability.png")


def fig_ece():
    """ECE per class, prediction-level and pixel-level, raw vs after temperature scaling."""
    res = json.load(open(JSON))
    fig, (axP, axX) = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(CLASS_IDS)); w = 0.38
    for ax, level, title in ((axP, "prediction_level", "Prediction-level (operator sees)"),
                             (axX, "pixel_level", "Pixel-level")):
        d = res[level]
        raw = [d[c]["raw"]["ece"] for c in CLASS_IDS]
        calb = [d[c]["calibrated"]["ece"] for c in CLASS_IDS]
        ax.bar(x - w / 2, raw, w, label="raw", color=MUTED, alpha=0.6)
        b = ax.bar(x + w / 2, calb, w, label="temperature-scaled",
                   color=[COLORS[c] for c in CLASS_IDS])
        for i in range(len(CLASS_IDS)):
            ax.text(i - w / 2, raw[i] + 0.003, f"{raw[i]:.02f}", ha="center", fontsize=8,
                    color=MUTED)
            ax.text(i + w / 2, calb[i] + 0.003, f"{calb[i]:.02f}", ha="center", fontsize=8)
        ax.set(xticks=x, xticklabels=[f"c{c}" for c in CLASS_IDS], ylabel="ECE (lower=better)",
               title=title)
        ax.legend(fontsize=8); ax.grid(axis="x", visible=False)
    fig.suptitle("XP02 — calibration error before vs after temperature scaling",
                 fontsize=12, y=1.02)
    _save(fig, "xp02_ece.png")


def fig_confidence_hist():
    """Distribution of the model's prediction certainty, split by whether it was correct."""
    npz = np.load(NPZ)
    conf = np.concatenate([npz[f"pred_{c}_conf_raw"] for c in CLASS_IDS])
    corr = np.concatenate([npz[f"pred_{c}_correct"] for c in CLASS_IDS]).astype(bool)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(conf.min(), 1.0, 26)
    ax.hist(conf[corr], bins=bins, color="#009E73", alpha=0.7, label="correct flags")
    ax.hist(conf[~corr], bins=bins, color="#D55E00", alpha=0.7, label="false alarms")
    ax.set(xlabel="prediction certainty (raw)", ylabel="count",
           title="XP02 — do confident flags separate from false alarms?")
    ax.legend(fontsize=9); ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.02, "If the two distributions overlap heavily, certainty does not "
             "separate right from wrong — a key calibration failure mode.",
             ha="center", fontsize=8, color=MUTED)
    _save(fig, "xp02_confidence_hist.png")


def main():
    if not (os.path.isfile(JSON) and os.path.isfile(NPZ)):
        print("run calibrate.py first", file=sys.stderr); return 2
    print("generating XP02 figures...")
    fig_reliability()
    fig_ece()
    fig_confidence_hist()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
