"""XP02 figures — clean-vs-defect calibration.

    python experiments/xp02_calibration/make_figures.py

Reads results/xp02_calibration.json + results/raw/xp02_reliability.npz.
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
from lib.severstal import ROOT                            # noqa: E402

FIG_DIR = os.path.join(ROOT, "results", "figures")
JSON = os.path.join(ROOT, "results", "xp02_calibration.json")
NPZ = os.path.join(ROOT, "results", "raw", "xp02_reliability.npz")

INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
RAW, CAL = "#D55E00", "#0072B2"      # raw = orange, calibrated = blue
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "font.size": 11, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight",
})


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=130)
    plt.close(fig)
    print(f"  wrote results/figures/{name}")


def _curve(ax, bins, color, label):
    xs = [b["conf_mean"] for b in bins if b["count"]]
    ys = [b["accuracy"] for b in bins if b["count"]]
    ax.plot(xs, ys, marker="o", ms=7, lw=2.5, color=color, label=label, zorder=4)


def fig_reliability():
    """Single, plain reliability diagram for the defect/no-defect decision."""
    d = json.load(open(JSON))
    fig, ax = plt.subplots(figsize=(7.2, 7))
    ax.fill_between([0, 1], [0, 1], [0, 0], color="#D55E00", alpha=0.06, zorder=0)
    ax.text(0.72, 0.28, "over-confident\n(says more than it delivers)", fontsize=9,
            color=RAW, ha="center", style="italic")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1.3, label="perfect (honest)")
    _curve(ax, d["raw"]["reliability"], RAW, f"raw  (ECE {d['raw']['ece']:.02f})")
    _curve(ax, d["calibrated"]["reliability"], CAL,
           f"calibrated, T={d['temperature']:.1f}  (ECE {d['calibrated']['ece']:.02f})")
    ax.set(xlim=(0, 1), ylim=(0, 1),
           xlabel="the model's certainty there is a defect",
           ylabel="how often there actually was a defect",
           title="XP02 — when the model says it's C% sure there's a defect,\n"
                 "is there really a defect C% of the time?")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_aspect("equal")
    _save(fig, "xp02_reliability.png")


def fig_confidence_hist():
    """Certainty distribution for real-defect strips vs real-clean strips (raw)."""
    npz = np.load(NPZ)
    conf, label = npz["conf_raw"], npz["label"].astype(bool)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(0, 1, 26)
    ax.hist(conf[label], bins=bins, color="#009E73", alpha=0.75, label="really has a defect")
    ax.hist(conf[~label], bins=bins, color=MUTED, alpha=0.7, label="really clean")
    ax.set(xlabel="the model's certainty there is a defect (raw)", ylabel="number of strips",
           title="XP02 — do defect strips and clean strips get different certainties?")
    ax.legend(fontsize=10); ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.02, "Good separation = the certainty is informative. Clean strips "
             "sitting at high certainty are false alarms.", ha="center", fontsize=9,
             color=MUTED)
    _save(fig, "xp02_confidence_hist.png")


def main():
    if not (os.path.isfile(JSON) and os.path.isfile(NPZ)):
        print("run calibrate.py first", file=sys.stderr); return 2
    print("generating XP02 figures...")
    fig_reliability()
    fig_confidence_hist()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
