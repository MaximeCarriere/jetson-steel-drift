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




def fig_reliability():
    """Grouped bars per certainty band: what the model SAID vs what ACTUALLY happened.

    Clearer than a line reliability diagram for this model, whose certainty is
    all-or-nothing: the strip COUNT on each band shows where the data actually is (almost
    all in the lowest and highest bands), and the gap between the two bars is the
    over-confidence.
    """
    npz = np.load(NPZ)
    conf, label = npz["conf_raw"], npz["label"].astype(float)
    edges = np.linspace(0, 1, 6)                            # 5 bands: 0-20 .. 80-100 %
    says, real, counts, mids = [], [], [], []
    for i in range(5):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf < hi if i < 4 else conf <= hi)
        counts.append(int(m.sum()))
        says.append(float(conf[m].mean()) if m.any() else 0.0)
        real.append(float(label[m].mean()) if m.any() else 0.0)
        mids.append((lo + hi) / 2)
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x - w / 2, says, w, color=RAW, alpha=0.85, label="what the model SAID")
    ax.bar(x + w / 2, real, w, color="#009E73", label="what ACTUALLY happened")
    for i in range(5):
        if counts[i]:
            ax.text(x[i] - w / 2, says[i] + 0.015, f"{says[i]:.0%}", ha="center", fontsize=8)
            ax.text(x[i] + w / 2, real[i] + 0.015, f"{real[i]:.0%}", ha="center", fontsize=8)
        top = max(says[i], real[i])
        ax.text(x[i], top + 0.08, f"{counts[i]:,} strips", ha="center", fontsize=8.5,
                color=MUTED, fontweight="bold")
    ax.set(xticks=x, xticklabels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
           ylim=(0, 1.12), ylabel="fraction", xlabel="the certainty the model gave",
           title="XP02 — when the model gives this certainty of a defect,\n"
                 "how often is it right?  (orange above green = over-confident)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.02, "Almost every strip is in the lowest or highest band — the model "
             "is all-or-nothing. In the top band it says ~99% sure but is right only 75%.",
             ha="center", fontsize=9, color=MUTED)
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
