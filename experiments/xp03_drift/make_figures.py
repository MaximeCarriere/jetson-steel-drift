"""XP03 figures — degradation curves (accuracy / recall / specificity vs drift severity).

    python experiments/xp03_drift/make_figures.py

Reads results/xp03_degradation.json. (The contact sheet is made by contact_sheet.py.)
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib.severstal import ROOT                            # noqa: E402

FIG_DIR = os.path.join(ROOT, "results", "figures")
JSON = os.path.join(ROOT, "results", "xp03_degradation.json")
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
C = {"accuracy": "#0072B2", "recall": "#D55E00", "specificity": "#009E73"}
TITLE = {"light_corner": "Strong glare — defects vanish (recall → 0)",
         "marks": "Blob contamination — occludes defects (misses)",
         "streaks": "Defect-like streaks — false alarms"}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "font.size": 11, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight",
})


def main():
    if not os.path.isfile(JSON):
        print("run degradation.py first", file=sys.stderr); return 2
    d = json.load(open(JSON))
    sev = d["severities"]
    kinds = list(d["curves"])
    fig, axes = plt.subplots(1, len(kinds), figsize=(7 * len(kinds), 5), sharey=True)
    for ax, kind in zip(axes, kinds):
        curve = d["curves"][kind]
        for metric in ("accuracy", "recall", "specificity"):
            ys = [curve[f"{s:g}"][metric] for s in sev]
            ax.plot(sev, ys, marker="o", ms=7, lw=2.5, color=C[metric],
                    label=f"{metric}  ({ys[0]:.2f} → {ys[-1]:.2f})")
        ax.set(xlabel="drift severity", ylim=(0, 1.02), title=TITLE.get(kind, kind))
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("clean-vs-defect score")
    fig.suptitle("XP03 — how the model's defect detection degrades as each drift ramps up",
                 fontsize=13, y=1.02)
    fig.text(0.5, -0.02, "Severity 0 = the clean baseline. Glare and blobs make the model "
             "MISS defects (recall down — washed out or covered); defect-like streaks make "
             "it FALSE-ALARM (specificity down).", ha="center", fontsize=9, color=MUTED)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "xp03_degradation.png"), dpi=130)
    print("  wrote results/figures/xp03_degradation.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
