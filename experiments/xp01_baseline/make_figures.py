"""XP01 — every figure for the experiment README.

    python experiments/xp01_baseline/make_figures.py            # all
    python experiments/xp01_baseline/make_figures.py --no-model # skip prediction panels

Writes PNGs to results/figures/xp01_*.png. Runs on the Jetson (needs the data + the
checkpoint + matplotlib). Figures that only read JSON/data run without the model; the
qualitative-prediction panel is the only one that loads the checkpoint.

Palette: Okabe-Ito, assigned to the four classes in fixed order and never cycled — it is
colourblind-safe, which matters because these figures are read, not just glanced at.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib.severstal import (CLASS_IDS, ROOT, TRAIN_IMG_DIR, load_index,  # noqa: E402
                           load_split, masks_for, rle_decode)

FIG_DIR = os.path.join(ROOT, "results", "figures")
RESULTS = os.path.join(ROOT, "results")

# Okabe-Ito, fixed order. Class c -> COLORS[c].
COLORS = {"1": "#E69F00", "2": "#56B4E9", "3": "#009E73", "4": "#D55E00"}
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "font.size": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "figure.dpi": 120, "savefig.bbox": "tight",
})


def _load(name: str) -> dict:
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def _save(fig, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


# --------------------------------------------------------------------------- fig 1
def fig_training_curves() -> None:
    log = _load("xp01_train_log.json")
    h = log["history"]
    ep = [r["epoch"] for r in h]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.2))

    axL.plot(ep, [r["train_loss"] for r in h], color=INK, lw=2, marker="o", ms=4)
    axL.set(xlabel="epoch", ylabel="train loss (Tversky + BCE)",
            title="Training loss falls smoothly")

    # Per-class validation recall — every class is detected and stays detected.
    for c in CLASS_IDS:
        axR.plot(ep, [r["img_recall"][c] for r in h], color=COLORS[c], lw=1.8,
                 marker=".", ms=6, label=f"class {c}", alpha=0.9)
    if all("macro_f1" in r for r in h):
        axR.plot(ep, [r["macro_f1"] for r in h], color=INK, lw=2.5, marker="o", ms=4,
                 label="macro-F1", zorder=5)
    axR.set(xlabel="epoch", ylabel="validation score", ylim=(0.0, 1.05),
            title="Per-class detection recall + macro-F1 (all four classes learned)")
    axR.legend(loc="lower right", ncol=3, framealpha=0.95, fontsize=8)
    fig.suptitle("XP01 — U-Net (ResNet-34) fine-tune, 20 epochs on the Jetson Orin Nano",
                 fontsize=12, y=1.02)
    _save(fig, "xp01_training_curves.png")


# --------------------------------------------------------------------------- galleries
def _overlay(ax, iid: str, rles: dict, img_dir: str, title: str,
             show_classes=CLASS_IDS) -> None:
    img = np.array(Image.open(os.path.join(img_dir, iid)).convert("L"))
    ax.imshow(img, cmap="gray", aspect="auto", vmin=0, vmax=255)
    m = masks_for(rles)
    for i, c in enumerate(CLASS_IDS):
        if c not in show_classes or m[i].sum() == 0:
            continue
        rgba = np.zeros((*m[i].shape, 4))
        rgb = matplotlib.colors.to_rgb(COLORS[c])
        rgba[m[i] > 0] = (*rgb, 0.45)
        ax.imshow(rgba, aspect="auto")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def fig_fold_gallery(fold: str, filename: str, title: str) -> None:
    """One clear gallery for a fold: five rows — clean, then each defect class — with ONE
    example each, shown twice side by side: the raw image, then the same image with its
    ground-truth mask. Clean has no mask, so both columns match — that IS the point.
    """
    index = load_index()
    ids = set(load_split()[fold])
    rows = [("clean", None)] + [(f"class {c}", c) for c in CLASS_IDS]

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 1.7 * len(rows)))
    for r, (label, cid) in enumerate(rows):
        if cid is None:
            pick = next((i for i in sorted(ids) if not index[i]), None)
            color = MUTED
        else:
            i = CLASS_IDS.index(cid)
            pick = next((x for x in sorted(ids)
                         if set(index[x]) == {cid} and masks_for(index[x])[i].sum() > 400),
                        None)
            color = COLORS[cid]
        if pick is not None:
            _overlay(axes[r, 0], pick, {}, TRAIN_IMG_DIR, pick)          # raw image
            _overlay(axes[r, 1], pick, index[pick], TRAIN_IMG_DIR, pick)  # + ground truth
        axes[r, 0].set_ylabel(label, color=color, fontsize=13, fontweight="bold",
                              rotation=90, labelpad=12)
        axes[r, 0].set_yticks([])
    axes[0, 0].set_title("image", fontsize=12)
    axes[0, 1].set_title("ground truth", fontsize=12)
    fig.suptitle(title, fontsize=13, y=1.0)
    _save(fig, filename)


# --------------------------------------------------------------------------- confusion
def _heatmap(ax, counts, labels_true, labels_pred, title, cmap="Blues"):
    """Row-normalised (%) confusion heatmap with count + % in each cell."""
    counts = np.asarray(counts, dtype=float)
    row = counts.sum(axis=1, keepdims=True)
    pct = np.divide(counts, row, out=np.zeros_like(counts), where=row > 0) * 100
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for (i, j), p in np.ndenumerate(pct):
        ax.text(j, i, f"{p:.0f}%\n{int(counts[i, j])}", ha="center", va="center",
                fontsize=10, color="white" if p > 55 else INK,
                fontweight="bold" if i == j else "normal")
    ax.set(xticks=range(len(labels_pred)), yticks=range(len(labels_true)),
           xticklabels=labels_pred, yticklabels=labels_true, title=title)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.grid(False)


def fig_presence_absence() -> None:
    """Does the model get 'is there ANY defect?' right — defect vs clean."""
    h = _load("xp01_baseline.json")["holdout"]["presence_absence"]
    cm = np.array(h["counts"])                            # [[TN,FP],[FN,TP]]
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    recall = tp / max(1, tp + fn)                         # defect caught
    spec = tn / max(1, tn + fp)                           # clean left alone
    acc = (tp + tn) / max(1, cm.sum())
    fig, ax = plt.subplots(figsize=(5.6, 5))
    _heatmap(ax, cm, ["clean", "defect"], ["clean", "defect"],
             f"Presence vs absence of defect (row %)\n"
             f"defect recall {recall:.0%} · clean kept {spec:.0%} · acc {acc:.0%}",
             cmap="Greens")
    fig.suptitle("XP01 — does the model see a defect at all?", fontsize=12, y=1.0)
    _save(fig, "xp01_presence_absence.png")


def fig_class_confusion() -> None:
    """Which class the model predicts for each true class (clean + 4), row %."""
    h = _load("xp01_baseline.json")["holdout"]["class_confusion"]
    labels = [{"clean": "clean"}.get(x, f"c{x}") for x in h["labels"]]
    fig, ax = plt.subplots(figsize=(7.2, 6))
    _heatmap(ax, h["counts"], labels, labels,
             "Class confusion (row % — of each true class, what was predicted)")
    fig.suptitle("XP01 — per-class confusion, frozen holdout", fontsize=12, y=1.0)
    fig.text(0.5, -0.02, "Multi-defect images (~6%) reduced to their largest class. "
             "Diagonal = correct.", ha="center", fontsize=8, color=MUTED)
    _save(fig, "xp01_class_confusion.png")


# --------------------------------------------------------------------------- specificity
def fig_specificity() -> None:
    """Per-class specificity: of the strips WITHOUT a class, how many the model correctly
    did NOT flag. The false-alarm-avoidance number, one per class."""
    h = _load("xp01_baseline.json")["holdout"]
    conf = h["confusion"]
    spec = [conf[c]["tn"] / max(1, conf[c]["tn"] + conf[c]["fp"]) for c in CLASS_IDS]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bars = ax.bar([f"class {c}" for c in CLASS_IDS], spec,
                  color=[COLORS[c] for c in CLASS_IDS])
    for b, v in zip(bars, spec):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.0%}", ha="center",
                fontsize=10, fontweight="bold")
    ax.set(ylim=(0, 1.05), ylabel="specificity",
           title="XP01 — specificity per class (how well it avoids false alarms)")
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.02, "specificity = of strips that do NOT have this class, the fraction "
             "correctly left unflagged. Class 3 lowest — it is the one over-predicted.",
             ha="center", fontsize=8, color=MUTED)
    _save(fig, "xp01_specificity.png")


# --------------------------------------------------------------------------- scorecard
def fig_holdout_dice() -> None:
    """Per-class scorecard on the frozen holdout: detection recall, precision, and
    defect-only mask Dice."""
    base = _load("xp01_baseline.json")
    h, x = base["holdout"], np.arange(len(CLASS_IDS))
    rec = [h["img_recall"][c] for c in CLASS_IDS]
    prec = [h["img_precision"][c] for c in CLASS_IDS]
    dice = [h["dice_defectonly_per_class"][c] for c in CLASS_IDS]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.26
    series = [("recall", rec, -w), ("precision", prec, 0.0), ("defect-only Dice", dice, w)]
    alphas = {"recall": 0.55, "precision": 0.8, "defect-only Dice": 1.0}
    for name, vals, off in series:
        bars = ax.bar(x + off, vals, w, label=name,
                      color=[COLORS[c] for c in CLASS_IDS], alpha=alphas[name])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                    fontsize=7.5)
    ax.set(xticks=x, xticklabels=[f"class {c}" for c in CLASS_IDS], ylim=(0, 1.05),
           ylabel="score", title="XP01 — frozen-holdout scorecard, per class")
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.02, "recall = defects found · precision = of the flags, how many were "
             "real · defect-only Dice = mask overlap where a defect exists. Class 2 (few "
             "examples) is the noisiest.", ha="center", fontsize=8, color=MUTED)
    _save(fig, "xp01_holdout_dice.png")


# --------------------------------------------------------------------------- main
def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--no-model", action="store_true")
    a.add_argument("--ckpt", default=os.path.join(ROOT, "results/raw/xp01_ckpt/best.pt"))
    args = a.parse_args()

    print("generating XP01 figures...")
    fig_fold_gallery("train", "xp01_examples.png",
                     "XP01 — Severstal: clean + the four defect classes "
                     "(image and ground truth)")
    fig_training_curves()
    if os.path.isfile(os.path.join(RESULTS, "xp01_baseline.json")):
        fig_presence_absence()
        fig_class_confusion()
        fig_specificity()
        fig_holdout_dice()
    else:
        print("  (skipping holdout figures — run evaluate.py first)")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
