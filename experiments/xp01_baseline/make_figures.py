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
from matplotlib.patches import Patch
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


def fig_fold_gallery(fold: str, filename: str, title: str, n_per: int = 3) -> None:
    """One gallery for a fold: five rows — clean, then each defect class overlaid.

    Clean is a real category (47% of the data), so it gets a row of its own with no
    overlay. Defect rows pick images carrying ONLY that class so the overlay is
    unambiguous. Called once per fold to produce a self-contained training gallery and a
    self-contained test gallery.
    """
    index = load_index()
    ids = set(load_split()[fold])
    rows = [("clean", None)] + [(f"class {c}", c) for c in CLASS_IDS]

    fig, axes = plt.subplots(len(rows), n_per, figsize=(5.2 * n_per, 1.55 * len(rows)))
    for r, (label, cid) in enumerate(rows):
        if cid is None:
            picks = [i for i in sorted(ids) if not index[i]][:n_per]
            color = MUTED
        else:
            i = CLASS_IDS.index(cid)
            picks = [x for x in sorted(ids)
                     if set(index[x]) == {cid} and masks_for(index[x])[i].sum() > 400][:n_per]
            color = COLORS[cid]
        for j in range(n_per):
            ax = axes[r, j]
            if j < len(picks):
                iid = picks[j]
                _overlay(ax, iid, index[iid] if cid else {}, TRAIN_IMG_DIR, iid)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(label, color=color, fontsize=13, fontweight="bold",
                              rotation=90, labelpad=12)
        axes[r, 0].set_yticks([])
    fig.suptitle(title, fontsize=13, y=1.005)
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


# --------------------------------------------------------------------------- scorecard
def fig_holdout_dice() -> None:
    """Per-class scorecard on the frozen holdout: detection recall, precision, and
    defect-only mask Dice — the three numbers that say whether it actually works."""
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
           ylabel="score",
           title="XP01 — frozen-holdout scorecard: every class detected, with its "
                 "mask quality")
    ax.legend(loc="upper center", ncol=3, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.text(0.5, -0.02, "recall = defects found · precision = of the flags, how many were "
             "real · defect-only Dice = mask overlap where a defect exists. Class 2 "
             "(few examples) is the noisiest.", ha="center", fontsize=8, color=MUTED)
    _save(fig, "xp01_holdout_dice.png")


# --------------------------------------------------------------------------- fig 6
def fig_predictions(ckpt: str, n_per: int = 1) -> None:
    """Qualitative: input | ground truth | model prediction, on holdout defects."""
    import torch
    from lib.models import build_model, pick_device
    base = _load("xp01_baseline.json")
    ops = base["holdout"]["operating_points"]     # per-class threshold + min_px
    th = np.array([ops[c]["threshold"] for c in CLASS_IDS])[:, None, None]
    mp = {i: ops[c]["min_px"] for i, c in enumerate(CLASS_IDS)}
    device = pick_device()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(encoder=ck["args"]["encoder"], weights=None).to(device).eval()
    model.load_state_dict(ck["model"])

    index = load_index()
    hold = load_split()["holdout"]
    picks = []
    for c in CLASS_IDS:
        for iid in sorted(hold):
            if set(index[iid]) == {c} and masks_for(index[iid])[CLASS_IDS.index(c)].sum() > 800:
                picks.append((c, iid)); break

    fig, axes = plt.subplots(len(picks), 3, figsize=(15, 1.8 * len(picks)))
    for r, (c, iid) in enumerate(picks):
        img = np.array(Image.open(os.path.join(TRAIN_IMG_DIR, iid)).convert("L"))
        x = torch.from_numpy(img).float().div(255).view(1, 1, *img.shape).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0].cpu().numpy()
        pred = prob > th                           # per-class threshold
        pred = np.stack([p if p.sum() >= mp[i] else np.zeros_like(p)
                         for i, p in enumerate(pred)])

        for ax in axes[r]:
            ax.imshow(img, cmap="gray", aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
        _paint(axes[r, 1], masks_for(index[iid]))
        _paint(axes[r, 2], pred.astype(np.float32))
        axes[r, 0].set_ylabel(f"class {c}", color=COLORS[c], fontweight="bold")
    for ax, t in zip(axes[0], ("input strip", "ground truth", "model prediction")):
        ax.set_title(t, fontsize=11)
    legend = [Patch(facecolor=COLORS[c], label=f"class {c}", alpha=0.6) for c in CLASS_IDS]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("XP01 — qualitative predictions on frozen-holdout defects "
                 "(one per class)", fontsize=12, y=1.02)
    _save(fig, "xp01_predictions.png")


def _paint(ax, masks: np.ndarray) -> None:
    for i, c in enumerate(CLASS_IDS):
        if masks[i].sum() == 0:
            continue
        rgba = np.zeros((*masks[i].shape, 4))
        rgba[masks[i] > 0] = (*matplotlib.colors.to_rgb(COLORS[c]), 0.55)
        ax.imshow(rgba, aspect="auto")


# --------------------------------------------------------------------------- main
def main() -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--no-model", action="store_true")
    a.add_argument("--ckpt", default=os.path.join(ROOT, "results/raw/xp01_ckpt/best.pt"))
    args = a.parse_args()

    print("generating XP01 figures...")
    fig_fold_gallery("train", "xp01_examples_train.png",
                     "XP01 — Severstal, TRAINING data: clean + the four defect classes")
    fig_fold_gallery("holdout", "xp01_examples_test.png",
                     "XP01 — Severstal, TEST data (frozen holdout): clean + the four "
                     "defect classes")
    fig_training_curves()
    if os.path.isfile(os.path.join(RESULTS, "xp01_baseline.json")):
        fig_presence_absence()
        fig_class_confusion()
        fig_holdout_dice()
        if not args.no_model:
            fig_predictions(args.ckpt)
    else:
        print("  (skipping holdout figures — run evaluate.py first)")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
