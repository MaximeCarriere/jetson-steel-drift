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
    axL.set(xlabel="epoch", ylabel="train loss (Dice + BCE)",
            title="Training loss falls smoothly, 0.92 → 0.19")

    axR.plot(ep, [r["dice_mean"] for r in h], color=INK, lw=2.5, marker="o", ms=4,
             label="mean", zorder=5)
    for c in CLASS_IDS:
        axR.plot(ep, [r["dice_per_class"][c] for r in h], color=COLORS[c], lw=1.6,
                 marker=".", ms=5, label=f"class {c}", alpha=0.9)
    axR.axhline(0.47, color=MUTED, ls=":", lw=1)
    axR.text(0.3, 0.482, "≈ all-clean freebie (empty=1.0)", ha="left",
             va="bottom", fontsize=8, color=MUTED)
    axR.set(xlabel="epoch", ylabel="validation Dice", ylim=(0.4, 1.0),
            title="Training-log Dice — but c1/c2's flat high lines are ABSTENTION,\n"
                  "not skill (see holdout scorecard)")
    axR.legend(loc="lower right", ncol=2, framealpha=0.95, fontsize=8)
    fig.suptitle("XP01 — U-Net (ResNet-34) fine-tune, 20 epochs on the Jetson Orin Nano",
                 fontsize=12, y=1.02)
    _save(fig, "xp01_training_curves.png")


# --------------------------------------------------------------------------- fig 2
def fig_data_distribution() -> None:
    prof = _load("xp01_data_profile.json")
    cls = prof["classes"]
    t = prof["totals"]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))

    # images per class + clean
    labels = [f"class {c}" for c in CLASS_IDS] + ["clean\n(no defect)"]
    vals = [cls[c]["images"] for c in CLASS_IDS] + [t["images_without_defect"]]
    bar_colors = [COLORS[c] for c in CLASS_IDS] + [MUTED]
    bars = ax1.bar(labels, vals, color=bar_colors)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,}", ha="center",
                 fontsize=9)
    ax1.set(ylabel="images", title="Class imbalance: c3 is 21× c2; 47% of images clean")
    ax1.grid(axis="x", visible=False)

    # defect area as % of image (median + IQR), log y
    meds = [cls[c]["area_pct_of_image"]["median"] for c in CLASS_IDS]
    p25 = [cls[c]["area_pct_of_image"]["p25"] for c in CLASS_IDS]
    p75 = [cls[c]["area_pct_of_image"]["p75"] for c in CLASS_IDS]
    x = np.arange(len(CLASS_IDS))
    err = np.array([np.array(meds) - np.array(p25), np.array(p75) - np.array(meds)])
    ax2.bar(x, meds, yerr=err, color=[COLORS[c] for c in CLASS_IDS], capsize=5,
            error_kw={"ecolor": MUTED})
    ax2.set(xticks=x, xticklabels=[f"c{c}" for c in CLASS_IDS], yscale="log",
            ylabel="defect area (% of image, log)",
            title="Defect size spans orders of magnitude\n(median, IQR whiskers)")
    ax2.grid(axis="x", visible=False)

    # bbox fill ratio — thin/wispy vs solid
    fills = [cls[c]["bbox_fill_ratio"]["median"] for c in CLASS_IDS]
    bars = ax3.bar([f"c{c}" for c in CLASS_IDS], fills,
                   color=[COLORS[c] for c in CLASS_IDS])
    for b, v in zip(bars, fills):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                 fontsize=9)
    ax3.set(ylabel="bbox fill ratio (1 = solid blob)", ylim=(0, 0.8),
            title="Shape: c3/c1 thin & wispy, c2 the most solid streak")
    ax3.grid(axis="x", visible=False)

    fig.suptitle("XP01 — Severstal training data: what the four (anonymous) classes "
                 "actually look like", fontsize=12, y=1.03)
    _save(fig, "xp01_data_distribution.png")


# --------------------------------------------------------------------------- fig 3
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


# --------------------------------------------------------------------------- fig 4
def fig_confusion() -> None:
    """Per-class image-level detection confusion on the frozen holdout.

    Counts come straight from the evaluator (stored, not reconstructed). c1/c2 land as
    all-zero TP columns — the model never fires on them — which is the finding, not a
    plotting glitch.
    """
    base = _load("xp01_baseline.json")
    h = base["holdout"]
    conf, ops = h["confusion"], h["operating_points"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, c in zip(axes, CLASS_IDS):
        cm = np.array([[conf[c]["tn"], conf[c]["fp"]],
                       [conf[c]["fn"], conf[c]["tp"]]])
        ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() * 0.5 else INK, fontweight="bold")
        rec, prec = h["img_recall"][c], h["img_precision"][c]
        op = ops[c]
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["pred\nclean", "pred\ndefect"],
               yticklabels=["true\nclean", "true\ndefect"],
               title=f"class {c}   recall {rec:.2f} · prec {prec:.2f}\n"
                     f"(op: thr={op['threshold']}, min_px={op['min_px']})")
        ax.title.set_color(COLORS[c])
        ax.grid(False)
    fig.suptitle("XP01 — image-level defect detection, frozen holdout (1,884 images, "
                 "per-class operating points)\nclasses 1 & 2: TP = 0 — the model never "
                 "detects them", fontsize=12, y=1.10)
    _save(fig, "xp01_confusion.png")


# --------------------------------------------------------------------------- fig 5
def fig_holdout_dice() -> None:
    """The honest scorecard: the freebie-inflated Kaggle Dice next to the two metrics
    that actually respond — detection recall and defect-only Dice."""
    base = _load("xp01_baseline.json")
    h, x = base["holdout"], np.arange(len(CLASS_IDS))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4.6))

    # left: Kaggle Dice (what the naive headline uses) vs defect-only Dice
    w = 0.38
    kag = [h["dice_kaggle_per_class"][c] for c in CLASS_IDS]
    pos = [h["dice_defectonly_per_class"][c] for c in CLASS_IDS]
    axL.bar(x - w / 2, kag, w, label="Kaggle Dice (incl. clean freebie)",
            color=MUTED, alpha=0.55)
    b2 = axL.bar(x + w / 2, pos, w, label="defect-only Dice (honest)",
                 color=[COLORS[c] for c in CLASS_IDS])
    for b, v in zip(b2, pos):
        axL.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                 fontsize=8)
    for i, v in enumerate(kag):
        axL.text(i - w / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=MUTED)
    axL.set(xticks=x, xticklabels=[f"c{c}" for c in CLASS_IDS], ylim=(0, 1.05),
            ylabel="Dice",
            title="Kaggle Dice hides the failure; defect-only Dice exposes it")
    axL.legend(loc="upper left", fontsize=8)
    axL.grid(axis="x", visible=False)

    # right: detection F1 & recall, val vs holdout — c1/c2 flatline at zero
    hf = [h["img_f1"][c] for c in CLASS_IDS]
    hr = [h["img_recall"][c] for c in CLASS_IDS]
    axR.bar(x - w / 2, hr, w, label="recall", color=[COLORS[c] for c in CLASS_IDS],
            alpha=0.55)
    b = axR.bar(x + w / 2, hf, w, label="F1", color=[COLORS[c] for c in CLASS_IDS])
    for i in range(len(CLASS_IDS)):
        axR.text(i - w / 2, hr[i] + 0.01, f"{hr[i]:.2f}", ha="center", fontsize=8)
        axR.text(i + w / 2, hf[i] + 0.01, f"{hf[i]:.2f}", ha="center", fontsize=8)
    axR.set(xticks=x, xticklabels=[f"c{c}" for c in CLASS_IDS], ylim=(0, 1.05),
            ylabel="score", title="Detection recall / F1 — classes 1 & 2 are at zero")
    axR.legend(loc="upper left", fontsize=8)
    axR.grid(axis="x", visible=False)

    fig.suptitle("XP01 — frozen-holdout scorecard: a competent c3/c4 detector, blind to "
                 "c1/c2", fontsize=12, y=1.02)
    fig.text(0.5, -0.02, "Class 2 has only 36 defective holdout images — noisy regardless. "
             "The point is c1 AND c2 sit at zero detection, not the exact digit.",
             ha="center", fontsize=8, color=MUTED)
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
    fig_data_distribution()
    fig_fold_gallery("train", "xp01_examples_train.png",
                     "XP01 — Severstal, TRAINING data: clean + the four defect classes")
    fig_fold_gallery("holdout", "xp01_examples_test.png",
                     "XP01 — Severstal, TEST data (frozen holdout): clean + the four "
                     "defect classes")
    fig_training_curves()
    if os.path.isfile(os.path.join(RESULTS, "xp01_baseline.json")):
        fig_confusion()
        fig_holdout_dice()
        if not args.no_model:
            fig_predictions(args.ckpt)
    else:
        print("  (skipping holdout figures — run evaluate.py first)")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
