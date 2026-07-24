"""XP03 — does drift make the model hallucinate a defect on CLEAN steel?

    python experiments/xp03_drift/clean_detection.py     # needs the model (run on the Jetson)

Takes one genuinely clean strip and, for each drift x severity, overlays the model's
predicted defect region in red. Severity 0 = untouched clean strip, so red should be empty.
Any red is a FALSE alarm — the drift making the model see a defect that isn't there.

Columns = the three drifts (same order as the degradation curves); rows = rising severity.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib import drift                                    # noqa: E402
from lib.calibration import sigmoid                      # noqa: E402
from lib.models import build_model, pick_device          # noqa: E402
from lib.severstal import CLASS_IDS, ROOT, TRAIN_IMG_DIR, load_split  # noqa: E402

FIG_DIR = os.path.join(ROOT, "results", "figures")
CKPT = os.path.join(ROOT, "results/raw/xp01_ckpt/best.pt")
BASELINE = os.path.join(ROOT, "results/xp01_baseline.json")
CLEAN = "74e586515.jpg"          # smooth, uniform, genuinely clean holdout strip
LABELS = {"light_corner": "light glare", "marks": "blob contamination",
          "streaks": "defect-like streaks"}


def _severity_arrow(fig) -> None:
    """A downward 'severity' arrow on the far left; rows labelled with just the number."""
    import matplotlib.patches as mpatches
    fig.subplots_adjust(left=0.10)
    fig.text(0.028, 0.5, "severity", rotation=90, va="center", ha="center", fontsize=13,
             fontweight="bold")
    fig.add_artist(mpatches.FancyArrowPatch(
        (0.055, 0.87), (0.055, 0.13), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=22, lw=2.2, color="#333"))


@torch.no_grad()
def predict_mask(model, img_u8, ops, device):
    """Predicted defect pixels (union of fired classes) on one strip -> bool [H,W]."""
    x = torch.from_numpy(img_u8).float().div(255).view(1, 1, *img_u8.shape).to(device)
    probs = sigmoid(model(x).float().cpu().numpy())[0]        # [4,H,W]
    pred = np.zeros(img_u8.shape, dtype=bool)
    fired = False
    for i, c in enumerate(CLASS_IDS):
        thr, mp = ops[c]["threshold"], ops[c]["min_px"]
        m = probs[i] > thr
        if m.sum() >= max(1, mp):
            pred |= m
            fired = True
    return pred, fired


def main() -> int:
    if not os.path.isfile(CKPT):
        print(f"error: no checkpoint at {CKPT}", file=sys.stderr); return 2
    ops = json.load(open(BASELINE))["holdout"]["operating_points"]
    device = pick_device()
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = build_model(encoder=ck["args"]["encoder"], weights=None).to(device).eval()
    model.load_state_dict(ck["model"])

    iid = CLEAN if CLEAN in load_split()["holdout"] else \
        next(i for i in sorted(load_split()["holdout"]))
    img = np.array(Image.open(os.path.join(TRAIN_IMG_DIR, iid)).convert("L"))

    kinds, sev = drift.KINDS, drift.SEVERITIES
    fig, axes = plt.subplots(len(sev), len(kinds), figsize=(5 * len(kinds), 1.5 * len(sev)))
    for r, s in enumerate(sev):
        for c, kind in enumerate(kinds):
            ax = axes[r, c]
            drifted = drift.apply(img, kind, s, seed=1)
            pred, fired = predict_mask(model, drifted, ops, device)
            ax.imshow(drifted, cmap="gray", aspect="auto", vmin=0, vmax=255)
            rgba = np.zeros((*pred.shape, 4))
            rgba[pred] = (0.9, 0.1, 0.1, 0.55)                # red = model's detection
            ax.imshow(rgba, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if fired:
                ax.text(0.02, 0.9, "DEFECT!", transform=ax.transAxes, color="red",
                        fontsize=10, fontweight="bold", va="top")
            if r == 0:
                ax.set_title(LABELS[kind], fontsize=12, fontweight="bold")
        axes[r, 0].set_ylabel(f"{s:g}", fontsize=12, fontweight="bold", rotation=0,
                              labelpad=12, va="center")
    _severity_arrow(fig)
    fig.suptitle(f"XP03 — a CLEAN strip: does drift make the model see a defect? "
                 f"(red = false detection) · strip {iid}", fontsize=13, y=1.0)
    fig.text(0.5, -0.01, "Severity 0 (top) is the untouched clean strip — it should be "
             "blank. Any red is a false alarm the drift created.", ha="center", fontsize=9,
             color="#666")
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "xp03_clean_detection.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
