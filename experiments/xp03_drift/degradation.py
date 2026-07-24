"""XP03 — degradation: how much does each drift break the model's defect/clean decision?

    python experiments/xp03_drift/degradation.py

For each drift and each severity, apply the drift to every frozen-holdout strip, run the
model, and score the **clean-vs-defect** decision (not per class). We report:

  accuracy     overall correct (defect flagged, clean left alone)
  recall       of real defects, how many still caught  -> glare should crush this
  specificity  of clean strips, how many left alone     -> marks should crush this

The true labels don't change — the steel is the same, we only corrupt the image — so this
is honest ground-truth degradation, the curve the drift monitor will later predict blind.

Writes results/xp03_degradation.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib import drift                                    # noqa: E402
from lib.calibration import sigmoid                      # noqa: E402
from lib.models import build_model, pick_device          # noqa: E402
from lib.severstal import (CLASS_IDS, ROOT, TRAIN_IMG_DIR,  # noqa: E402
                           load_index, load_split)

CKPT = os.path.join(ROOT, "results/raw/xp01_ckpt/best.pt")
BASELINE = os.path.join(ROOT, "results/xp01_baseline.json")
OUT_JSON = os.path.join(ROOT, "results/xp03_degradation.json")
SEVERITIES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BS = 4                          # small: full-strip activations share the Jetson's 7 GB


def _load(iid):
    return np.array(Image.open(os.path.join(TRAIN_IMG_DIR, iid)).convert("L"))


@torch.no_grad()
def score(model, ids, labels, ops, kind, severity, device):
    """One full drifted-holdout pass -> clean-vs-defect confusion counts.

    Images are loaded per batch (not preloaded): a 700 MB image cache would compete with
    the U-Net's activations in the Jetson's unified memory and OOM the inference.
    """
    thr = np.array([ops[c]["threshold"] for c in CLASS_IDS])[:, None, None]
    mp = [ops[c]["min_px"] for c in CLASS_IDS]
    tp = fp = fn = tn = 0
    for i in range(0, len(ids), BS):
        batch = range(i, min(i + BS, len(ids)))
        drifted = [drift.apply(_load(ids[j]), kind, severity, seed=j) for j in batch]
        x = torch.from_numpy(np.stack(drifted)).float().div_(255.0).unsqueeze(1).to(device)
        probs = sigmoid(model(x).float().cpu().numpy())              # [b,4,H,W]
        del x
        for k, j in enumerate(batch):
            fired = any((probs[k, c] > thr[c]).sum() >= max(1, mp[c])
                        for c in range(len(CLASS_IDS)))
            true = bool(labels[j])
            tp += fired and true
            fp += fired and not true
            fn += (not fired) and true
            tn += (not fired) and not true
    n = tp + fp + fn + tn
    return {"accuracy": round((tp + tn) / n, 4),
            "recall": round(tp / max(1, tp + fn), 4),
            "specificity": round(tn / max(1, tn + fp), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> int:
    if not os.path.isfile(CKPT):
        print(f"error: no checkpoint at {CKPT}", file=sys.stderr); return 2
    ops = json.load(open(BASELINE))["holdout"]["operating_points"]
    device = pick_device()
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = build_model(encoder=ck["args"]["encoder"], weights=None).to(device).eval()
    model.load_state_dict(ck["model"])

    index = load_index()
    ids = load_split()["holdout"]
    labels = np.array([1 if index[i] else 0 for i in ids])
    print(f"scoring {len(ids)} holdout strips x {len(drift.KINDS)} drifts "
          f"x {len(SEVERITIES)} severities...")

    out = {"experiment": "xp03_drift", "checkpoint": os.path.relpath(CKPT, ROOT),
           "decision": "clean vs defect", "severities": list(SEVERITIES),
           "n_holdout": len(ids), "curves": {}}
    for kind in drift.KINDS:
        out["curves"][kind] = {}
        for s in SEVERITIES:
            m = score(model, ids, labels, ops, kind, s, device)
            out["curves"][kind][f"{s:g}"] = m
            print(f"  {kind:14} sev {s:<4} acc {m['accuracy']:.3f}  "
                  f"recall {m['recall']:.3f}  spec {m['specificity']:.3f}", flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.relpath(OUT_JSON, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
