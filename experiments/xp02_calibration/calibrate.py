"""XP02 — calibration of the clean-vs-defect decision. No retraining.

    python experiments/xp02_calibration/calibrate.py

For each strip the model produces a single **certainty that the strip has a defect**. We
ask, across the frozen test set: of the strips it was ~C certain about, how many actually
had a defect? A trustworthy model lands on the diagonal (certainty = reality).

The defect certainty is the strength of the strongest defect evidence on the strip — the
average of the TOP-K most defect-like pixels (max over the four class channels), squashed
to 0..1. Clean strips score low, defect strips score high; false alarms are clean strips
that score high.

Temperature scaling fits ONE number on validation (a calibration transform, not a weight
update) and is applied unchanged to the test set. Writes results/xp02_calibration.json and
results/raw/xp02_reliability.npz.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib import calibration as cal                        # noqa: E402
from lib.data import SeverstalDataset                     # noqa: E402
from lib.models import build_model, pick_device           # noqa: E402
from lib.severstal import ROOT, load_index                # noqa: E402

CKPT = os.path.join(ROOT, "results/raw/xp01_ckpt/best.pt")
OUT_JSON = os.path.join(ROOT, "results/xp02_calibration.json")
OUT_NPZ = os.path.join(ROOT, "results/raw/xp02_reliability.npz")
TOPK = 200                             # pixels of strongest defect evidence to average


@torch.no_grad()
def collect(model, fold, device, bs=8, workers=4):
    """Per strip: (defect-evidence logit, has-defect label)."""
    ds = SeverstalDataset(fold, train=False, index=load_index())
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    logit, label = [], []
    for i, (x, y, _) in enumerate(dl):
        logits = model(x.to(device)).float().cpu().numpy()        # [B,4,H,W]
        t = y.numpy()
        for b in range(logits.shape[0]):
            chan_max = logits[b].max(axis=0).ravel()              # strongest class per pixel
            top = np.partition(chan_max, -TOPK)[-TOPK:]
            logit.append(float(top.mean()))
            label.append(int((t[b] > 0.5).any()))
        if i % 50 == 0:
            print(f"  {fold} [{i}/{len(dl)}]", flush=True)
    return np.asarray(logit), np.asarray(label)


def main() -> int:
    if not os.path.isfile(CKPT):
        print(f"error: no checkpoint at {CKPT}", file=sys.stderr); return 2
    device = pick_device()
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = build_model(encoder=ck["args"]["encoder"], weights=None).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {os.path.relpath(CKPT, ROOT)} (epoch {ck.get('epoch')}) on {device.type}")

    print("collecting val (to fit temperature)...")
    vlogit, vlabel = collect(model, "val", device)
    print("collecting holdout (measured once)...")
    hlogit, hlabel = collect(model, "holdout", device)

    T = cal.fit_temperature(vlogit, vlabel)
    raw = cal.sigmoid(hlogit)
    calib = cal.sigmoid(hlogit / T)

    result = {
        "experiment": "xp02_calibration",
        "checkpoint": os.path.relpath(CKPT, ROOT),
        "decision": "clean vs defect (any of the four classes)",
        "note": "No retraining. Temperature fit on val, applied to holdout.",
        "temperature": T,
        "n_holdout": int(len(hlabel)),
        "defect_rate": round(float(hlabel.mean()), 4),
        "raw": {**cal.summary(raw, hlabel),
                "reliability": cal.reliability_bins(raw, hlabel)},
        "calibrated": {**cal.summary(calib, hlabel),
                       "reliability": cal.reliability_bins(calib, hlabel)},
    }
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, conf_raw=raw, conf_cal=calib, label=hlabel)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    r, c = result["raw"], result["calibrated"]
    print("\n=== CLEAN vs DEFECT calibration (frozen holdout) ===")
    print(f"  temperature T = {T}")
    print(f"  raw       : says {r['mean_confidence']:.0%} defect on avg, "
          f"real defect rate {r['accuracy']:.0%}, ECE {r['ece']:.3f}")
    print(f"  calibrated: says {c['mean_confidence']:.0%} defect on avg, "
          f"ECE {c['ece']:.3f}")
    print(f"\nwrote {os.path.relpath(OUT_JSON, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
