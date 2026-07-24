"""XP02 — calibration: when the model predicts class X, can we trust its confidence?

    python experiments/xp02_calibration/calibrate.py

**No retraining.** Loads the XP01 model, runs it on val + holdout, and analyses the
confidence numbers it already produces. Temperature scaling fits ONE scalar per class on
val (a calibration transform, not a weight update) and is applied unchanged to holdout.

Two calibration views, both per class:

  * **prediction-level** (operator-facing, the headline): for each image where the model
    fires class c, certainty = the confidence of that firing; correct = the image really
    has class c. Answers "when it flags class X at C% certainty, is it right C% of the
    time?".
  * **pixel-level**: the classic segmentation view, over a pixel subsample.

Writes results/xp02_calibration.json and results/raw/xp02_reliability.npz.
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
from lib.severstal import CLASS_IDS, ROOT, load_index, load_split  # noqa: E402

CKPT = os.path.join(ROOT, "results/raw/xp01_ckpt/best.pt")
BASELINE = os.path.join(ROOT, "results/xp01_baseline.json")
OUT_JSON = os.path.join(ROOT, "results/xp02_calibration.json")
OUT_NPZ = os.path.join(ROOT, "results/raw/xp02_reliability.npz")
PIX_PER_IMG = 40                       # sampled foreground + background pixels, per class


@torch.no_grad()
def collect(model, fold, device, ops, bs=8, workers=4):
    """Run a fold once; return per-class prediction-level and pixel-level (logit, label)."""
    ds = SeverstalDataset(fold, train=False, index=load_index())
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    pred = {c: {"logit": [], "correct": []} for c in CLASS_IDS}   # fired image-classes
    pix = {c: {"logit": [], "label": []} for c in CLASS_IDS}      # sampled pixels
    rng = np.random.default_rng(0)
    for i, (x, y, _) in enumerate(dl):
        logits = model(x.to(device)).float().cpu().numpy()        # [B,4,H,W]
        probs = cal.sigmoid(logits)
        t = y.numpy()
        for b in range(logits.shape[0]):
            for ci, c in enumerate(CLASS_IDS):
                lg, pr, tm = logits[b, ci], probs[b, ci], t[b, ci] > 0.5
                present = bool(tm.any())
                thr, mp = ops[c]["threshold"], ops[c]["min_px"]
                fired = pr > thr
                if int(fired.sum()) >= max(1, mp):                # model claims class c
                    pred[c]["logit"].append(float(lg[fired].mean()))
                    pred[c]["correct"].append(int(present))
                # pixel sample: foreground (true) + random background
                fg = np.flatnonzero(tm.ravel())
                bg = np.flatnonzero(~tm.ravel())
                take_fg = fg[rng.integers(0, len(fg), min(PIX_PER_IMG, len(fg)))] if len(fg) else []
                take_bg = bg[rng.integers(0, len(bg), PIX_PER_IMG)] if len(bg) else []
                for idx in (*take_fg, *take_bg):
                    pix[c]["logit"].append(float(lg.ravel()[idx]))
                    pix[c]["label"].append(int(tm.ravel()[idx]))
        if i % 50 == 0:
            print(f"  {fold} [{i}/{len(dl)}]", flush=True)
    to_np = lambda d: {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in d.items()}
    return to_np(pred), to_np(pix)


def view_metrics(logit, correct, temperatures=None):
    """Raw + (optionally) temperature-scaled calibration summary for one class/view."""
    raw = cal.sigmoid(logit)
    out = {"raw": cal.summary(raw, correct)}
    if temperatures is not None:
        T = temperatures
        calib = cal.sigmoid(logit / T)
        out["temperature"] = T
        out["calibrated"] = cal.summary(calib, correct)
    return out


def main() -> int:
    if not os.path.isfile(CKPT):
        print(f"error: no checkpoint at {CKPT}", file=sys.stderr); return 2
    ops = json.load(open(BASELINE))["holdout"]["operating_points"]
    device = pick_device()
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = build_model(encoder=ck["args"]["encoder"], weights=None).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {os.path.relpath(CKPT, ROOT)} (epoch {ck.get('epoch')}) on {device.type}")

    print("collecting val (to fit temperature)...")
    vpred, vpix = collect(model, "val", device, ops)
    print("collecting holdout (measured once)...")
    hpred, hpix = collect(model, "holdout", device, ops)

    result = {"experiment": "xp02_calibration", "checkpoint": os.path.relpath(CKPT, ROOT),
              "note": "No retraining. Temperature fit on val, applied to holdout.",
              "prediction_level": {}, "pixel_level": {}}
    npz = {}
    for level, vv, hh in (("prediction_level", vpred, hpred),
                          ("pixel_level", vpix, hpix)):
        key = "correct" if level == "prediction_level" else "label"
        for c in CLASS_IDS:
            vl, vc = vv[c]["logit"], vv[c][key]
            hl, hc = hh[c]["logit"], hh[c][key]
            T = cal.fit_temperature(vl, vc) if len(vl) else 1.0
            result[level][c] = view_metrics(hl, hc, temperatures=T)
            if level == "prediction_level":
                npz[f"pred_{c}_conf_raw"] = cal.sigmoid(hl)
                npz[f"pred_{c}_conf_cal"] = cal.sigmoid(hl / T)
                npz[f"pred_{c}_correct"] = np.asarray(hc)
        # pooled across classes
        pl = np.concatenate([vv[c]["logit"] for c in CLASS_IDS])
        pk = np.concatenate([vv[c][key] for c in CLASS_IDS])
        hpl = np.concatenate([hh[c]["logit"] for c in CLASS_IDS])
        hpk = np.concatenate([hh[c][key] for c in CLASS_IDS])
        Tp = cal.fit_temperature(pl, pk) if len(pl) else 1.0
        result[level]["overall"] = view_metrics(hpl, hpk, temperatures=Tp)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, **npz)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== PREDICTION-LEVEL (when it flags class c, is it right?) ===")
    print(f"{'cls':>4} {'mean_conf':>9} {'accuracy':>9} {'ECE_raw':>8} {'T':>6} {'ECE_cal':>8}")
    for c in [*CLASS_IDS, "overall"]:
        r = result["prediction_level"][c]
        print(f"{c:>4} {r['raw']['mean_confidence']:>9.3f} {r['raw']['accuracy']:>9.3f} "
              f"{r['raw']['ece']:>8.3f} {r['temperature']:>6.2f} "
              f"{r['calibrated']['ece']:>8.3f}")
    print(f"\nwrote {os.path.relpath(OUT_JSON, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
