"""XP01 — measure the baseline. The only script that touches the frozen holdout.

    python experiments/xp01_baseline/evaluate.py --ckpt results/raw/xp01_ckpt/best.pt

Post-processing (probability threshold + a minimum-blob-size floor) is tuned on **val**,
then applied unchanged to the holdout, which is measured exactly once. Tuning on the
holdout would inflate every number here and silently corrupt every drift measurement from
XP06 on, since those are all deltas against this baseline (PLAN §8).

Metrics accumulate per batch across the whole (threshold x min_px) grid at once. Holding
the raw probabilities instead would need 1884 x 4 x 256 x 1600 x 4 B = **12 GB** on a
board with 8 — the obvious implementation does not fit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np                                                   # noqa: E402

from lib.data import SeverstalDataset                                # noqa: E402
from lib.models import (build_model, dice_per_class,                 # noqa: E402
                        image_level_stats, pick_device)
from lib.severstal import CLASS_IDS, ROOT, load_index, load_split    # noqa: E402

OUT_JSON = os.path.join(ROOT, "results", "xp01_baseline.json")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
MIN_PX = (0, 200, 600, 1200, 2000)
NC = len(CLASS_IDS)


class GridAccumulator:
    """Running metrics for every (threshold, min_px) pair, per class.

    Tracks three things per cell, because they answer different questions and — the whole
    lesson of this experiment — the wrong one is quietly misleading:

      * `dice_all`   Kaggle mean-Dice over ALL images (clean ones score the empty=1.0
                     freebie). Reported for comparability; NEVER tuned on, because with
                     47% clean images it rewards a model for staying silent on rare
                     classes.
      * `dice_pos`   Dice over images that actually CONTAIN the class — segmentation
                     quality where it matters, with no freebie.
      * confusion    image-level TP/FP/FN/TN, from which detection F1 is derived. F1 is
                     the tuning objective: it is 0 when recall is 0, so it structurally
                     cannot be maximised by abandoning a class.
    """

    def __init__(self, thresholds=THRESHOLDS, min_px=MIN_PX):
        self.keys = [(t, m) for t in thresholds for m in min_px]
        self.dice_all = {k: torch.zeros(NC, dtype=torch.float64) for k in self.keys}
        self.dice_pos = {k: torch.zeros(NC, dtype=torch.float64) for k in self.keys}
        self.pos_n = torch.zeros(NC, dtype=torch.float64)
        self.conf = {k: {c: torch.zeros(NC, dtype=torch.long)
                         for c in ("tp", "fp", "fn", "tn")} for k in self.keys}
        self.n = 0
        self._counted_pos = False

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        self.n += probs.shape[0]
        has = targets.bool().any(dim=(2, 3)).double()          # [B, 4] positive images
        self.pos_n += has.sum(0)
        for k in self.keys:
            th, mp = k
            d = dice_per_class(probs, targets, th, mp)          # [B, 4]
            self.dice_all[k] += d.sum(0).double()
            self.dice_pos[k] += (d * has).sum(0).double()
            for name, v in image_level_stats(probs, targets, th, mp).items():
                self.conf[k][name] += v

    def score(self, key) -> dict:
        dice_all = self.dice_all[key] / max(1, self.n)
        dice_pos = self.dice_pos[key] / self.pos_n.clamp(min=1)
        c = self.conf[key]
        tp, fp, fn, tn = (c[k].double() for k in ("tp", "fp", "fn", "tn"))
        rec = tp / (tp + fn).clamp(min=1)
        prec = tp / (tp + fp).clamp(min=1)
        f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
        r = lambda t: {cid: round(float(v), 4) for cid, v in zip(CLASS_IDS, t)}  # noqa: E731
        return {
            "dice_kaggle_mean": round(float(dice_all.mean()), 4),
            "dice_kaggle_per_class": r(dice_all),
            "dice_defectonly_per_class": r(dice_pos),
            "img_f1": r(f1), "img_recall": r(rec), "img_precision": r(prec),
            "img_fp_rate": r(fp / (fp + tn).clamp(min=1)),
        }

    def best_per_class(self) -> dict:
        """Per class, the (thresh, min_px) that maximises that class's detection F1."""
        out = {}
        for i, cid in enumerate(CLASS_IDS):
            out[cid] = max(self.keys, key=lambda k: self.score(k)["img_f1"][cid])
        return out

    def best_mean_kaggle(self):
        """The single global point that maxes Kaggle mean-Dice — kept only to SHOW the
        trap: it is what a naive tuner picks, and it abandons the rare classes."""
        return max(self.keys, key=lambda k: self.score(k)["dice_kaggle_mean"])

    def score_per_class(self, per_class_keys: dict) -> dict:
        """Assemble a report where each class uses its OWN operating point."""
        r = {"operating_points": {c: {"threshold": k[0], "min_px": k[1]}
                                  for c, k in per_class_keys.items()}}
        for field in ("dice_kaggle_per_class", "dice_defectonly_per_class",
                      "img_f1", "img_recall", "img_precision", "img_fp_rate"):
            r[field] = {c: self.score(k)[field][c] for c, k in per_class_keys.items()}
        conf = {}
        for c, k in per_class_keys.items():
            i = CLASS_IDS.index(c)
            conf[c] = {n: int(self.conf[k][n][i]) for n in ("tp", "fp", "fn", "tn")}
        r["confusion"] = conf
        r["dice_defectonly_mean"] = round(
            sum(r["dice_defectonly_per_class"].values()) / NC, 4)
        r["img_f1_mean"] = round(sum(r["img_f1"].values()) / NC, 4)
        return r


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(ROOT, "results/raw/xp01_ckpt/best.pt"))
    p.add_argument("--encoder", default=None, help="default: read from the checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="smoke test: N batches")
    return p.parse_args()


class CleanFalseAlarm:
    """'Cries wolf on good steel': of genuinely-clean images, how many does the model wrongly
    flag with a defect? The number a plant cares about most, isolated from other-defect
    confusion — computed with each class's own chosen operating point.
    """

    def __init__(self, per_class_keys: dict):
        self.keys = per_class_keys
        self.n_clean = 0
        self.any_fired = 0
        self.per_class = {c: 0 for c in CLASS_IDS}

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        clean = ~targets.bool().any(dim=(1, 2, 3))            # [B] truly-clean images
        if int(clean.sum()) == 0:
            return
        pc = probs[clean]
        self.n_clean += pc.shape[0]
        fired_any = torch.zeros(pc.shape[0], dtype=torch.bool)
        for i, c in enumerate(CLASS_IDS):
            thr, mp = self.keys[c]
            area = (pc[:, i] > thr).sum(dim=(1, 2))
            fired = area >= max(1, mp)
            self.per_class[c] += int(fired.sum())
            fired_any |= fired
        self.any_fired += int(fired_any.sum())

    def report(self) -> dict:
        n = max(1, self.n_clean)
        return {
            "n_clean_images": self.n_clean,
            "any_defect_false_alarm_rate": round(self.any_fired / n, 4),
            "per_class_false_alarm_rate": {c: round(self.per_class[c] / n, 4)
                                           for c in CLASS_IDS},
        }


class Confusion:
    """Image-level confusion matrices, each class using its own operating point:

      * presence/absence — a 2x2 over {clean, defect}: does the model get "is there ANY
        defect?" right, regardless of which class.
      * class 5x5 — over {clean, c1, c2, c3, c4}. Multi-label images (~6%) are reduced to
        their single largest-area class (true) and the model's largest fired class (pred),
        so this is a standard confusion the % view can read row-wise.
    """

    LABELS = ["clean", *CLASS_IDS]

    def __init__(self, per_class_keys: dict):
        self.keys = per_class_keys
        self.binary = np.zeros((2, 2), dtype=np.int64)      # [true][pred], 0=clean 1=defect
        self.cls = np.zeros((5, 5), dtype=np.int64)         # [true][pred] over LABELS

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        p = probs.numpy()
        t = targets.numpy()
        for b in range(p.shape[0]):
            t_area = [float(t[b, i].sum()) for i in range(len(CLASS_IDS))]
            p_area = []
            for i, c in enumerate(CLASS_IDS):
                thr, mp = self.keys[c]
                a = float((p[b, i] > thr).sum())
                p_area.append(a if a >= max(1, mp) else 0.0)
            true_def = any(v > 0 for v in t_area)
            pred_def = any(v > 0 for v in p_area)
            self.binary[int(true_def), int(pred_def)] += 1
            ti = 1 + int(np.argmax(t_area)) if true_def else 0
            pi = 1 + int(np.argmax(p_area)) if pred_def else 0
            self.cls[ti, pi] += 1

    def report(self) -> dict:
        return {
            "presence_absence": {"labels": ["clean", "defect"],
                                 "counts": self.binary.tolist()},
            "class_confusion": {"labels": self.LABELS, "counts": self.cls.tolist()},
        }


@torch.no_grad()
def run_fold(model, fold: str, device, a, grid: GridAccumulator,
             clean_fa: "CleanFalseAlarm | None" = None,
             confusion: "Confusion | None" = None) -> GridAccumulator:
    ds = SeverstalDataset(fold, train=False, index=load_index())
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                                     num_workers=a.workers)
    for i, (x, y, _) in enumerate(dl):
        if a.limit and i >= a.limit:
            break
        probs = torch.sigmoid(model(x.to(device))).float().cpu()
        grid.update(probs, y)
        if clean_fa is not None:
            clean_fa.update(probs, y)
        if confusion is not None:
            confusion.update(probs, y)
        if i % 50 == 0:
            print(f"  {fold} [{i}/{len(dl)}]", flush=True)
    return grid


def main() -> int:
    a = parse_args()
    device = pick_device(a.device)
    if not os.path.isfile(a.ckpt):
        print(f"error: no checkpoint at {a.ckpt} — run train.py first", file=sys.stderr)
        return 2

    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    encoder = a.encoder or ck.get("args", {}).get("encoder", "resnet34")
    model = build_model(encoder=encoder, weights=None).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {os.path.relpath(a.ckpt, ROOT)} (epoch {ck.get('epoch')}, "
          f"val dice {ck.get('best', float('nan')):.4f}) encoder={encoder} "
          f"device={device.type}")

    # ---- 1. tune per-class operating points on VAL only, by detection F1
    print("\ntuning per-class operating points on val (objective: detection F1)...")
    vgrid = run_fold(model, "val", device, a, GridAccumulator())
    per_class_keys = vgrid.best_per_class()
    naive_key = vgrid.best_mean_kaggle()          # the trap, for comparison
    for c, k in per_class_keys.items():
        s = vgrid.score(k)
        print(f"  class {c}: thr={k[0]} min_px={k[1]:<4} "
              f"F1={s['img_f1'][c]:.3f} recall={s['img_recall'][c]:.3f} "
              f"defect-Dice={s['dice_defectonly_per_class'][c]:.3f}")
    print(f"  [naive mean-Dice tuner would pick {naive_key} for ALL classes -> "
          f"c1/c2 recall {vgrid.score(naive_key)['img_recall']['1']:.2f}/"
          f"{vgrid.score(naive_key)['img_recall']['2']:.2f}]")

    # ---- 2. measure the holdout ONCE, per-class points applied unchanged
    print("\nmeasuring frozen holdout (once, no tuning)...")
    cfa = CleanFalseAlarm(per_class_keys)
    cm = Confusion(per_class_keys)
    hgrid = run_fold(model, "holdout", device, a, GridAccumulator(), clean_fa=cfa,
                     confusion=cm)
    val_report = vgrid.score_per_class(per_class_keys)
    hold_report = hgrid.score_per_class(per_class_keys)
    hold_report["clean_false_alarm"] = cfa.report()
    hold_report.update(cm.report())                       # presence_absence + class_confusion
    naive_hold = hgrid.score(naive_key)

    split = load_split()
    out = {
        "experiment": "xp01_baseline",
        "artifact": "baseline_metrics",
        "checkpoint": os.path.relpath(a.ckpt, ROOT),
        "encoder": encoder,
        "epoch": ck.get("epoch"),
        "device": device.type,
        "tuning": {
            "objective": "per-class image-level detection F1, tuned on val",
            "grid": {"thresholds": list(THRESHOLDS), "min_px": list(MIN_PX)},
            "why_not_mean_dice": (
                "47% of images are clean and score the empty=1.0 Dice freebie, so mean "
                "Dice is maximised by suppressing predictions on rare classes. The naive "
                "mean-Dice point below abandons c1 and c2 (recall 0). F1 cannot, since "
                "F1=0 when recall=0."),
        },
        "val": {"n_images": vgrid.n, **val_report},
        "holdout": {"n_images": hgrid.n, **hold_report},
        "naive_mean_dice_trap": {
            "operating_point": {"threshold": naive_key[0], "min_px": naive_key[1]},
            "holdout_dice_kaggle_mean": naive_hold["dice_kaggle_mean"],
            "holdout_recall": naive_hold["img_recall"],
            "note": ("This is what tuning a single global threshold on mean Dice gives: "
                     "a headline 0.92 that is really a c3/c4 detector blind to c1/c2."),
        },
        "caveat": ("Class 2 has only 36 defective holdout images, so its per-class "
                   "numbers are noisy; read the trend, not the digit."),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== FROZEN HOLDOUT (per-class operating points) ===")
    print(f"{'cls':>4} {'F1':>6} {'recall':>7} {'prec':>6} {'defDice':>8} {'kagDice':>8}")
    for c in CLASS_IDS:
        h = out["holdout"]
        print(f"{c:>4} {h['img_f1'][c]:>6.3f} {h['img_recall'][c]:>7.3f} "
              f"{h['img_precision'][c]:>6.3f} "
              f"{h['dice_defectonly_per_class'][c]:>8.3f} "
              f"{h['dice_kaggle_per_class'][c]:>8.3f}")
    print(f"mean  F1={out['holdout']['img_f1_mean']:.3f}   "
          f"defect-Dice={out['holdout']['dice_defectonly_mean']:.3f}")
    fa = out["holdout"]["clean_false_alarm"]
    print(f"clean false-alarm: {fa['any_defect_false_alarm_rate']:.3f} of "
          f"{fa['n_clean_images']} clean strips wrongly flagged  "
          f"(per class {fa['per_class_false_alarm_rate']})")
    print(f"\nwrote {os.path.relpath(OUT_JSON, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
