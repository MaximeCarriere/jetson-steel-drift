"""XP01 — fine-tune the U-Net baseline. Target hardware: Jetson Orin Nano Super (8 GB).

    python experiments/xp01_baseline/train.py --epochs 20 --batch-size 12

Only the ImageNet-pretrained encoder is inherited; the decoder is trained from scratch.
Resumable by design — an 8 GB board training for hours should survive a dropped SSH
session, so every epoch writes a checkpoint and `--resume` picks it back up.

The frozen holdout is never loaded here. `evaluate.py` is the only script that touches it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib.data import make_loaders                                    # noqa: E402
from lib.models import (TverskyBCELoss, build_model, dice_per_class,  # noqa: E402
                        image_level_stats, pick_device)
from lib.severstal import CLASS_IDS, ROOT                            # noqa: E402

CKPT_DIR = os.path.join(ROOT, "results", "raw", "xp01_ckpt")
LOG_JSON = os.path.join(ROOT, "results", "xp01_train_log.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XP01 baseline fine-tune")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=12,
                   help="12 fits an 8 GB Orin Nano at 256px with AMP; lower if OOM")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--encoder", default="resnet34")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--resume", action="store_true", help="continue from last.pt")
    p.add_argument("--limit-batches", type=int, default=0,
                   help="smoke test: stop each epoch after N batches")
    return p.parse_args()


@torch.no_grad()
def validate(model, loader, device, limit: int = 0) -> dict:
    """Full-strip validation. Returns per-class Dice + image-level rates."""
    model.eval()
    dice_sum = torch.zeros(len(CLASS_IDS), dtype=torch.float64)
    conf = {k: torch.zeros(len(CLASS_IDS), dtype=torch.long) for k in
            ("tp", "fp", "fn", "tn")}
    n = 0
    for i, (x, y, _) in enumerate(loader):
        if limit and i >= limit:
            break
        x, y = x.to(device), y.to(device)
        probs = torch.sigmoid(model(x)).float()
        dice_sum += dice_per_class(probs, y).sum(0).double().cpu()
        for k, v in image_level_stats(probs, y).items():
            conf[k] += v.cpu()
        n += x.shape[0]

    per_class = (dice_sum / max(1, n)).tolist()
    tp, fp, fn = conf["tp"].double(), conf["fp"].double(), conf["fn"].double()
    rec = tp / (tp + fn).clamp(min=1)
    prec = tp / (tp + fp).clamp(min=1)
    f1 = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    return {
        "n_images": n,
        "dice_per_class": {c: round(d, 4) for c, d in zip(CLASS_IDS, per_class)},
        "dice_mean": round(sum(per_class) / len(per_class), 4),
        "img_recall": {c: round(float(r), 4) for c, r in zip(CLASS_IDS, rec.tolist())},
        "img_precision": {c: round(float(p), 4) for c, p in zip(CLASS_IDS, prec.tolist())},
        # macro F1 gives every class equal say, so a model that abandons c1/c2 scores low —
        # unlike mean Dice, which the run-1 blindness could hide behind the clean freebie.
        "macro_f1": round(float(f1.mean()), 4),
    }


def main() -> int:
    a = parse_args()
    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_loader, val_loader = make_loaders(batch_size=a.batch_size, workers=a.workers,
                                            crop=a.crop, seed=a.seed)
    model = build_model(encoder=a.encoder).to(device)
    crit = TverskyBCELoss(alpha=0.3, beta=0.7)         # beta>alpha: punish misses harder
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    # AMP is CUDA-only here; MPS autocast exists but the GradScaler does not.
    use_amp = a.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best, history = 0, -1.0, []
    last = os.path.join(CKPT_DIR, "last.pt")
    if a.resume and os.path.isfile(last):
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
        print(f"resumed from epoch {start_epoch} (best val dice {best:.4f})")

    print(f"device={device.type}  encoder={a.encoder}  batch={a.batch_size}  "
          f"crop={a.crop}  amp={use_amp}")
    print(f"train batches/epoch={len(train_loader)}  val images={len(val_loader.dataset)}")

    for epoch in range(start_epoch, a.epochs):
        model.train()
        t0, running, seen = time.perf_counter(), 0.0, 0
        for i, (x, y, _) in enumerate(train_loader):
            if a.limit_batches and i >= a.limit_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.shape[0]
            seen += x.shape[0]
            if i % 50 == 0:
                print(f"  e{epoch} [{i}/{len(train_loader)}] loss {running/max(1,seen):.4f}",
                      flush=True)
        sched.step()

        val = validate(model, val_loader, device, limit=a.limit_batches)
        rec = {"epoch": epoch, "train_loss": round(running / max(1, seen), 4),
               "lr": round(sched.get_last_lr()[0], 7),
               "secs": round(time.perf_counter() - t0, 1), **val}
        history.append(rec)
        rc = val["img_recall"]
        print(f"epoch {epoch}: loss {rec['train_loss']:.4f}  macroF1 {val['macro_f1']:.4f}"
              f"  recall c1={rc['1']:.2f} c2={rc['2']:.2f} c3={rc['3']:.2f} c4={rc['4']:.2f}"
              f"  ({rec['secs']:.0f}s)", flush=True)

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best": best, "history": history,
                 "args": vars(a)}
        torch.save(state, last)
        # Select on macro F1, not mean Dice: the run-1 checkpoint "won" by abandoning
        # c1/c2, which macro F1 makes impossible.
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            state["best"] = best
            torch.save(state, os.path.join(CKPT_DIR, "best.pt"))
            print(f"  new best macroF1 {best:.4f}")

        with open(LOG_JSON, "w") as f:
            json.dump({"experiment": "xp01_baseline", "artifact": "train_log",
                       "device": device.type, "args": vars(a), "best_macro_f1": best,
                       "selection_metric": "macro_f1", "history": history}, f, indent=2)

    print(f"\nbest macro F1 {best:.4f}  ->  {os.path.join(CKPT_DIR, 'best.pt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
