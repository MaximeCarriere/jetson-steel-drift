"""Profile the Severstal training set — the numbers XP01's split and loss design rest on.

Severstal never published what ClassId 1-4 *mean*; the classes are anonymous by design.
So we characterise them the only honest way available: by measuring their geometry from
the labelled masks. Writes results/xp01_data_profile.json.

    python experiments/xp01_baseline/profile_data.py
"""
from __future__ import annotations

import collections
import csv
import json
import os
import sys

import numpy as np

H, W = 256, 1600                      # every Severstal image; verified, not assumed
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA = os.path.join(ROOT, "data", "severstal")
OUT = os.path.join(ROOT, "results", "xp01_data_profile.json")


def decode_rle(rle: str) -> np.ndarray:
    """RLE -> bool mask [H, W]. Pixels run top-to-bottom then left-to-right (column-major)."""
    flat = np.zeros(H * W, dtype=bool)
    nums = rle.split()
    starts = np.asarray(nums[0::2], dtype=np.int64) - 1      # the format is 1-based
    lengths = np.asarray(nums[1::2], dtype=np.int64)
    for s, l in zip(starts, lengths):
        flat[s:s + l] = True
    return flat.reshape((W, H)).T                            # column-major -> [H, W]


def main() -> int:
    csv_path = os.path.join(DATA, "train.csv")
    img_dir = os.path.join(DATA, "train_images")
    if not os.path.isfile(csv_path):
        print(f"error: {csv_path} not found — see data/download.md", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(open(csv_path)))
    on_disk = {f for f in os.listdir(img_dir) if f.endswith(".jpg")}

    per_class: dict[str, list[dict]] = collections.defaultdict(list)
    img_classes: dict[str, set[str]] = collections.defaultdict(set)

    for r in rows:
        cid, iid = r["ClassId"], r["ImageId"]
        img_classes[iid].add(cid)
        m = decode_rle(r["EncodedPixels"])
        ys, xs = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
        per_class[cid].append({
            "area": int(m.sum()),
            "bbox_h": int(ys.max() - ys.min() + 1),
            "bbox_w": int(xs.max() - xs.min() + 1),
            # How much of its own bounding box the defect fills: near 1.0 means a solid
            # blob, low means a thin/wispy shape that a coarse model will smear over.
            "fill": float(m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))),
        })

    def stats(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        return {
            "min": round(float(a.min()), 4), "p25": round(float(np.percentile(a, 25)), 4),
            "median": round(float(np.median(a)), 4),
            "p75": round(float(np.percentile(a, 75)), 4),
            "max": round(float(a.max()), 4), "mean": round(float(a.mean()), 4),
        }

    classes = {}
    for cid in sorted(per_class):
        inst = per_class[cid]
        areas = [i["area"] for i in inst]
        classes[cid] = {
            "instances": len(inst),
            "images": len({i for i, cs in img_classes.items() if cid in cs}),
            "share_of_annotations_pct": round(100 * len(inst) / len(rows), 2),
            "total_defect_px": int(sum(areas)),
            "area_px": stats(areas),
            "area_pct_of_image": stats([a / (H * W) * 100 for a in areas]),
            "bbox_h_px": stats([i["bbox_h"] for i in inst]),
            "bbox_w_px": stats([i["bbox_w"] for i in inst]),
            "bbox_fill_ratio": stats([i["fill"] for i in inst]),
        }

    combos = collections.Counter(
        "+".join(sorted(cs)) for cs in img_classes.values())
    co = {a: {b: 0 for b in "1234"} for a in "1234"}
    for cs in img_classes.values():
        for a in cs:
            for b in cs:
                co[a][b] += 1

    defect_imgs = set(img_classes)
    profile = {
        "experiment": "xp01_baseline",
        "artifact": "data_profile",
        "dataset": "Severstal Steel Defect Detection (train split)",
        "image_shape_hw": [H, W],
        "note": ("Severstal never published the semantic meaning of ClassId 1-4. The "
                 "classes are anonymous; everything below is measured geometry, not a "
                 "defect taxonomy."),
        "totals": {
            "train_images_on_disk": len(on_disk),
            "images_with_defect": len(defect_imgs),
            "images_without_defect": len(on_disk - defect_imgs),
            "pct_without_defect": round(100 * len(on_disk - defect_imgs) / len(on_disk), 2),
            "annotation_rows": len(rows),
            "images_with_multiple_classes": sum(1 for cs in img_classes.values() if len(cs) > 1),
        },
        "classes": classes,
        "class_combinations": dict(combos.most_common()),
        "co_occurrence_images": co,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(profile, f, indent=2)

    t = profile["totals"]
    print(f"images {t['train_images_on_disk']}  defective {t['images_with_defect']}  "
          f"clean {t['images_without_defect']} ({t['pct_without_defect']}%)")
    print(f"{'cls':>4} {'inst':>6} {'imgs':>6} {'med area':>9} {'med %img':>9} "
          f"{'med w':>7} {'med h':>7} {'fill':>6}")
    for cid, c in classes.items():
        print(f"{cid:>4} {c['instances']:>6} {c['images']:>6} "
              f"{c['area_px']['median']:>9.0f} {c['area_pct_of_image']['median']:>9.2f} "
              f"{c['bbox_w_px']['median']:>7.0f} {c['bbox_h_px']['median']:>7.0f} "
              f"{c['bbox_fill_ratio']['median']:>6.2f}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
