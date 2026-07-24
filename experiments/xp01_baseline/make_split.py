"""Create and freeze the train / val / holdout split. Run once.

Stratified on the **class combination** ('3', '3+4', 'clean', ...), not on class id: 427
images carry more than one class, and 47% carry none. Stratifying on class id alone would
scatter the multi-class images and let the clean/defective ratio drift between folds,
which would quietly bias every degradation measurement from XP06 on.

The holdout is the project's measurement instrument — it is measured once, at the end, and
never fitted to (PLAN §8). Writing it to disk is what makes that auditable.

    python experiments/xp01_baseline/make_split.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib.severstal import (CLASS_IDS, ROOT, SPLIT_JSON, combo_label,  # noqa: E402
                           load_index)

SEED = 42
FRAC = {"train": 0.70, "val": 0.15, "holdout": 0.15}


def stratified_split(groups: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    """Split each stratum by FRAC, deterministically.

    Done by hand rather than with sklearn for two reasons: no extra dependency on the
    Jetson, and sklearn's stratified split *fails* on strata with fewer members than
    folds — and we have strata of size 1 ('2+4') and 2 ('1+2+3'). Here a tiny stratum
    degrades gracefully: it goes to train, which is where an unmeasurable single example
    is least harmful.
    """
    rng = np.random.default_rng(seed)
    folds: dict[str, list[str]] = {k: [] for k in FRAC}
    for combo in sorted(groups):                      # sorted: order must not depend on dict
        ids = np.array(sorted(groups[combo]))         # sorted: nor on the filesystem
        rng.shuffle(ids)
        n = len(ids)
        n_val = int(round(n * FRAC["val"]))
        n_hold = int(round(n * FRAC["holdout"]))
        # Guarantee train is never starved for a small stratum.
        while n_val + n_hold >= n and (n_val or n_hold):
            if n_hold >= n_val:
                n_hold -= 1
            else:
                n_val -= 1
        folds["val"].extend(ids[:n_val].tolist())
        folds["holdout"].extend(ids[n_val:n_val + n_hold].tolist())
        folds["train"].extend(ids[n_val + n_hold:].tolist())
    return {k: sorted(v) for k, v in folds.items()}


def summarise(fold_ids: list[str], index: dict) -> dict:
    combos = collections.Counter(combo_label(index[i]) for i in fold_ids)
    per_class = {c: sum(1 for i in fold_ids if c in index[i]) for c in CLASS_IDS}
    clean = combos.get("clean", 0)
    return {
        "n": len(fold_ids),
        "clean": clean,
        "defective": len(fold_ids) - clean,
        "pct_clean": round(100 * clean / max(1, len(fold_ids)), 2),
        "per_class_images": per_class,
        "combinations": dict(combos.most_common()),
    }


def main() -> int:
    index = load_index()
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for iid, rles in index.items():
        groups[combo_label(rles)].append(iid)

    folds = stratified_split(groups, SEED)

    # --- invariants. A silently broken split poisons every downstream number.
    all_ids = [i for f in folds.values() for i in f]
    assert len(all_ids) == len(index), f"{len(all_ids)} split vs {len(index)} images"
    assert len(set(all_ids)) == len(all_ids), "an image landed in two folds"
    for a in ("train", "val", "holdout"):
        for b in ("train", "val", "holdout"):
            if a < b:
                assert not (set(folds[a]) & set(folds[b])), f"{a}/{b} overlap"

    out = {
        "experiment": "xp01_baseline",
        "artifact": "split",
        "seed": SEED,
        "fractions": FRAC,
        "stratified_on": "class combination (multi-label), including 'clean'",
        "note": ("The holdout is measured once and never fitted to. lib.severstal."
                 "assert_not_holdout() guards any path that could violate that."),
        "summary": {f: summarise(folds[f], index) for f in ("train", "val", "holdout")},
        "folds": folds,
    }
    os.makedirs(os.path.dirname(SPLIT_JSON), exist_ok=True)
    with open(SPLIT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'fold':<9} {'n':>6} {'clean':>6} {'%clean':>7}  per-class images")
    for f in ("train", "val", "holdout"):
        s = out["summary"][f]
        pc = " ".join(f"c{c}={s['per_class_images'][c]:>4}" for c in CLASS_IDS)
        print(f"{f:<9} {s['n']:>6} {s['clean']:>6} {s['pct_clean']:>6.1f}%  {pc}")
    print(f"\nwrote {os.path.relpath(SPLIT_JSON, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
