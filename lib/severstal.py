"""Severstal data access — RLE codec, the image index, and the frozen split.

Shared by every experiment from XP01 on, so that "the holdout" means exactly one thing
across the whole project. Nothing here imports torch; the torch Dataset lives in
`lib/data.py` so that analysis-only experiments stay importable without it.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Iterator, Optional

import numpy as np

H, W = 256, 1600           # every Severstal image, verified on disk
N_CLASSES = 4
CLASS_IDS = ("1", "2", "3", "4")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT, "data", "severstal")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SPLIT_JSON = os.path.join(ROOT, "results", "xp01_split.json")


# --------------------------------------------------------------------------- RLE
def rle_decode(rle: str, h: int = H, w: int = W) -> np.ndarray:
    """RLE string -> bool mask [h, w].

    Severstal numbers pixels top-to-bottom then left-to-right — i.e. **column-major**,
    1-based. Decoding row-major instead produces a mask that looks plausible and is
    transposed nonsense, which is the classic silent bug on this dataset.
    """
    flat = np.zeros(h * w, dtype=bool)
    if not rle or not rle.strip():
        return flat.reshape((w, h)).T
    nums = rle.split()
    starts = np.asarray(nums[0::2], dtype=np.int64) - 1
    lengths = np.asarray(nums[1::2], dtype=np.int64)
    for s, ln in zip(starts, lengths):
        flat[s:s + ln] = True
    return flat.reshape((w, h)).T


def rle_encode(mask: np.ndarray) -> str:
    """bool mask [h, w] -> RLE string. Inverse of `rle_decode`."""
    flat = np.asarray(mask, dtype=bool).T.reshape(-1)          # back to column-major
    padded = np.concatenate(([False], flat, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    starts, ends = changes[0::2], changes[1::2]
    return " ".join(f"{s} {e - s}" for s, e in zip(starts, ends))


# --------------------------------------------------------------------------- index
def load_index(csv_path: str = TRAIN_CSV,
               img_dir: str = TRAIN_IMG_DIR) -> dict[str, dict[str, str]]:
    """{image_id: {class_id: rle}}. Images with no defect map to an empty dict.

    Defect-free images are **absent from train.csv entirely** — 5,902 of them — so they
    have to be recovered from the directory listing. Trusting the CSV alone silently
    throws away 47% of the data.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{csv_path} not found — see data/download.md")
    index: dict[str, dict[str, str]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            index.setdefault(row["ImageId"], {})[row["ClassId"]] = row["EncodedPixels"]
    for name in sorted(os.listdir(img_dir)):
        if name.endswith(".jpg"):
            index.setdefault(name, {})
    return index


def combo_label(classes: dict | set | list) -> str:
    """The stratification key: '3', '3+4', or 'clean'. Order-independent."""
    ids = sorted(classes)
    return "+".join(ids) if ids else "clean"


def masks_for(rles: dict[str, str], h: int = H, w: int = W) -> np.ndarray:
    """{class_id: rle} -> float32 [4, h, w], channel c = ClassId c+1."""
    out = np.zeros((N_CLASSES, h, w), dtype=np.float32)
    for i, cid in enumerate(CLASS_IDS):
        if cid in rles:
            out[i] = rle_decode(rles[cid], h, w)
    return out


# --------------------------------------------------------------------------- split
def load_split(path: str = SPLIT_JSON) -> dict[str, list[str]]:
    """{'train': [...], 'val': [...], 'holdout': [...]}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found — run experiments/xp01_baseline/make_split.py first")
    with open(path) as f:
        d = json.load(f)
    return {k: d["folds"][k] for k in ("train", "val", "holdout")}


def iter_fold(fold: str, path: str = SPLIT_JSON) -> Iterator[tuple[str, dict[str, str]]]:
    """Yield (image_id, {class_id: rle}) for one fold, in split order."""
    index = load_index()
    for iid in load_split(path)[fold]:
        yield iid, index[iid]


def assert_not_holdout(ids: list[str] | set[str], where: str,
                       path: str = SPLIT_JSON) -> None:
    """Guard for any tuning path. The holdout is measured once, never fitted to (PLAN §8).

    Cheap insurance against the one mistake that would invalidate every number in the
    project without leaving a trace in the results.
    """
    leak = set(ids) & set(load_split(path)["holdout"])
    if leak:
        raise RuntimeError(
            f"{where} touched {len(leak)} frozen-holdout images "
            f"(e.g. {sorted(leak)[:3]}). The holdout is never used for tuning.")
