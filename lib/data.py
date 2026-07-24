"""Torch Dataset for Severstal. Imports torch; keep torch-free helpers in `severstal.py`.

Geometry decision, and the reason for it: Severstal images are 1600x256 — a 6.25:1 strip.
Feeding the full strip to a U-Net wastes most of the compute on defect-free background and
forces a tiny batch. We train on **256x256 crops** and evaluate on the **full strip**,
which a fully-convolutional U-Net handles unchanged (both dimensions are divisible by 32).

Crops are sampled defect-biased during training: a uniform random crop over a 6.25:1 strip
where the median defect covers ~3% of the image would show the model almost nothing but
background, and class 2 — 175 training images of a ~24px-wide streak — would essentially
never appear.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .severstal import H, N_CLASSES, TRAIN_IMG_DIR, W, load_index, load_split, masks_for

CROP = 256


class SeverstalDataset(Dataset):
    """One Severstal fold.

    train=True  -> random 256x256 crop + flips, defect-biased sampling.
    train=False -> the full 256x1600 strip, no augmentation, deterministic.
    """

    def __init__(self, fold: str, train: bool, crop: int = CROP,
                 defect_crop_prob: float = 0.75, seed: int = 0,
                 img_dir: str = TRAIN_IMG_DIR, index: Optional[dict] = None,
                 ids: Optional[list[str]] = None):
        self.ids = ids if ids is not None else load_split()[fold]
        self.index = index if index is not None else load_index()
        self.train, self.crop, self.img_dir = train, crop, img_dir
        self.defect_crop_prob = defect_crop_prob
        self.seed = seed
        self.fold = fold

    def __len__(self) -> int:
        return len(self.ids)

    def _load(self, iid: str) -> tuple[np.ndarray, np.ndarray]:
        # Stored as RGB but visually grayscale; take one channel rather than averaging
        # three identical ones.
        img = np.array(Image.open(os.path.join(self.img_dir, iid)).convert("L"),
                       dtype=np.uint8)
        return img, masks_for(self.index[iid])

    def _pick_x(self, mask: np.ndarray, rng: np.random.Generator) -> int:
        """Left edge of the crop; biased toward a defect column when one exists."""
        max_x = W - self.crop
        cols = np.flatnonzero(mask.any(axis=(0, 1)))
        if cols.size and rng.random() < self.defect_crop_prob:
            c = int(rng.choice(cols))
            return int(np.clip(c - rng.integers(0, self.crop), 0, max_x))
        return int(rng.integers(0, max_x + 1))

    def __getitem__(self, i: int):
        iid = self.ids[i]
        img, mask = self._load(iid)

        if self.train:
            # Seeded per (epoch-agnostic) index so a worker restart is reproducible but
            # crops still vary across epochs via torch's own shuffling of `i`.
            rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**31))
            x0 = self._pick_x(mask, rng)
            img = img[:, x0:x0 + self.crop]
            mask = mask[:, :, x0:x0 + self.crop]
            if rng.random() < 0.5:                      # horizontal flip
                img, mask = img[:, ::-1], mask[:, :, ::-1]
            if rng.random() < 0.5:                      # vertical flip
                img, mask = img[::-1, :], mask[:, ::-1, :]
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

        x = torch.from_numpy(img).float().div_(255.0).unsqueeze(0)   # [1,h,w]
        y = torch.from_numpy(mask)                                    # [4,h,w]
        return x, y, iid


def make_loaders(batch_size: int = 16, workers: int = 4, crop: int = CROP,
                 val_batch: int = 4, seed: int = 0):
    """Train + val loaders. The holdout is deliberately NOT exposed here."""
    index = load_index()
    tr = SeverstalDataset("train", train=True, crop=crop, seed=seed, index=index)
    va = SeverstalDataset("val", train=False, index=index)
    kw = dict(num_workers=workers, pin_memory=False, persistent_workers=workers > 0)
    return (
        torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                    drop_last=True, **kw),
        torch.utils.data.DataLoader(va, batch_size=val_batch, shuffle=False, **kw),
    )
