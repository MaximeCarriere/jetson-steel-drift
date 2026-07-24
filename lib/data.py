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

from .severstal import (CLASS_IDS, H, N_CLASSES, TRAIN_IMG_DIR, W, load_index,
                        load_split, masks_for)

CROP = 256


class SeverstalDataset(Dataset):
    """One Severstal fold.

    train=True  -> 256x256 crop + flips, CLASS-BALANCED sampling (run-2 fix).
    train=False -> the full 256x1600 strip, no augmentation, deterministic.

    Run 1 cropped around "any defect", which is usually class 3 (73% of defects), so the
    model barely saw c1/c2 and never learned them. Run 2 samples a *target category* first
    — clean / c1 / c2 / c3 / c4, with the rare classes boosted far above their natural rate
    — then picks an image that has it and crops around that class. This guarantees the
    model sees every class every epoch, which is the single change no loss tweak can
    substitute for: a class the model never sees cannot be learned.
    """

    def __init__(self, fold: str, train: bool, crop: int = CROP,
                 seed: int = 0, clean_prob: float = 0.35,
                 img_dir: str = TRAIN_IMG_DIR, index: Optional[dict] = None,
                 ids: Optional[list[str]] = None):
        self.ids = ids if ids is not None else load_split()[fold]
        self.index = index if index is not None else load_index()
        self.train, self.crop, self.img_dir = train, crop, img_dir
        self.seed, self.clean_prob, self.fold = seed, clean_prob, fold

        if train:
            # Category -> image ids. Defect lists allow overlaps (a multi-class image
            # appears under each of its classes); that is fine and mildly helpful.
            self.by_cat: dict[str, list[str]] = {c: [] for c in CLASS_IDS}
            self.clean: list[str] = []
            for iid in self.ids:
                cs = self.index[iid]
                if not cs:
                    self.clean.append(iid)
                for c in cs:
                    self.by_cat[c].append(iid)
            # 35% clean, remaining 65% split EQUALLY across the four defect classes.
            defect_p = (1.0 - clean_prob) / len(CLASS_IDS)
            self.cats = ["clean", *CLASS_IDS]
            self.cat_p = [clean_prob, *([defect_p] * len(CLASS_IDS))]

    def __len__(self) -> int:
        return len(self.ids)

    def _load(self, iid: str) -> tuple[np.ndarray, np.ndarray]:
        # Stored as RGB but visually grayscale; take one channel rather than averaging
        # three identical ones.
        img = np.array(Image.open(os.path.join(self.img_dir, iid)).convert("L"),
                       dtype=np.uint8)
        return img, masks_for(self.index[iid])

    def _crop_around(self, mask: np.ndarray, cid: Optional[str],
                     rng: np.random.Generator) -> int:
        """Left edge of the crop, centred on class `cid`'s columns (random if clean)."""
        max_x = W - self.crop
        if cid is not None:
            cols = np.flatnonzero(mask[CLASS_IDS.index(cid)].any(axis=0))
            if cols.size:
                c = int(rng.choice(cols))
                return int(np.clip(c - rng.integers(0, self.crop), 0, max_x))
        return int(rng.integers(0, max_x + 1))

    def __getitem__(self, i: int):
        # torch.initial_seed() differs per worker and per epoch, so crops vary across
        # epochs while staying reproducible within a run.
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**31))

        if self.train:
            cat = self.cats[int(rng.choice(len(self.cats), p=self.cat_p))]
            pool = self.clean if cat == "clean" else self.by_cat[cat]
            if not pool:                                    # empty class -> fall back
                pool, cat = self.clean or self.ids, None
            iid = str(rng.choice(pool))
            img, mask = self._load(iid)
            x0 = self._crop_around(mask, None if cat == "clean" else cat, rng)
            img = img[:, x0:x0 + self.crop]
            mask = mask[:, :, x0:x0 + self.crop]
            if rng.random() < 0.5:                          # horizontal flip
                img, mask = img[:, ::-1], mask[:, :, ::-1]
            if rng.random() < 0.5:                          # vertical flip
                img, mask = img[::-1, :], mask[:, ::-1, :]
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        else:
            iid = self.ids[i]
            img, mask = self._load(iid)

        x = torch.from_numpy(img).float().div_(255.0).unsqueeze(0)   # [1,h,w]
        y = torch.from_numpy(mask)                                    # [4,h,w]
        return x, y, iid


def make_loaders(batch_size: int = 16, workers: int = 4, crop: int = CROP,
                 val_batch: int = 4, seed: int = 0, clean_prob: float = 0.35):
    """Train + val loaders. The holdout is deliberately NOT exposed here."""
    index = load_index()
    tr = SeverstalDataset("train", train=True, crop=crop, seed=seed, index=index,
                          clean_prob=clean_prob)
    va = SeverstalDataset("val", train=False, index=index)
    kw = dict(num_workers=workers, pin_memory=False, persistent_workers=workers > 0)
    return (
        # shuffle=True still varies which index hits which worker RNG; the balanced
        # sampler makes the *content* class-balanced regardless.
        torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True,
                                    drop_last=True, **kw),
        torch.utils.data.DataLoader(va, batch_size=val_batch, shuffle=False, **kw),
    )
