"""Model definition, loss, and metrics.

A U-Net with an ImageNet-pretrained encoder from segmentation_models.pytorch (MIT). Not
trained from scratch and not an ensemble: XP02 has to export ONE engine to TensorRT and
XP12 needs a single-model latency figure to measure monitoring overhead against.

The head emits **four independent sigmoid channels**, not a 5-way softmax — 427 training
images carry more than one defect class, which a softmax would assert is impossible.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .severstal import N_CLASSES

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"


def build_model(encoder: str = ENCODER, weights: Optional[str] = ENCODER_WEIGHTS,
                in_channels: int = 1, classes: int = N_CLASSES) -> nn.Module:
    """U-Net, single-channel in, `classes` logit maps out (no activation)."""
    import segmentation_models_pytorch as smp
    # in_channels=1 with pretrained weights: SMP sums the RGB stem kernels, which is the
    # right thing for grayscale input and keeps the pretraining useful.
    return smp.Unet(encoder_name=encoder, encoder_weights=weights,
                    in_channels=in_channels, classes=classes, activation=None)


def pick_device(prefer: str = "auto") -> torch.device:
    """cuda > mps > cpu, unless overridden. Jetson takes cuda; this laptop takes mps."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- loss
class DiceBCELoss(nn.Module):
    """BCE + soft Dice, per class.

    BCE alone is the wrong loss here and it fails quietly: 47% of images are defect-free
    and the median defect covers ~3% of its image, so predicting "no defect everywhere"
    scores well on pixel-averaged BCE. Class 2 (175 training images, ~24px wide) would be
    the first casualty. Dice is computed per class so a rare class contributes to the
    gradient in proportion to its own area, not the batch's.
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight, self.smooth = bce_weight, smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)                                    # keep the class axis
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = 1.0 - ((2 * inter + self.smooth) / (denom + self.smooth))
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


# --------------------------------------------------------------------------- metrics
@torch.no_grad()
def dice_per_class(probs: torch.Tensor, target: torch.Tensor, thresh: float = 0.5,
                   min_px: int = 0) -> torch.Tensor:
    """Kaggle-style Dice per (image, class) -> [B, 4].

    Follows the competition convention: **an empty prediction on an empty target scores
    1.0**, not 0. With 47% clean images that convention dominates the mean, so getting it
    wrong makes every number in the project incomparable to any published Severstal
    result. `min_px` zeroes predictions smaller than a floor — the standard trick for
    suppressing speckle, tuned on val only.
    """
    pred = (probs > thresh).float()
    if min_px > 0:
        keep = (pred.sum(dim=(2, 3), keepdim=True) >= min_px).float()
        pred = pred * keep
    dims = (2, 3)
    inter = (pred * target).sum(dims)
    psum, tsum = pred.sum(dims), target.sum(dims)
    dice = (2 * inter) / (psum + tsum).clamp(min=1e-6)
    both_empty = (psum == 0) & (tsum == 0)
    return torch.where(both_empty, torch.ones_like(dice), dice)


@torch.no_grad()
def image_level_stats(probs: torch.Tensor, target: torch.Tensor, thresh: float = 0.5,
                      min_px: int = 0) -> dict[str, torch.Tensor]:
    """Per (image, class) defect / no-defect confusion counts.

    With 47% of images clean, "is there anything here at all" is half the problem — and it
    is the decision XP04's calibration analysis attaches to.
    """
    pred = (probs > thresh)
    if min_px > 0:
        pred = pred & (pred.sum(dim=(2, 3), keepdim=True) >= min_px)
    p = pred.any(dim=(2, 3))
    t = target.bool().any(dim=(2, 3))
    return {"tp": (p & t).sum(0), "fp": (p & ~t).sum(0),
            "fn": (~p & t).sum(0), "tn": (~p & ~t).sum(0)}
