"""Drift simulation harness — named, physically-motivated corruptions with a severity dial.

Each drift is a function of (image, severity in [0,1]) and is exactly reproducible (seeded).
severity = 0 is the identity, so the baseline is always recoverable. numpy only, no torch:
drift is applied to the input image before it reaches the model.

Three drifts, chosen because they fail the model in three different ways:

  light_corner  a bright glare in one corner (aged/replaced lamp, a reflection). It washes
                out real defects -> the model MISSES them (false negatives).
  marks         random soft dark/bright blobs (dirt, oil, dust on the lens). They do NOT
                resemble this model's line/patch defects, so it shrugs them off -> a
                near-harmless drift (a useful negative control for the monitor).
  streaks       thin dark vertical scratches — shaped like real class-3 defects. The model
                mistakes them for defects -> it HALLUCINATES them (false positives).
"""
from __future__ import annotations

import numpy as np

KINDS = ("light_corner", "marks", "streaks")
SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _as_float(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return (float image in [0,1], was_uint8)."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, True
    return img.astype(np.float32), False


def _restore(img: np.ndarray, was_uint8: bool) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8) if was_uint8 else img


def light_corner(image: np.ndarray, severity: float, corner: str = "top_right",
                 max_boost: float = 1.6) -> np.ndarray:
    """Smooth bright glare from a corner; strength AND reach scale with severity.

    Additive glare that saturates the region to pure white (over-exposure), hiding any
    defect under it. At high severity it spreads well across the strip, not just the
    corner, so it wipes out a real fraction of the image.
    """
    img, u8 = _as_float(image)
    if severity <= 0:
        return _restore(img, u8)
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = 0.0 if "top" in corner else h - 1.0
    cx = w - 1.0 if "right" in corner else 0.0
    # reach grows with severity: a corner spot at low severity, a broad wash at high.
    sig_y = h * (0.6 + 0.8 * severity)
    sig_x = w * (0.4 + 0.6 * severity)
    mask = np.exp(-(((ys - cy) / sig_y) ** 2 + ((xs - cx) / sig_x) ** 2))
    out = img + severity * max_boost * mask
    return _restore(out, u8)


def marks(image: np.ndarray, severity: float, seed: int = 0,
          max_blobs: int = 20) -> np.ndarray:
    """Random soft dark/bright blobs (lens contamination); count/size/opacity ~ severity.

    Stronger than a light dusting: at high severity, many large near-opaque blobs that
    cover a real fraction of the strip.
    """
    img, u8 = _as_float(image)
    if severity <= 0:
        return _restore(img, u8)
    h, w = img.shape
    rng = np.random.default_rng(seed)
    n = int(round(severity * max_blobs))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    out = img.copy()
    for _ in range(n):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        radius = rng.uniform(10, 18 + severity * 70)
        opacity = 0.45 + severity * 0.5
        colour = 0.0 if rng.random() < 0.5 else 1.0     # oil (dark) or dust glare (bright)
        blob = np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * radius ** 2)))
        a = opacity * blob
        out = out * (1 - a) + colour * a
    return _restore(out, u8)


def streaks(image: np.ndarray, severity: float, seed: int = 0,
            max_streaks: int = 6) -> np.ndarray:
    """Thin dark vertical scratches, shaped like real class-3 defects; count/opacity ~ sev.

    Unlike `marks`, these look like the defects the model was trained on, so it is expected
    to fire on them -> false positives on clean steel.
    """
    img, u8 = _as_float(image)
    if severity <= 0:
        return _restore(img, u8)
    h, w = img.shape
    rng = np.random.default_rng(seed + 1000)              # distinct stream from `marks`
    n = int(round(severity * max_streaks))
    ys = np.arange(h)[:, None].astype(np.float32)
    xs = np.arange(w)[None, :].astype(np.float32)
    out = img.copy()
    for _ in range(n):
        cx = rng.uniform(0, w)
        width = rng.uniform(1.3, 3.0)                     # thin, like a scratch
        y0, y1 = rng.uniform(0, h * 0.4), rng.uniform(h * 0.6, h)
        lean = rng.uniform(-15, 15)                       # slight tilt down the strip
        opacity = 0.5 + severity * 0.4
        t = np.clip((ys - y0) / max(1.0, y1 - y0), 0, 1)
        center = cx + lean * t                            # per-row x centre
        line = np.exp(-((xs - center) ** 2) / (2 * width ** 2))
        line *= ((ys >= y0) & (ys <= y1))
        a = opacity * line
        out = out * (1 - a)                               # dark scratch -> multiply toward 0
    return _restore(out, u8)


_FUNCS = {"light_corner": light_corner, "marks": marks, "streaks": streaks}
_SEEDED = {"marks", "streaks"}


def apply(image: np.ndarray, kind: str, severity: float, seed: int = 0) -> np.ndarray:
    """apply(image, kind, severity) -> drifted image. severity 0 = identity."""
    if kind not in _FUNCS:
        raise ValueError(f"unknown drift {kind!r}; choose from {KINDS}")
    fn = _FUNCS[kind]
    return fn(image, severity, seed=seed) if kind in _SEEDED else fn(image, severity)
