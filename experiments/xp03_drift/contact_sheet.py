"""XP03 — drift contact sheet: each drift at severity 0 / 0.25 / 0.5 / 0.75 / 1.0.

    python experiments/xp03_drift/contact_sheet.py

A human sanity check that the drifts look like real degradation, not Instagram filters —
before any accuracy is measured. Uses one real holdout strip.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from lib import drift                                    # noqa: E402
from lib.severstal import ROOT, TRAIN_IMG_DIR, load_index, load_split  # noqa: E402

FIG_DIR = os.path.join(ROOT, "results", "figures")
# Column order matches the degradation curves below: light glare, blobs, streaks.
LABELS = {"light_corner": "light glare", "marks": "blob contamination",
          "streaks": "defect-like streaks"}


PREFERRED = "0d78ac743.jpg"       # bright strip, visible class-4 defect, no dark border


def main() -> int:
    # a bright defective strip, so the drifts are clearly visible against a real defect
    index = load_index()
    hold = load_split()["holdout"]
    iid = PREFERRED if PREFERRED in hold else next(i for i in sorted(hold) if index[i])
    img = np.array(Image.open(os.path.join(TRAIN_IMG_DIR, iid)).convert("L"))

    # Columns = the three drifts (same left-to-right order as the degradation curves);
    # rows = increasing severity, top to bottom.
    kinds, sev = drift.KINDS, drift.SEVERITIES
    fig, axes = plt.subplots(len(sev), len(kinds), figsize=(5 * len(kinds), 1.5 * len(sev)))
    for r, s in enumerate(sev):
        for c, kind in enumerate(kinds):
            ax = axes[r, c]
            ax.imshow(drift.apply(img, kind, s, seed=1), cmap="gray", aspect="auto",
                      vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(LABELS[kind], fontsize=12, fontweight="bold")
        axes[r, 0].set_ylabel(f"severity {s:g}", fontsize=12, fontweight="bold")
    fig.suptitle(f"XP03 — drift contact sheet: three drifts (columns), rising severity "
                 f"(rows) · strip {iid}", fontsize=13, y=1.0)
    fig.text(0.5, -0.01, "severity 0 = original, increasing downward. Same drift order "
             "(left to right) as the degradation curves.", ha="center", fontsize=9,
             color="#666")
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "xp03_contact_sheet.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
