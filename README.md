# Drift Monitoring for Industrial Vision on the Edge

**A model deployed on a factory line has no labels.** It answers every image confidently,
forever, whether or not it is still right. Nobody finds out until a bad batch ships.

This repo builds and validates a **label-free drift monitor** that runs on a single NVIDIA
Jetson Orin Nano Super (8 GB, 25 W, $249): it watches the incoming images and says *"the
conditions have changed — this model is now outside the envelope it was validated in."*

**The claim to be earned:** a signal computed *without labels* tracks accuracy degradation
we *cannot* measure in production. We prove it by deliberately hiding labels we actually
have.

> Systems / methodology study. Not a safety claim, and not an attempt at a leaderboard
> score. The question is not "is the model good?" but **"how do I know it is still
> working?"**

## Status

Building in experiment order (XP01 → XP13). See [PLAN.md](PLAN.md) for the full spine.

| # | Experiment | Status | Result |
|---|---|---|---|
| [XP01](experiments/xp01_baseline/) | Baseline model | ✅ done | U-Net (ResNet-34) fine-tuned on the Jetson; all 4 classes detected (recall 0.70–0.97), 88% defect/clean accuracy |
| XP02 | Edge deployment | ⬜ | |
| XP03 | Day-0 certification | ⬜ | |
| XP04 | Calibration | ⬜ | |
| — | **GATE 1** | ⬜ | quantised model with known accuracy *and* known confidence quality |
| XP05 | Drift simulation harness | ⬜ | |
| XP06 | Ground-truth degradation | ⬜ | |
| XP07 | Calibration under drift | ⬜ | |
| XP08 | Label-free signals | ⬜ | |
| XP09 | **Correlation — the result** | ⬜ | |
| — | **GATE 2** | ⬜ | if signals do not track accuracy, the thesis is wrong — stop and report it |
| XP10 | Detector ROC | ⬜ | |
| XP11 | Three-tier envelope | ⬜ | |
| XP12 | On-device cost | ⬜ | |
| XP13 | Live demo loop | ⬜ | |

Each `experiments/xpNN_*/` folder carries its own README: the question, the method, the
result, and what it feeds into.

## Data

**Severstal: Steel Defect Detection** — 12,568 labelled training images of cold-rolled
steel strip at 1600×256, four defect classes with run-length-encoded masks. Structure and
restore instructions: [`data/download.md`](data/download.md); measured class profile:
[XP01](experiments/xp01_baseline/).

> Alexey Grishin, BorisV, iBardintsev, inversion, and Oleg. *Severstal: Steel Defect
> Detection.* <https://kaggle.com/competitions/severstal-steel-defect-detection>, 2019.
> Kaggle.

Competition data is governed by the rules accepted at download time — check them before
publishing images or derived weights.

`data/` and `results/raw/` are gitignored — do not commit images or weights.

## Honest limitations — read these first

1. **Simulated drift is not real drift.** Synthetic lens contamination is not a lens
   getting dusty over six months. This is the standard limitation of the field, and the
   natural ask to a design partner: *give us six months of real line data and we will
   validate on that.*
2. **No public leaderboard comparison.** Severstal's test labels are private, so we split
   the training set. We are not claiming SOTA and do not need to.
3. **The defect classes are anonymous.** Severstal never published what ClassId 1–4 mean.
   We characterise them by measured geometry, not by a defect taxonomy — and one class
   (2, n=247) is rare enough that its per-class numbers will be noisy throughout.
4. **Drift detection is not novel research.** There is a substantial literature. What is
   under-served is doing it *on-device, label-free, with validated thresholds, on
   industrial vision* — that is the framing here.
5. **One dataset, one domain.** Whether the signals transfer is unproven until tested.

## Measurement hygiene

Non-negotiable, carried over from `jetson-xray-panel` because it is what separates a demo
from evidence:

- Fixed seeds everywhere; drift transforms exactly reproducible.
- Warm-up discarded; steady-state timing only. Every configuration run **3×**, mean ± std.
- Power logged continuously via `tegrastats`, aligned to the run window.
- Power mode locked (`nvpmodel`, `jetson_clocks`) and recorded in every result file.
- **The frozen holdout is never used for tuning. Not once.**
- Every experiment writes machine-readable JSON to `results/`.

## Layout

```
jetson-steel-drift/
├── PLAN.md                    # the master plan
├── data/download.md           # dataset provenance + verified structure
├── lib/                       # models · drift · calibration · signals · bench · power
├── experiments/xpNN_*/        # one folder per experiment, each with its own README
├── results/                   # one JSON per run, machine-readable
└── figures/                   # committed
```

---

*Owner: Maxime Carriere · Hardware: NVIDIA Jetson Orin Nano Super (8 GB, 25 W)*
