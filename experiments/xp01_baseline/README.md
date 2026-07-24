# XP01 — Baseline model

**Question.** Can we train a *competent* steel-defect model?

Not a winning one. XP01 exists to produce a model good enough that its **degradation under
drift is meaningful** — everything from XP06 onward measures changes against this baseline,
so the baseline has to be stable and honestly measured, not maximal. Tuning for the
leaderboard here would buy nothing the rest of the project can use.

**Status:** data profiled ✅ · model not yet trained ⬜

## The data

Severstal train split, `data/severstal/` (see [`data/download.md`](../../data/download.md)).
Profiled by [`profile_data.py`](profile_data.py) → [`results/xp01_data_profile.json`](../../results/xp01_data_profile.json).

| | |
|---|---|
| Images | **12,568**, all **1600 × 256** (W × H) — wide strips, ~6.25:1 |
| With ≥1 defect | 6,666 |
| **Clean (no defect)** | **5,902 — 47%** |
| Annotation rows | 7,095 |
| Multi-class images | 427 |
| Test set | 5,506 images, **labels private → unusable for measurement** |

### The defect classes

**Severstal never published what ClassId 1–4 mean.** The classes are anonymous by design —
there is no official mapping to "scratch", "inclusion", "pitting" or anything else, and any
source that gives you one is guessing. What we *can* state is measured geometry, from the
masks themselves:

| Class | Instances | Images | Share | Median area | Median % of image | Median bbox (w×h) | Fill ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | 897 | 897 | 12.6% | 3,326 px | 0.81% | 58 × 165 | 0.39 |
| **2** | **247** | 247 | **3.5%** | 2,944 px | 0.72% | **24 × 208** | **0.66** |
| **3** | **5,150** | 5,150 | **72.6%** | 11,954 px | 2.92% | 306 × 255 | 0.30 |
| **4** | 801 | 801 | 11.3% | **25,357 px** | **6.19%** | 363 × 210 | 0.44 |

*Fill ratio = defect pixels ÷ bounding-box area. Near 1.0 is a solid blob; low is a thin or
wispy shape.*

Read across the rows and the four classes are genuinely different problems:

- **Class 2 is the hard one.** 247 examples — a **21× imbalance** against class 3 — and the
  narrowest shape in the set (median 24 px wide on a 1600 px strip, but 208 px tall, i.e. a
  near-vertical streak spanning most of the strip height). Rare *and* thin is the worst
  combination for a segmentation model: it contributes almost nothing to a pixel-averaged
  loss, and a few pixels of boundary error wreck its Dice. **Expect class 2 to be the first
  thing that breaks, and expect XP04 to find its calibration is the worst of the four** —
  exactly the case the plan predicts an unchanged mean will hide.
- **Class 3 dominates.** 72.6% of all annotations. A model that only ever predicts class 3
  will look respectable on a mean-Dice metric. This is the failure mode to guard the
  baseline against.
- **Class 4 is large and solid** — median 6.2% of the image, up to 47%. Easiest to
  segment; least informative about drift sensitivity.
- **Classes 1 and 3 are wispy** (fill 0.39 / 0.30) — they scatter across their bounding box
  rather than filling it, which is why coarse or downsampled predictions smear them.
- Area spans four orders of magnitude: class 3 runs from 0.03% to **89.9%** of the image.
  Any single resize or crop strategy trades one end of that range against the other.

![training examples](../../results/figures/xp01_examples_train.png)
*Training data — clean strips and the four defect classes, masks overlaid. Clean = no
defect (47% of images). c1 = scattered spots, c2 = thin vertical streak, c3 = vertical
scratches, c4 = large patches. The size gap between the classes is the whole story.*

### Class co-occurrence

Almost all defective images carry exactly one class:

| Combination | Images |
|---|---:|
| 3 only | 4,759 |
| 1 only | 769 |
| 4 only | 516 |
| **3 + 4** | **284** |
| 2 only | 195 |
| 1 + 3 | 91 |
| 1 + 2 | 35 |
| 2 + 3 | 14 |
| 1+2+3 | 2 |
| 2 + 4 | 1 |

**Classes 1 and 4 never co-occur** (0 images). 3+4 is the only common pairing. So the task
is close to — but not exactly — single-label, which means **four independent binary masks,
not a 5-way softmax**. A softmax would assert mutual exclusivity that 427 images violate.

## The model

**U-Net with a ResNet-34 encoder, fine-tuned** (via
[segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch),
MIT). The encoder starts from ImageNet weights, so we only fine-tune on Severstal — a few
GPU-hours, not a from-scratch train. (No Kaggle solution publishes usable weights, so
fine-tuning our own is the shortest path anyway.) The head is **four independent sigmoid
masks**, one per class — not a 5-way softmax — because 427 images carry more than one
defect at once.

**Loss: Dice + BCE.** BCE alone fails silently here: 47% of images are clean and the median
defect covers ~3% of its image, so "predict nothing" already scores well on pixel-averaged
BCE. Dice measures overlap per class, so a rare, small defect still contributes to the
gradient in proportion to its own area — which is what keeps the small classes from being
ignored.

## Result — run 1: a competent c3/c4 detector, blind to c1/c2

**Trained on the Jetson Orin Nano itself** (CUDA, AMP, 20 epochs × ~6 min, ~2 h total).
U-Net / ResNet-34, 24.4 M params, Dice+BCE loss, defect-biased 256×256 crops + flips.

The honest frozen-holdout scorecard (`results/xp01_baseline.json`):

| Class | detection recall | detection F1 | defect-only Dice | Kaggle Dice (freebie) |
|---|---:|---:|---:|---:|
| **1** | **0.00** | **0.00** | **0.00** | 0.93 |
| **2** | **0.00** | **0.00** | **0.00** | 0.98 |
| 3 | 0.90 | 0.88 | 0.64 | 0.80 |
| 4 | 0.90 | 0.85 | 0.63 | 0.96 |

**The model never detects classes 1 or 2** — 0 true positives out of 134 and 36 holdout
defect images, even at the most permissive operating point (thr 0.3, no min-px floor). It
is a strong detector of classes 3 and 4 (recall 0.90, defect-Dice ~0.63) and blind to the
other two. This is run 1, not the final baseline — the fix is below.

### The metric trap this exposes (and why it matters for the whole project)

A single global threshold tuned on **mean Dice** gives a headline **0.918** on the holdout —
and it is a lie. Because 47% of images are clean and the competition scores an
empty-prediction-on-empty-target as **1.0**, mean Dice is *maximised* by suppressing
predictions on the rare classes and banking the freebie. The naive tuner drives `min_px`
all the way to 2000, which zeroes every c1/c2 prediction, and reports 0.92 for a model
that detects half the classes. This is the project's central thesis — *don't trust the
headline number* — appearing in its own baseline. We tune per-class operating points on
**detection F1** instead (F1 = 0 when recall = 0, so it cannot be gamed by abstention) and
report defect-only Dice as the real quality number.

![training curves](../../results/figures/xp01_training_curves.png)
*Left: training loss falls cleanly, 0.92 → 0.19. Right: the training-log Dice — but c1/c2's
flat high lines (0.93 / 0.98) are **abstention**, not skill: on a mostly-clean val set a
model that predicts nothing on a class scores ~1.0 there. That illusion is exactly what the
holdout scorecard below strips away.*

![holdout scorecard](../../results/figures/xp01_holdout_dice.png)
*The grey Kaggle-Dice bars (0.93, 0.98) for c1/c2 are pure freebie; defect-only Dice and
detection recall are flat at zero. c3/c4 are genuine.*

![confusion matrices](../../results/figures/xp01_confusion.png)
*Image-level detection on the 1,884-image holdout. c1/c2 have an empty predicted-defect
column: every one of their 134 / 36 defect images is called clean. c3/c4 catch ~90%.*

![qualitative predictions](../../results/figures/xp01_predictions.png)
*One holdout defect per class: input · ground truth · model. c1/c2 — the model draws
nothing. c3 — clean match. c4 — catches the blob but also hallucinates c3 streaks (the c3
head is trigger-happy: 109 false positives).*

### Why c1/c2 collapsed — and the fix for run 2

It is not mysterious once you look at the data figure and the examples: class 3 is **72.6%
of all annotations**, and c1/c2 are the smallest, thinnest defects (c1 median 0.81% of the
image in scattered spots, c2 a ~24 px-wide streak). Per-class Dice in the loss still lets
c3's gradient dominate, and defect-biased cropping mostly surfaces c3. The model took the
cheap win: learn c3/c4, ignore c1/c2.

**Run 2 (next):** rebalance so the rare classes carry weight — inverse-frequency class
weights in the loss and/or class-balanced crop sampling (sample the defect *class* to crop
around uniformly, not whatever defect is present). Everything else — split, eval,
per-class F1 tuning, figures — stays; only `lib/models.DiceBCELoss` and the crop sampler in
`lib/data.py` change. A baseline that can't see two of four classes cannot anchor the drift
experiments (you can't measure drift-degradation on a class the model never detects), so
this is a gate before XP02.

### Test data

The same five categories on the **frozen holdout** — the split the model never trains on,
used only for the measurement above:

![test examples](../../results/figures/xp01_examples_test.png)
*Test data (frozen holdout): clean + the four defect classes. Same distribution as the
training gallery; Severstal's real test labels are private, so this holdout is our test.*

![data distribution](../../results/figures/xp01_data_distribution.png)
*Why c1/c2 are hard: 21× rarer than c3, and the smallest/thinnest shapes.*

## Method

1. **Split.** Test labels are private, so partition `train_images/` into
   train / val / **frozen holdout**. Stratify on the class combination above, not on class
   id, so the 427 multi-class images and the 5,902 clean images distribute properly. The
   holdout is sacred: touched once, at the end, and never for tuning (PLAN §8).
2. **Model.** SMP U-Net, ImageNet-pretrained `resnet34`/`efficientnet-b0` encoder — see the
   survey above. Boring on purpose.
3. **Input geometry.** 1600×256 is the plan's flagged trap. Options are full-strip, or
   tiling into ~256×256 crops — the choice interacts with the tiny-class-2 problem and with
   XP02's latency budget, so decide it with a measurement, not a preference.
4. **Loss.** Pixel-averaged BCE alone will ignore class 2. Dice/Tversky or class weighting
   is not optional here.
5. **Metrics on the frozen holdout.** Per-class Dice/IoU **and** an image-level
   defect/no-defect metric — with 47% clean images, the "is there anything here at all"
   decision is half the problem and is what XP04's calibration analysis will attach to.

## Deliverable

Trained model + baseline metrics on the frozen holdout, per class, with the clean-image
false-positive rate stated separately.

**Feeds:** XP02 (export/deploy), XP03 (certification reference), XP04 (calibration), and
the entire drift-degradation baseline from XP06 on.

## Reproduce

```bash
python experiments/xp01_baseline/profile_data.py     # data profile  -> results/xp01_data_profile.json
python experiments/xp01_baseline/make_split.py       # frozen split  -> results/xp01_split.json
python experiments/xp01_baseline/train.py --epochs 20 --batch-size 12   # ~2 h on the Orin Nano
python experiments/xp01_baseline/evaluate.py         # holdout metrics -> results/xp01_baseline.json
python experiments/xp01_baseline/make_figures.py     # all 7 figures  -> results/figures/xp01_*.png
```
