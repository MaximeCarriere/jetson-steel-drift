# XP01 — Baseline model

**Goal:** train a *competent* steel-defect model — good enough to measure drift against
later, not a leaderboard winner. Everything from XP06 on measures change against this
baseline, so it must be stable and honestly scored.

**Status:** trained and evaluated ✅ — but it misses two of the four classes and needs a
rebalanced retrain (see the end).

## The data

Severstal training set (`data/severstal/`, see [download.md](../../data/download.md)):

| | |
|---|---|
| Images | **12,568** — all **1600 × 256** (wide strips) |
| Clean (no defect) | **5,902 (47%)** |
| With ≥1 defect | 6,666 |
| Defect classes | 4, with pixel masks |

The real Severstal test set has **private labels**, so we can't score on it. We carve our
own frozen test set out of the training data instead (see [Method](#method)).

### The four defect classes

Severstal never published what the classes *are* — no "scratch" / "pitting" names, they're
just 1–4. Measured from the masks:

| Class | Share of defects | Typical size | Shape |
|---|---:|---|---|
| **1** | 13% | small (~0.8% of image) | scattered spots |
| **2** | 3.5% | small & thin | vertical streak |
| **3** | **73%** | tiny → huge | scratches / lines |
| **4** | 11% | large (~6% of image) | solid patches |

- **Class 1** — scattered small spots.
- **Class 2** — the hard one: rarest (21× fewer than c3) and thinnest. Rare + thin = easy
  to miss.
- **Class 3** — the common one, 73% of all defects.
- **Class 4** — big solid patches, easiest to find.

![training examples](../../results/figures/xp01_examples_train.png)
*Training data: clean strips + the four defect classes, masks overlaid. The size gap
between the classes is the whole story.*

Almost every defective image has just **one** class; the only common pairing is 3+4, and
classes 1 and 4 never appear together. That's why the model outputs **4 independent masks**
rather than forcing one label per pixel — an image can have more than one defect.

## The model

- **Architecture:** U-Net with a **ResNet-34 encoder**.
- **Fine-tuned:** the encoder starts from ImageNet weights; we only train on Severstal (a
  few GPU-hours, not from scratch).
- **Output:** 4 independent masks, one per defect — not a single "pick one" softmax,
  because an image can have several defects at once.

**Loss = Dice + BCE.** Two loss terms, added together:

- **BCE** (binary cross-entropy) — grades each pixel: "defect or not?" Simple, but fooled
  here: 47% of images are clean, so a model that predicts *nothing* already scores well.
- **Dice** — grades the **overlap** between the predicted mask and the true mask (1 =
  perfect overlap, 0 = none). It cares about shape, so small rare defects still count.

BCE keeps pixel accuracy honest; Dice stops the model ignoring the small classes. We use
both.

## Result: strong on classes 3 & 4, blind to 1 & 2

Trained on the Jetson Orin Nano itself (20 epochs, ~2 h), scored on the frozen test set:

| Class | Detection recall (did it find the defect?) | Mask overlap (Dice) |
|---|---:|---:|
| **1** | **0.00** | **0.00** |
| **2** | **0.00** | **0.00** |
| 3 | 0.90 | 0.64 |
| 4 | 0.90 | 0.63 |

The model catches classes 3 and 4 well — it finds ~90% of their defect images with good
mask overlap — but **never detects classes 1 or 2**: zero hits out of their 134 and 36 test
images. This is run 1, not the final baseline.

![qualitative predictions](../../results/figures/xp01_predictions.png)
*One test defect per class: input · ground truth · model. For classes 1 and 2 the model
draws nothing. Class 3 is a clean match. Class 4 is found, but the model also paints spurious
class-3 streaks (its c3 output is trigger-happy).*

### The metric trap — why this matters for the whole project

A single "average Dice" for this model is **0.92** — and it's misleading. Because 47% of
images are clean, and scoring an empty prediction on a clean image counts as a **perfect
1.0**, the average is propped up by images that have no defect at all. A model can score
0.92 while detecting only half the classes.

So we don't report the average. We report **per-class detection** (did it find the
defect?), which can't be faked by staying quiet. **"Don't trust the headline number" is the
entire point of this project** — and here it is, in our own baseline.

![holdout scorecard](../../results/figures/xp01_holdout_dice.png)
*Left: the flattering "average Dice" bars (grey) for classes 1 & 2 are pure clean-image
freebie; the real overlap is zero. Right: detection recall / F1 — classes 1 & 2 flatline.*

![training curves](../../results/figures/xp01_training_curves.png)
*Training loss falls cleanly (left). The per-class val "Dice" (right) looks high for c1/c2,
but that's the same freebie — the model scoring points for staying silent, not for skill.*

### Why classes 1 & 2 failed, and the fix

No mystery: class 3 is **73% of all defects**, and c1/c2 are the smallest, thinnest ones.
The training signal from c3 drowns them out, so the model took the cheap win — learn c3/c4,
ignore c1/c2.

**Run 2 (next):** weight the rare classes more (inverse-frequency class weights + balanced
crop sampling). Only the loss and the crop sampler change; the split, evaluation and figures
stay. A baseline blind to two classes can't anchor the drift experiments, so this is a gate
before XP02.

### Test data

The same five categories on the frozen test set (the split the model never trains on):

![test examples](../../results/figures/xp01_examples_test.png)
*Frozen test set: clean + the four defect classes. Severstal's real test labels are private,
so this held-out slice of the training data is our test.*

## Method

- **Split.** Partition the training images into **train / val / frozen test** (70/15/15),
  stratified so clean images and each class are evenly spread. The test set is measured
  **once**, at the end, and never used for tuning.
- **Post-processing.** A probability threshold + a minimum-blob-size floor, tuned **per
  class on val** (not on the test set) to maximise detection F1 — a metric that can't be
  gamed by predicting nothing.
- **Feeds:** XP02 (deploy), XP03 (certification), XP04 (calibration), and the drift baseline
  from XP06 on.

## Reproduce

```bash
python experiments/xp01_baseline/profile_data.py    # data profile
python experiments/xp01_baseline/make_split.py      # frozen split
python experiments/xp01_baseline/train.py --epochs 20 --batch-size 12   # ~2 h on the Orin Nano
python experiments/xp01_baseline/evaluate.py        # scores on the frozen test set
python experiments/xp01_baseline/make_figures.py    # figures -> results/figures/
```
