# XP03 — Drift: how much does a corrupted image break the model?

**Question.** A factory camera doesn't stay perfect — a lamp ages, oil films the lens, the
mount vibrates. As the *image* drifts (the steel underneath is unchanged), how much does the
model's **defect-vs-clean** decision degrade? This is the ground truth a label-free drift
monitor (later experiments) will have to predict without ever seeing the answer.

We simulate three named drifts, each with a **severity dial 0 → 1** (0 = untouched), and
measure accuracy on the frozen holdout as severity rises. The labels never change, so any
drop is honest degradation.

## The three drifts

The same strip at severity 0 / 0.25 / 0.5 / 0.75 / 1.0:

![contact sheet](../../results/figures/xp03_contact_sheet.png)

- **Light glare (top-right)** — an aged/replaced lamp or a reflection over-exposes a corner.
- **Blob contamination** — random soft dark/bright spots: dirt, oil, dust on the lens.
- **Defect-like streaks** — thin dark scratches, *shaped like real class-3 defects*.

They're built to fail the model in three different ways, and they do.

## Results — three different failure modes

![degradation curves](../../results/figures/xp03_degradation.png)

| Drift | Accuracy | What breaks | Failure mode |
|---|---|---|---|
| **Light glare** | 0.88 → 0.83 | **recall** 0.94 → 0.85 | **misses** defects (washed out) |
| **Blob contamination** | 0.88 → 0.87 | nothing | **harmless** — model shrugs it off |
| **Defect-like streaks** | 0.88 → **0.63** | **specificity** 0.80 → **0.23** | **false alarms** (cries wolf) |

Reading each curve (recall = defects caught, specificity = clean strips left alone):

- **Glare washes defects out**, so the model *misses* them — recall falls. A moderate,
  steady decline.
- **Blob contamination barely matters.** The blobs are round; the model's defects are lines
  and patches, so it doesn't mistake them for defects. A drift that changes the picture but
  not the accuracy — a useful *negative control*.
- **Defect-like streaks are catastrophic.** They look like real scratches, so the model
  fires on them — specificity collapses from 80% to 23% (four clean strips in five now get a
  false "defect"). Shape is what matters: the same amount of contamination is harmless as
  blobs and ruinous as streaks.

## Why this matters for the drift monitor

These three give the monitor its test cases — and they're not all "alarm":

- glare → image changed **and** accuracy dropped → the monitor **should** alarm.
- streaks → image changed **and** accuracy collapsed → the monitor **should** alarm loudly.
- blobs → image changed but accuracy held → the monitor should **stay quiet**.

A monitor that fires on all three is as useless as one that fires on none. XP03 is the
ground truth; the label-free signals (XP08) will be judged on whether they track *these*
curves — up for glare and streaks, flat for blobs.

## Reproduce

```bash
python experiments/xp03_drift/contact_sheet.py   # the severity contact sheet
python experiments/xp03_drift/degradation.py     # accuracy vs severity -> results/xp03_degradation.json
python experiments/xp03_drift/make_figures.py    # the degradation curves
```
