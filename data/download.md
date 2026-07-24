# Data

`data/` is gitignored — **do not commit images, CSVs, or weights.** This file records what
should be on disk and how to get it back.

## Severstal Steel Defect Detection (present on disk)

Supplied by the repo owner from Kaggle on 2026-07-23, extracted to `data/severstal/`.

> Alexey Grishin, BorisV, iBardintsev, inversion, and Oleg. *Severstal: Steel Defect
> Detection.* <https://kaggle.com/competitions/severstal-steel-defect-detection>, 2019.
> Kaggle.
>
> Competition data is governed by the rules accepted at download time — worth re-reading
> the "DATA ACCESS AND USE" clause before publishing images or derived weights.

```
data/severstal/
├── train.csv               17.8 MB  ImageId, ClassId, EncodedPixels (RLE)
├── sample_submission.csv    138 KB
├── train_images/          12,568 jpg
└── test_images/            5,506 jpg     (labels are private — unusable for measurement)
```

Restore with:

```bash
mkdir -p data/severstal
unzip -q data_steel.zip -d data/severstal
# or, with the Kaggle CLI and competition rules accepted:
kaggle competitions download -c severstal-steel-defect-detection -p data/
```

### Verified structure — checked on disk 2026-07-23

| Property | Value |
|---|---|
| Image dimensions | **1600 × 256** (W × H) — the wide strips PLAN §XP01 warns about |
| Mode | RGB (visually grayscale; 3 identical channels) |
| Train images | 12,568 |
| — with ≥1 defect | 6,666 |
| — with **no** defect | 5,902 (47%) |
| — with multiple defects | 427 |
| Annotation rows | 7,095 |
| Test images | 5,506 (**labels private — not usable for accuracy**) |

Per-class image counts — note the imbalance, which XP04 expects to punish calibration on
the rare classes:

| ClassId | Images | Share of defective |
|---|---:|---:|
| 1 | 897 | 13.5% |
| 2 | **247** | 3.7% |
| 3 | **5,150** | 77.3% |
| 4 | 801 | 12.0% |

**Class 2 has 247 examples against class 3's 5,150 — a 21× imbalance.** Any per-class
metric on class 2 will be noisy, and this must be carried into XP01's split design and
stated with the results.

### Two things to get right in XP01

1. **Test labels are private.** `test_images/` cannot measure anything. Split
   `train_images/` into train / val / **frozen holdout**. The holdout is the only honest
   accuracy number in the project and is never used for tuning (PLAN §8).
2. **Masks are run-length encoded** in `train.csv`, one row per (image, class) pair.
   Images with no defect are absent from the CSV entirely — they must be recovered by
   diffing against the directory listing, not by assuming the CSV is complete.

## DAGM 2007 (fallback — not downloaded)

Held in reserve should Severstal ever become unusable: *Weakly Supervised Learning for
Industrial Optical Inspection*, CC BY 4.0, DOI
[10.5281/zenodo.12750201](https://doi.org/10.5281/zenodo.12750201). Synthetic textures with
weak ellipse labels rather than real steel with RLE masks, so it would rescope XP01 from
segmentation to classification with coarse localisation.
