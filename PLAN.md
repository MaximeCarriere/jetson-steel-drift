# Master Plan — Drift Monitoring for Industrial Vision on the Edge

**Project:** `jetson-steel-drift`
**Owner:** Maxime Carriere · **Hardware:** NVIDIA Jetson Orin Nano Super (8 GB, 25 W, $249)
**Structure:** incremental experiments (XP00 → XP12), same convention as `jetson-xray-panel`

---

## 0. TL;DR

A model deployed on a factory line has no labels. It answers every image confidently, forever,
whether or not it is still right. Nobody finds out until a bad batch ships.

**This project builds and validates a label-free drift monitor**: something that runs on-device,
watches the incoming steel images, and says *"the conditions have changed — this model is now
outside the envelope it was validated in."*

**The central claim we must earn:** a signal computed *without labels* tracks the accuracy
degradation we *cannot* measure in production. We prove it by deliberately hiding labels we
actually have.

Everything else in this plan is scaffolding for that one result.

---

## 1. Goal & Non-Goals

### Goal
Demonstrate an on-device drift monitor for an industrial defect-detection model, with measured
evidence that its label-free alerts correspond to real accuracy loss — and a tuned three-tier
alert whose false-alarm and missed-detection rates are known, not guessed.

### Why this and not "we ran a Kaggle winner on a Jetson"
Running a good model on a Jetson proves nothing anyone needs. The unanswered industrial question
is *"how do I know it is still working?"* — which is exactly what Andrew Hamilton named as the
adoption blocker (*"black box hard to be accepted by the companies"*). This project answers that
question with numbers.

### Non-Goals — say no to these explicitly
- **NOT** chasing the Severstal leaderboard. We need a *competent* model, not a winning one.
- **NOT** inventing a novel drift-detection algorithm. Drift detection has a large literature.
  We are productising and validating on edge hardware, not advancing the science.
- **NOT** a clinical/safety claim of any kind.
- **NOT** Pierre's sealing layer (that is XP12, stretch only).
- **NOT** real production drift — we simulate. Stating this honestly is mandatory (see §5).

---

## 2. The spine — experiments at a glance

| XP | Name | Question it answers | Effort |
|---|---|---|---|
| **00** | Licence & data gate | Are we legally allowed to build this demo on this data? | 0.5 d |
| **01** | Baseline model | Can we train a competent defect model? | 2 d |
| **02** | Edge deployment | Does it run fast enough on the Jetson? | 1 d |
| **03** | Day-0 certification | Did compression break it? | 1 d |
| **04** | **Calibration** | When the model says 80%, is it right 80% of the time? | 1.5 d |
| — | **GATE 1** | Working quantised model with known accuracy *and* known confidence quality? | — |
| **05** | Drift simulation harness | Can we generate realistic, controllable drift? | 1.5 d |
| **06** | Ground-truth degradation | How much does accuracy *actually* fall under drift? | 1 d |
| **07** | **Calibration under drift** | Does the model get *confidently* wrong? | 1 d |
| **08** | Label-free signals | What can we measure without labels? | 2 d |
| **09** | **Correlation — THE RESULT** | Do label-free signals predict accuracy loss? | 1.5 d |
| — | **GATE 2** | If signals do NOT track accuracy, the product thesis is wrong. Stop and rethink. | — |
| **10** | Detector ROC | What are the false-alarm / missed-detection rates? | 1.5 d |
| **11** | Three-tier envelope | Can we set green/amber/red thresholds defensibly? | 1 d |
| **12** | On-device cost | Can the monitor run continuously alongside inference? | 1 d |
| **13** | Live demo loop | Does it work end to end, watchable in two minutes? | 1.5 d |
| **14** | *(stretch)* Sealed monitor | Can the alert be made tamper-evident? | — |

**~19 working days.** Phases A–B (XP00–09) are the science. Phase C (XP10–13) is the product.

> **Why calibration sits on the critical path.** One of the label-free drift signals in XP08 is
> *confidence distribution shift*. That signal is only meaningful if the confidence numbers mean
> something in the first place. XP04 establishes whether they do; XP07 establishes whether they
> survive drift. If the answer to either is no, XP08 must drop that signal — better to find out
> before building on it.

---

## 3. PHASE A — Foundation

### XP00 · Licence & data gate
**Question:** are we allowed to use this data for a demonstration on a company website?

Do this **before writing any code**. The MVTec lesson applies here too.

- Read the Severstal competition rules on Kaggle. Competition data often carries
  "non-commercial research use" terms — verify what a public technical demo counts as.
- If restricted, evaluate alternatives and check *their* licences: **NEU-DET**, **GC10-DET**,
  **KolektorSDD2**, **DAGM 2007**.
- If nothing in the steel domain is clean, fall back to a permissively-licensed industrial set
  and keep the same method — the method is the point, not the material.

**Deliverable:** one paragraph in the README stating the dataset, its licence, and why the
intended use is permitted. **Gate: do not proceed without this.**

---

### XP01 · Baseline model
**Question:** can we train a competent steel-defect model?

- Severstal: ~12.5k training images, 4 defect classes, run-length-encoded masks.
  **Note the unusual aspect ratio (very wide strips) — verify dimensions before writing loaders.**
- **Test labels are not public.** Split the *training* set into train/val/**frozen holdout**.
  The frozen holdout is sacred: it is the only honest accuracy measurement in the whole project.
- Architecture: a standard U-Net or segmentation model with a pretrained backbone. Boring on
  purpose. Do not tune for the leaderboard.

**Deliverable:** a trained model + baseline metrics on the frozen holdout (Dice/IoU per class,
plus a defect/no-defect classification metric).
**Depends on:** XP00.

---

### XP02 · Edge deployment
**Question:** does it run on the Jetson, and how fast?

Reuse the pipeline from `jetson-xray-panel` directly — this is why that project was worth doing.

- Export → ONNX → TensorRT engine, FP16 and INT8.
- Measure throughput, latency (mean/p95), peak memory, power via `tegrastats`.
- Warm-up, steady-state only, 3 repeats.

**Deliverable:** the same benchmark table as the X-ray work, for steel.
**Depends on:** XP01.

---

### XP03 · Day-0 certification
**Question:** did compression break the model?

This is the certification product applied to an industrial model.

- FP32 reference vs FP16 vs INT8 on the frozen holdout.
- Report: metric delta, **per-class** degradation, **agreement rate** (do the two models flag
  the same pixels?), and a **visual diff** showing where they disagree.
- The visual diff is the money shot: pixel masks let you show *where* the compressed model
  started disagreeing, not just that a number moved.

**Deliverable:** a certification report (markdown + figures) for the deployed model.

---

### XP04 · Calibration — is the predicted probability a real probability?
**Question:** when the model outputs 0.8, is it right roughly 80% of the time?

Accuracy tells you how often the model is right. It says nothing about whether you can *trust the
number it reports*. Those are different properties, and the second one is what a factory needs if
it wants to route uncertain cases to a human.

**Method**
- **Reliability diagram** — bin predictions by confidence, plot predicted confidence against
  observed frequency. Perfect calibration is the diagonal; below it means over-confident.
- **ECE** (Expected Calibration Error) and **MCE** (Maximum), plus **Brier score**.
- **Per-class.** Severstal is heavily imbalanced — expect the rare defect classes to be badly
  calibrated even when the average looks acceptable. An unchanged mean hides this.
- **Segmentation-specific:** report *pixel-level* and *image-level* calibration separately.
  They can disagree, and the image-level one is what an operator actually sees.
- **FP32 vs FP16 vs INT8.** Does quantisation change calibration even where it leaves accuracy
  untouched? *(In the X-ray project, ranking metrics were bit-identical while the probability
  distribution was badly distorted — so check this explicitly rather than assuming.)*
- **Temperature scaling.** Fit a single scalar on validation, re-measure ECE. If it fixes the
  problem, miscalibration was a scale issue. **If it barely moves ECE, the miscalibration is
  structural** — a much more interesting and more reportable finding.

**Watch for the trap.** Ranking metrics (AUROC, Dice) are invariant to any monotonic transform of
the scores. A model can be badly miscalibrated with *identical* AUROC. If you only report ranking
metrics you will never see it. This is exactly the failure mode found in `xp14_calibration` of the
X-ray project, and it is the reason this experiment exists.

**Deliverable:** reliability diagrams (FP32/FP16/INT8), ECE/MCE/Brier table, per-class breakdown,
before/after temperature scaling — and an explicit verdict: **is this confidence usable as a
decision variable, yes or no?**
**Feeds:** XP07, and the viability of the confidence-based signal in XP08.
**Depends on:** XP03.

> ### GATE 1
> Do we have a quantised model on the Jetson with a known, documented accuracy **and** a known
> calibration quality? If not, fix it before touching drift. Everything downstream measures
> *changes* against this baseline — an unstable or unmeasured baseline makes the project
> meaningless.

---

## 4. PHASE B — The science

### XP05 · Drift simulation harness
**Question:** can we generate realistic, controllable, *industrially plausible* drift?

Not random noise — **named physical causes**, each with a severity parameter 0→1:

| Drift | Physical cause | Transform |
|---|---|---|
| Illumination shift | lamp aged / replaced / different shift | global brightness + gamma |
| Contrast loss | dust or oil film on the lens | contrast reduction + slight blur |
| Focus drift | vibration loosened the mount | progressive Gaussian blur |
| Sensor gain drift | camera electronics ageing | gain + additive sensor noise |
| Surface finish change | new steel grade / different mill | texture contrast + local histogram shift |
| Geometric drift | camera bumped | small rotation / translation / scale |

Each parameterised so you can sweep severity smoothly and reproducibly (fixed seeds).

**Deliverable:** `drift.py` — apply(image, kind, severity) → image, plus a contact sheet showing
each drift at severity 0.0 / 0.25 / 0.5 / 0.75 / 1.0 so a human can sanity-check that they look
like real degradation and not like Instagram filters.

---

### XP06 · Ground-truth degradation curves
**Question:** how much does accuracy *actually* fall, per drift type, per severity?

- For each drift kind × severity: run the frozen holdout **with labels**, measure true accuracy.
- Produce degradation curves: accuracy vs severity, one line per drift kind.

This is the ground truth the monitor will later have to predict blind. Expect the curves to differ
sharply by drift type — some corruptions barely matter, some are catastrophic. **That asymmetry
is itself a finding**, and it tells you which failure modes a plant should actually worry about.

**Deliverable:** degradation curves + a table of "severity at which accuracy drops 5% / 10% / 20%".

---

### XP07 · Calibration under drift — does the model become *confidently* wrong?
**Question:** as conditions drift, does the model lose accuracy without losing confidence?

**This is the silent-failure mechanism, made measurable.** A model that becomes visibly uncertain
when it starts failing is manageable — an operator sees hesitation and intervenes. A model that
keeps reporting 0.95 while its accuracy collapses is dangerous, because nothing in the output
signals that anything is wrong.

**Method**
- For each drift kind × severity, measure *both* accuracy (XP06) and calibration (XP04 metrics).
- Key plot: **mean confidence vs true accuracy, as severity increases.** If the two lines diverge,
  the model is becoming over-confident. The size of that gap is the story.
- Track ECE against severity: does calibration degrade faster, slower, or in step with accuracy?
- Check whether temperature scaling fitted on clean data still holds under drift. It almost
  certainly will not — which is a useful finding, because it means calibration is not a one-time
  fix you apply at deployment and forget.

**Two possible outcomes, both useful:**
- **Confidence *does* fall under drift** → you have a free, zero-cost drift signal. Excellent for
  XP08, and it means the model partially self-reports.
- **Confidence *stays flat* while accuracy falls** → confidence is a **bad** drift signal and must
  be dropped or down-weighted in XP08. Also the more likely outcome, and the more compelling
  narrative: *the model does not know it is failing, which is precisely why you need an external
  monitor.*

Either way this experiment directly determines what XP08 is allowed to use.

**Deliverable:** confidence-vs-accuracy divergence curves, ECE-vs-severity curves, and a verdict on
whether confidence is viable as a drift signal.
**Depends on:** XP04, XP05, XP06. **Feeds:** XP08, XP09.

---

### XP08 · Label-free signal candidates
**Question:** what can we compute in production, where there are no labels?

Implement at least five, cheapest first:

1. **Input statistics** — mean/std/histogram of pixel intensities; distance to the training
   distribution via Wasserstein or KL.
2. **Prediction-rate shift** — "defect rate moved from 2% to 15%". Free to compute, and the
   signal a plant manager understands instantly.
3. **Confidence / entropy distribution shift** — is the model becoming less certain without
   saying so? **Include only if XP07 showed confidence actually moves under drift.** If XP07
   found confidence stays flat while accuracy collapses, this signal carries no information and
   shipping it would be worse than useless — it would give false reassurance.
4. **Embedding drift** — Mahalanobis distance of intermediate-layer features against training-set
   statistics. Catches *semantic* change (new steel grade) that pixel statistics miss.
5. **Reconstruction / self-consistency** *(optional)* — e.g. prediction stability under small
   test-time perturbations.

Each must produce a single scalar per image (or per batch), and each must be computable on-device.

**Deliverable:** `signals.py` with a uniform interface, plus per-signal compute cost in ms and MB.

---

### XP09 · Correlation — **THE RESULT**
**Question:** do the label-free signals predict the accuracy loss we cannot see?

- For every (drift kind × severity) cell: plot label-free signal against true accuracy loss.
- Compute rank correlation (Spearman) per signal, per drift kind, and overall.
- Identify which signals are **general** (work across drift types) versus **specialist** (only
  catch one failure mode).
- Expect a mixed picture. Pixel statistics will likely catch illumination and miss surface-finish
  change; embedding distance likely the reverse. **A combination will probably beat any single
  signal — and demonstrating that is a legitimate contribution.**

**Deliverable:** correlation matrix + scatter plots (signal vs true degradation), and a ranked
recommendation of which signals to ship.

> ### GATE 2 — the honest one
> **If no label-free signal tracks accuracy loss, the drift-monitoring product thesis is wrong**,
> at least in this form. That is a genuinely valuable negative result and you should publish it
> rather than bury it. Do not proceed to Phase C by force-fitting a threshold to a signal that
> does not carry information.

---

## 5. PHASE C — The product

### XP10 · Detector ROC — tuning the paranoia dial
**Question:** at a given sensitivity, how often do we cry wolf, and how often do we miss?

Define "true degradation event" = accuracy drop beyond a threshold (e.g. −10%).
Then sweep the alert threshold and produce, for the detector itself:

- **False-alarm rate** — alerts fired while accuracy was fine.
- **Missed-degradation rate** — accuracy dropped and nothing fired.
- **Detection latency** — how many frames until the alert fires.

**This is the product.** Everyone can compute a distance; setting the threshold so it speaks when
it matters and stays quiet otherwise is the hard part, and the reason a plant would pay.

**Deliverable:** ROC (or DET) curve of the drift detector, plus a recommended operating point
with its error rates stated explicitly.

---

### XP11 · Three-tier operating envelope
**Question:** can we map continuous drift to discrete industrial action?

| Tier | Meaning | Action |
|---|---|---|
| **GREEN** | inside the validated envelope | none — trust the model |
| **AMBER** | conditions changed, model probably fine but no longer proven | review, spot-check against a human |
| **RED** | outside where this model was validated | stop automated decisions, revert to manual inspection |

Thresholds come from XP10, not from intuition. Each tier must be justified by its measured error
rates. The RED tier is the legally meaningful one — *"the model is outside its validated
envelope"* is the statement that maps to AI Act Article 15 robustness obligations.

**Deliverable:** the envelope definition + a one-page "model datasheet" stating the conditions
under which this model was validated. Nobody ships these. That is the point.

---

### XP12 · On-device cost
**Question:** can the monitor run continuously *alongside* inference on one box?

- Measure inference alone vs inference + monitoring: throughput, latency, power, memory.
- Directly reuses the concurrency methodology from `jetson-xray-panel`.
- Target: monitoring should cost a small single-digit percentage. If it costs 40%, nobody enables it.

**Deliverable:** the overhead table. Honest even if the number is bad.

---

### XP13 · Live demo loop
**Question:** does it work end to end, and can someone watch it in two minutes?

A single script that:
1. Streams held-out images through the deployed model on the Jetson
2. Displays predictions + the live drift signal + the current tier
3. **Gradually introduces drift** (e.g. lens contamination ramping over time)
4. Shows the tier flip GREEN → AMBER → RED while the model keeps confidently predicting
5. Shows, alongside, the true accuracy falling — the thing the factory would never have seen

**That last juxtaposition is the entire demo.** The model looks fine. The accuracy is collapsing.
Only the monitor knows.

**Deliverable:** 2-minute screen recording + the script.

---

### XP14 · *(stretch)* Sealed monitor — with Pierre
**Question:** can the alert be made tamper-evident?

An alert nobody can trust is worth nothing. Pierre's layer signs the monitor's output so the
evidence cannot be forged or silently disabled. Only attempt once XP01–13 are complete; this is a
joint piece, not a solo one.

---

## 6. Honest limitations — state these yourself, prominently

Put these in the README and say them out loud in any demo. Pre-empting the criticism is what makes
the rest credible.

1. **Simulated drift is not real drift.** Synthetic lens contamination is not a lens getting dusty
   over six months. This is the standard limitation of the whole field. It is also the natural ask
   to a design partner: *"we validated on simulated drift — give us six months of your real line
   data and we will validate on that."*
2. **No public leaderboard comparison.** Severstal test labels are private, so we split the train
   set. We are not claiming SOTA and do not need to.
3. **Drift detection is not novel research.** There is a substantial literature. What is
   under-served is doing it *on-device, label-free, with validated thresholds, on industrial
   vision* — and that is the framing to use.
4. **One dataset, one domain.** Steel surface. Whether the signals transfer to other industrial
   imagery is unproven until tested (see optional cross-dataset check below).

---

## 7. Repository structure

```
jetson-steel-drift/
├── README.md                  # headline results + honest limitations
├── PLAN.md                    # this document
├── data/
│   └── LICENCE_NOTE.md        # XP00 output — dataset, licence, permitted use
├── lib/
│   ├── models.py              # model definition + loading
│   ├── drift.py               # XP05 — drift transforms
│   ├── calibration.py         # XP04 — ECE, reliability diagrams, temperature scaling
│   ├── signals.py             # XP08 — label-free signal implementations
│   ├── bench.py               # timing / power (reuse from jetson-xray-panel)
│   └── power_logger.py        # tegrastats wrapper (reuse)
├── experiments/
│   ├── xp00_licence_gate/
│   ├── xp01_baseline/
│   ├── xp02_edge_deploy/
│   ├── xp03_certification/
│   ├── xp04_calibration/
│   ├── xp05_drift_harness/
│   ├── xp06_degradation/
│   ├── xp07_calibration_drift/
│   ├── xp08_signals/
│   ├── xp09_correlation/
│   ├── xp10_detector_roc/
│   ├── xp11_envelope/
│   ├── xp12_overhead/
│   └── xp13_demo/
├── results/                   # one JSON per run, machine-readable
└── figures/                   # committed
```

Each `xpNN_*/` gets its own `README.md` stating: the question, the method, the result, and what it
feeds into. Same convention as `jetson-xray-panel` — it worked, keep it.

---

## 8. Measurement hygiene (non-negotiable)

Carried over from the X-ray project, because it is what separates a demo from evidence:

- Fixed seeds everywhere; drift transforms must be exactly reproducible.
- Warm-up iterations discarded; steady-state timing only.
- Every configuration run **3×**, report mean ± std.
- Power logged continuously via `tegrastats`, aligned to the run window.
- Power mode locked (`nvpmodel`, `jetson_clocks`); record the mode in every result file.
- The frozen holdout is **never** used for tuning. Not once.

---

## 9. Deliverables

1. **The correlation result (XP09)** — label-free signals vs true degradation. The scientific core.
2. **The detector ROC (XP10)** — false-alarm vs missed-detection, with a recommended operating point.
3. **The model datasheet (XP11)** — validated operating envelope. The artifact nobody else ships.
4. **The 2-minute video (XP13)** — model confidently wrong, monitor catching it.
5. **Clean repo + README** with limitations stated up front.
6. **Two blog posts**, in this order:
   - **"When the model says 80%, is it right 80% of the time?"** (XP04 + XP07) — the calibration
     post. Publishable as soon as XP07 lands, well before the rest of the project finishes. It
     stands alone, it is useful to anyone deploying a model, and it seeds the drift argument.
   - **"The model didn't change. The steel did."** (XP09 + XP13) — the drift result, once the
     correlation is proven.

---

## 10. Optional extensions (only after XP13)

- **Cross-dataset transfer:** do the signals work on NEU-DET or DAGM without retuning? Tests
  generality, and generality is what makes it a product rather than a project.
- **Slow drift:** everything above is step-change drift. Gradual drift over thousands of frames is
  harder and more realistic — and is what actually happens in a factory.
- **Recovery detection:** does the monitor correctly return to GREEN when the lens is cleaned?
  Trivial-sounding, genuinely important for trust.

---

## 11. Notes for the coding agent

1. Read this whole file before writing code. Build in XP order; do not jump ahead.
2. Respect the gates. Gate 2 in particular — if the correlation is not there, stop and report it.
3. Every experiment writes machine-readable JSON to `results/`, so figures are trivial later.
4. Reuse `jetson-xray-panel` code wherever possible (bench, power logging, TensorRT export).
5. Never touch the frozen holdout for anything except final measurement.
6. Prefer the simpler measurement and note the assumption in a comment over blocking on ambiguity.
7. `data/` and `results/raw/` stay out of git.

---

*Last updated: 2026-07-22 · Owner: Maxime Carriere*
