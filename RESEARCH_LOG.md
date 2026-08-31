# ASSIS FOD Module — Research Log

Dated entries only. Each entry states what was actually run, on what data,
with what result — kept separate from planning notes so this file stays
useful as evidence of progress rather than a to-do list. Add a new entry per
work session; do not edit past entries except to fix a factual error (note
the correction inline, don't silently rewrite history).

Format per entry: date, what was done, what artifact it produced, what it
does *not* yet show.

---

## 2026-08-18 — Repository scaffold and gap-driven scope definition

**Done:**
- Completed a competitive/gap analysis across 8 commercial and
  government-funded FOD detection systems (QinetiQ Tarsier, Xsight
  FODetect/RunWize, Stratech iFerret, Trex FOD Finder, ELVA-1, Illuminex AI
  FODᴬᴵ, Safe Pro SPOTD/AFWERX SBIR, FAA/Volpe sUAS+FastFlow research) and
  cross-referenced FAA AC 150/5220-24's published performance requirements.
  See `docs/GAP_ANALYSIS_SUMMARY.md`.
- Scoped this module's Phase 2 MVP to a specific, defensible gap: small-object
  (sub-4cm-proxy) detection, evaluated against FAA AC 150/5220-24-style
  metrics, designed for existing CCTV/PTZ camera feeds rather than new
  dedicated hardware.
- Built the repository scaffold: data preparation pipeline
  (`src/data_prep.py`), training wrapper (`src/train.py`), inference script
  (`src/infer.py`), FAA-style benchmark script (`src/benchmark_faa.py`),
  Streamlit demo app (`app/streamlit_app.py`), and Colab training notebook
  (`notebooks/ASSIS_FOD_Training_Colab.ipynb`).
- Unit-tested the benchmark script's IoU and size-bucket logic
  (`tests/test_benchmark_faa.py`) against synthetic ground truth.

**Does NOT yet show:**
- No model has been trained yet. No accuracy, recall, or false-alarm numbers
  in this log are real until a training run and benchmark run actually
  complete against the FOD-A small-object split.
- No cross-site validation, no thermal/RGB fusion, no open-world evaluation —
  these remain roadmap items, not built or tested.
- Camera/PTZ integration is a design target, not yet implemented against a
  live video feed.

**Next planned session:** run `src/data_prep.py --download` +
`--build-split` against FOD-A, run a first training pass with `src/train.py`,
and record the first real `src/benchmark_faa.py` result in this log.

---

## 2026-08-18 (session 2) — VOC pipeline, environmental-metadata benchmark, full smoke test, positioning fix

**Corrections to session 1's work, made explicit rather than silently edited:**
- Session 1's README and `docs/GAP_ANALYSIS_SUMMARY.md` framed this module
  around "small and non-hub airports." That framing has been removed and
  replaced with a cost/deployability differentiator applicable at any
  airport size (marginal cost of one more detection head on existing camera
  infrastructure vs. a new capital purchase, amortized across ASSIS's other
  camera-based modules). This was a real positioning mistake, not a wording
  preference — corrected in `README.md` and `docs/GAP_ANALYSIS_SUMMARY.md`.
- `docs/GAP_ANALYSIS_SUMMARY.md` now states explicitly, at the top, that its
  vendor cost/technology claims have not been independently re-verified
  against primary sources and should not be cited in RFE material until
  they are.
- Session 1's `configs/fod.yaml` used an invented placeholder class taxonomy
  (`fastener`, `metal_fragment`, `tool`, ...) that was never checked against
  FOD-A's real labels. Replaced with the confirmed real starting subset
  (Wrench, Hammer, Screwdriver, SodaCan, Wood) — large/high-contrast classes
  chosen deliberately to validate the pipeline before extending to the
  actual small-fastener target classes (documented as a planned Phase 2b
  extension in `configs/fod.yaml`, with exact label names flagged as
  unconfirmed rather than guessed).

**Done, this session:**
- Confirmed via primary sources (arXiv:2110.03072, the FOD-UNOmaha/FOD-data
  GitHub repo) that FOD-A ships Pascal VOC XML annotations, 31 object
  categories per FAA guidance, and *separate* light-level (bright/dim/dark)
  and weather (dry/wet) categorization metadata — this last point was new
  information, not previously incorporated into this module's design.
- Built `src/voc_to_yolo.py` (VOC XML → YOLO txt converter), including a
  `--list-classes-only` mode specifically so real dataset class names are
  read from the data rather than assumed from memory. Tested against a
  hand-computed fixture (`tests/test_voc_to_yolo.py`, 6 tests) — exact
  coordinate math, unknown-class handling, malformed-XML handling, and
  out-of-frame bbox clamping all verified against manually worked values,
  not just "does it run."
- Extended `src/benchmark_faa.py` with an environmental-condition breakdown:
  if a FOD-A-style light/weather metadata CSV is supplied, the benchmark
  reports detection rate stratified by lighting and weather condition, not
  just size bucket. This is a genuinely new capability this session added —
  no vendor or paper reviewed in the gap analysis publishes this. Column
  names are auto-detected with an explicit override path and a printed
  warning if nothing matches, since the real CSV's schema was not verified
  (no network access to the live dataset). 5 new unit tests
  (`test_parse_metadata_csv_*`) against hand-built CSV fixtures.
- **Found and fixed a real bug via end-to-end testing, not just unit tests:**
  `benchmark_faa.py`'s labels-directory path resolution used
  `images_dir.parent.parent`, which pointed at the dataset root instead of
  the split's own `labels/` folder — this silently produced zero ground
  truth for every image instead of raising an error. Caught only because the full
  pipeline was run end to end against synthetic data with `data_prep.py`'s
  actual output layout, not a hand-rolled test directory. Fixed, refactored
  into a standalone `resolve_labels_dir()` function specifically so this
  exact path transformation is now unit-tested
  (`test_resolve_labels_dir_matches_data_prep_output_layout`).
- Added `configs/fod.yaml` augmentation settings applied from the start —
  `copy_paste: 0.3` (proactively applying the Phase 1 lesson that
  underrepresented classes never recover once training starts without
  enough of them in view) and `hsv_v: 0.5` (lighting-variation augmentation,
  an honest partial mitigation for the environmental-robustness gap — not
  equivalent to real night/rain data or thermal fusion, and documented as
  such in `configs/fod.yaml`'s own comments).
- Built `notebooks/ASSIS_FOD_Colab_Full.py`: the complete pipeline as one
  file, matching this project's established convention for a single
  runnable training file. It orchestrates the tested `src/` modules — it
  does not reimplement their logic — and has two explicit "CONFIRM BEFORE
  CONTINUING" checkpoints (real downloaded folder structure; real metadata
  CSV filename) instead of asserting unverified dataset internals as fact.
- **Ran the complete pipeline end to end** (download step stubbed with a
  synthetic 24-image VOC dataset, since Kaggle is not reachable from this
  session's network) — VOC→YOLO conversion → small-object split →
  1-epoch CPU training (yolov8n, imgsz=320, for speed) → inference →
  FAA benchmark with the environmental breakdown, all executed without
  error after the labels_dir fix above. All 18 unit tests pass
  (`pytest tests/ -v`).

**Does NOT yet show — read this carefully; these are not real results:**
- The synthetic-data training/benchmark run above proves the *pipeline*
  works, not that the model detects anything real. It used solid-color
  placeholder images and random box placement, 1 epoch, on CPU. Detection
  rate was 0% across all buckets, as expected for a smoke test, not a
  finding about FOD-A or small-object detectability.
- No model has been trained on real FOD-A data yet. That is still the
  single biggest gap between this repo and a citable result — see "Next
  planned session."
- FOD-A's real folder layout (`JPEGImages`/`Annotations` assumed by
  default in `notebooks/ASSIS_FOD_Colab_Full.py`) and the light/weather
  metadata CSV's real column names remain unconfirmed against the live
  dataset. Both are explicit manual-confirmation checkpoints in that file,
  not assumptions baked into the code.
- Camera/CCTV work: still not started, still not authorized. Nothing in
  this session touched a live camera feed or implied CHS access.

**Next planned session:** run `notebooks/ASSIS_FOD_Colab_Full.py` for real
in Colab against the actual FOD-A download — confirm the real folder
structure and metadata CSV at the two checkpoints, then record the first
real per-class training and FAA-benchmark numbers (good, bad, or mixed) in
this log.

---

## 2026-08-19 — First real run against live FOD-A data (Colab, paid tier, GPU)

**Done:**
- Ran `notebooks/ASSIS_FOD_Colab_Full.py`'s steps for real, against the
  actual downloaded FOD-A dataset (not synthetic data), in Google Colab.
- Confirmed the real downloaded folder structure at the Step 1b checkpoint:
  `FODPascalVOCFormat-V.2.1/VOC2007/Annotations/` and
  `.../VOC2007/JPEGImages/` — different from the flat `Annotations/` /
  `JPEGImages/` layout assumed by default in the script. Script paths
  adjusted accordingly for this run.
- Confirmed the real 31 FOD-A class names directly against the live
  dataset (33,793 XML files scanned, 0 malformed): Bolt, Pliers, Wrench,
  Washer, Wire, PlasticPart, LuggageTag, Cutter, Label, Nut, Nail, Battery,
  BoltWasher, MetalPart, PaintChip, SodaCan, ClampPart, Screwdriver,
  Hammer, LuggagePart, Rock, FuelCap, AdjustableClamp, BoltNutSet, Pen,
  AdjustableWrench, MetalSheet, Hose, Wood, Screw, Tape. All 5 classes
  currently in `configs/fod.yaml` (Wrench, Hammer, Screwdriver, SodaCan,
  Wood) matched exactly, including capitalization — no config edit needed.
- **New finding, not previously documented:** none of FOD-A's 31 classes
  cover tire-fragment or rubber-debris FOD (burst retreads, rubber left on
  taxiways/runways) — a real, known airfield FOD hazard. Logged as a
  limitation in `README.md` and `docs/GAP_ANALYSIS_SUMMARY.md`. No public
  dataset covering this was identified; not started.
- Ran the real VOC→YOLO conversion against all 33,793 XML files: 5,295
  label files written (matching the 5 in-scope classes), 29,177 objects in
  other classes correctly skipped as out-of-scope (not an error — expected
  given the deliberately narrow 5-class starting subset).
- Assembled the real `images/`+`labels/` pair: 5,295 image/label pairs,
  zero mismatches.
- Built the real small-object-weighted split: 4,502 base training images,
  789 additional oversampled small-object images, 793 held-out test
  images. Source composition by size bucket: small 309, medium 1,481,
  large 3,505.
- Started real training (`src/train.py`, GPU, paid Colab tier) against
  this real split. In progress as of this entry.

**Does NOT yet show:**
- Training was still running at the time this entry was written — no
  final weights, no real detection-rate or FAA-benchmark numbers yet. Do
  not cite any accuracy figure for this run until it's recorded here after
  completion.
- Tire-fragment/rubber-debris detection: still not covered, not started,
  no dataset identified — see above.
- Environmental (light/weather) metadata breakdown: not run yet this
  session — `METADATA_CSV` was left unset for this pass; the real CSV's
  filename/columns still haven't been located in the live dataset. Follow-
  up item for next session, not skipped by oversight.

**Next planned session:** record the finished training run's real
per-class detection results and `src/benchmark_faa.py` output (good, bad,
or mixed) here. Locate FOD-A's light/weather metadata CSV and re-run the
benchmark with `--metadata-csv` for the environmental breakdown.

---

## 2026-08-20 — First real training run completed (100 epochs, real FOD-A data)

**Done:**
- Completed the real training run started 2026-08-19, on the real
  4,502-base + 789-oversampled-small-object split described in that entry.
  100 epochs, yolov8n, imgsz 960, GPU (Colab paid tier). Training survived
  two real Colab disconnects along the way (interrupted once manually at
  epoch 21, disconnected again partway through epoch 82) — both times
  resumed cleanly from a checkpoint saved to Google Drive
  (`/content/drive/MyDrive/ASSIS-FOD-runs/detect/train/weights/last.pt`),
  set up specifically after the first disconnect to prevent losing
  progress. No training progress was actually lost across either
  disconnect — Colab's background execution continued on the VM between
  epochs 83–98 even while the browser view was disconnected.
- **Real validation results, on the 793-image held-out test split**
  (5 classes: Wrench, Hammer, Screwdriver, SodaCan, Wood):

  | Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
  |---|---|---|---|---|---|---|
  | all | 793 | 793 | 0.995 | 0.997 | 0.995 | 0.964 |
  | Wrench | 375 | 375 | 0.992 | 0.986 | 0.995 | 0.870 |
  | Hammer | 128 | 128 | 0.995 | 1.000 | 0.995 | 0.976 |
  | Screwdriver | 113 | 113 | 0.994 | 1.000 | 0.995 | 0.992 |
  | SodaCan | 134 | 134 | 0.995 | 1.000 | 0.995 | 0.991 |
  | Wood | 43 | 43 | 1.000 | 1.000 | 0.995 | 0.993 |

  This is the first real, non-synthetic result this module has produced —
  every earlier number in this log was either a smoke test or a planning
  estimate.

**Does NOT yet show:**
- FAA AC 150/5220-24-style benchmark (`src/benchmark_faa.py`) not yet run
  against these real weights as of this entry — planned as the immediate
  next step, same session.
- Environmental (light/weather) metadata breakdown — FOD-A's real
  metadata CSV still hasn't been located; still an open item.
- These results are on a held-out split of the *same* source dataset, not
  an independent site or camera — per `docs/FAA_AC_150_5220-24_BENCHMARK.md`,
  cross-site validation remains a real, unclosed gap, not something this
  result should be read as answering.
- Tire-fragment/rubber-debris detection: still not covered — see the
  2026-08-19 entry and `README.md`.
- Only 5 of FOD-A's 31 real classes are covered by this trained model
  (the deliberately narrow starting subset) — the small-fastener classes
  (Bolt, Nut, Washer, Screw, BoltWasher, BoltNutSet) that are the more
  operationally realistic FOD targets are not yet trained on.

**Next planned session:** run and record the FAA-style benchmark against
these real weights; plan and execute the MKS field pilot test (PPE/
RampGuard module) once authorization is confirmed in writing; consider
extending training to the small-fastener classes given how strong this
first 5-class result is.

---

## 2026-08-20 (same session) — First real FAA AC 150/5220-24-style benchmark

**Done:**
- Ran `src/benchmark_faa.py` against the trained weights above, on the
  same 793-image held-out test split. Full report:
  `docs/benchmark_results/benchmark_20260820T182741Z.md` /
  `.json`.
- **Real results by size bucket** (pixel-area proxy, not calibrated
  centimeters — see limitation below):

  | Bucket | Ground truth | Detected (TP) | Missed (FN) | Detection rate | Meets FAA's referenced 90% threshold? |
  |---|---|---|---|---|---|
  | Small | 46 | 24 | 22 | 52.2% | **No** |
  | Medium | 222 | 221 | 1 | 99.5% | Yes |
  | Large | 525 | 525 | 0 | 100% | Yes |

  False positives: 0.0101 per image. Mean localization error: 0.30% of
  frame diagonal (both pixel-based proxies, not the AC's real units — see
  limitation below).

**What this result actually means — read before citing it anywhere:**
This is a genuinely useful, credible result specifically *because* it
confirms rather than avoids the module's own founding premise: small-
object FOD detection is the real, unsolved, hard part of this problem,
consistent with the entire competitive gap analysis
(`docs/GAP_ANALYSIS_SUMMARY.md`). A result showing near-perfect detection
on every size bucket would have been suspicious, not reassuring, given
what every reviewed vendor and paper already documents about this gap.
This number should be presented honestly as evidence that the problem
this module targets is real and measured, not as evidence the problem is
solved — it is not.

**Does NOT yet show / limitations, carried directly from the benchmark
script's own printed output:**
- Size buckets are a pixel-area proxy, not calibrated real-world
  centimeters — FOD-A has no ground-sample-distance data.
- False-alarm rate is per-image, not the AC's per-day figure — converting
  requires a real deployment's scan cadence, which doesn't exist yet.
- Localization error is in pixels/frame-diagonal-percent, not meters.
- This is a held-out split of the *same* source dataset, not cross-site
  validation.
- Light/weather metadata breakdown was not run this session
  (`results_by_light_level` / `results_by_weather` both null) — the real
  metadata CSV still hasn't been located in the live dataset.

---

## 2026-08-23 — Second training run; split-reproducibility defect found and fixed

**Done:**
- Ran a second independent 100-epoch training run on the same source data
  and the same `--seed 42` (Colab, A100). Completed after several
  disconnects, resumed each time from the Drive checkpoint.

  | Class | Instances | P | R | mAP50 | mAP50-95 |
  |---|---|---|---|---|---|
  | all | 793 | 0.997 | 0.989 | 0.993 | 0.958 |
  | Wrench | 378 | 0.994 | 0.944 | 0.985 | 0.858 |
  | Hammer | 109 | 0.999 | 1.000 | 0.995 | 0.963 |
  | Screwdriver | 121 | 0.999 | 1.000 | 0.995 | 0.987 |
  | SodaCan | 151 | 0.999 | 1.000 | 0.995 | 0.988 |
  | Wood | 34 | 0.994 | 1.000 | 0.995 | 0.995 |

- **Found a real reproducibility defect by comparing the two runs.** The
  per-class test-set composition differed between run 1 and run 2 despite
  identical source data and identical `--seed 42`:

  | Class | Run 1 (2026-08-20) | Run 2 (2026-08-23) |
  |---|---|---|
  | Wrench | 375 | 378 |
  | Hammer | 128 | 109 |
  | Screwdriver | 113 | 121 |
  | SodaCan | 134 | 151 |
  | Wood | 43 | 34 |
  | **total** | **793** | **793** |

  Cause: `find_labeled_images()` in `src/data_prep.py` enumerated files with
  `Path.rglob()`, which returns filesystem order. On ext4 that order comes
  from a hash seeded per filesystem — stable on one machine, different on
  another. Seeding the shuffle does not make the split reproducible when the
  list being shuffled arrives in a different order. The two runs were on
  different Colab VMs, which is exactly the condition that exposes it.

  This was invisible in the split manifest: `n_test`, `n_train_base`, and the
  bucket counts were identical across both runs, because those are aggregates.
  Only the per-class breakdown printed by the training run revealed it.

  Fixed by sorting the enumeration before it reaches the seeded shuffle.
  Five regression tests added (`tests/test_split_reproducibility.py`),
  including a sanity test that a *different* seed still changes the split —
  without which, sorting everything and never shuffling would pass. Verified
  that the guarding test genuinely fails when the fix is reverted; the
  cross-creation-order test does not fail on ext4 and is documented as such
  in the file rather than left to look load-bearing. All 32 tests pass.

**What this means for the numbers already reported:**
- Run 1's and run 2's results were computed on **different held-out test
  sets**. They are two valid measurements, not a repeat of one measurement.
  Neither is invalidated, but they should not be described as identical
  conditions, and any claim of "reproducible with seed 42" was not true for
  runs on different machines before this fix.
- That the two runs agree within roughly a percentage point on overall
  mAP (0.995/0.964 vs 0.993/0.958) across *different* test splits is
  arguably stronger evidence of stability than two runs on one split would
  have been — but it is evidence of a different thing, and should be
  described as such.
- Splits built with `data_prep.py` from this commit onward are reproducible
  across machines. Splits built before it are not, including both runs above.

**Does NOT yet show:**
- The FAA benchmark has not yet been re-run against run 2's weights, so
  there is no second small-object detection figure to compare against run
  1's 52.2%. Until that exists, 52.2% is a single measurement.
- No cross-site validation, no calibrated size measurement, no tire-fragment
  coverage — all unchanged from the 2026-08-20 entry.

**Second FAA benchmark (same session), and what two measurements actually
support:**

Ran `src/benchmark_faa.py` against run 2's weights on run 2's held-out split.
Report: `benchmark_20260823T003420Z.md` / `.json`. **NOT PRESERVED — corrected 2026-08-31.** That file was written to a Colab VM that was later recycled, and no copy reached Drive or this repository. Run 2's figures below are transcribed from the session output and cannot be re-derived from a committed artifact; treat them accordingly. Runs 1 (`benchmark_20260820T182741Z`) and 3 (`benchmark_20260831T195805Z`) ARE committed under `docs/benchmark_results/`. Step 11 of the run 3 notebook now copies benchmark output to Drive for exactly this reason.

| Bucket | Ground truth | Run 1 detected | Run 2 detected |
|---|---|---|---|
| Small | 46 | 24 (52.2%) | 22 (47.8%) |
| Medium | 222 | 221 (99.5%) | 219 (98.6%) |
| Large | 525 | 525 (100%) | 524 (99.8%) |

False positives per image: 0.0101 (run 1) vs 0.0214 (run 2). Mean
localization error: 0.296% vs 0.322% of frame diagonal.

Note the bucket-level ground-truth counts are identical (46 / 222 / 525)
even though the per-class composition differed. That is consistent with the
splitter's design: it draws `test_frac` from each size bucket, so bucket
totals are fixed by the source composition while *which* images land in each
bucket varied under the ordering defect described above.

**Statistical reading — this corrects how the small-object figure should be
quoted:**
- The two runs differ by 0.043 (52.2% vs 47.8%), against a standard error of
  the difference of 0.104. That is 0.42 standard errors: the runs are not
  distinguishable from each other. The measurement is stable.
- With only 46 small-object instances per run, a single run's 95% Wilson
  interval is roughly ±14 percentage points (run 1: [38.1%, 65.9%]; run 2:
  [34.1%, 61.9%]). **Quoting "52.2%" implies a precision this sample does not
  support.** Earlier entries in this log and the first versions of the
  outreach documents did exactly that; this entry supersedes them on that
  point.
- Pooled across both runs (46/92): **50.0%, 95% CI [40.0%, 60.0%]**. This is
  the figure to quote, with the interval attached.
- The conclusion is robust regardless: the entire confidence interval sits
  far below the AC's referenced 90% threshold. "Small-object detection falls
  well short of the FAA threshold" is well supported; any specific decimal
  is not.
- For reference, reaching a ±10-point margin at p≈0.5 would need roughly 96
  small-object instances; ±5 points would need roughly 384. The current
  held-out split provides 46. Enlarging the small-object test set is
  therefore a prerequisite for any tighter claim, and is a more useful next
  step than further repeat runs at this sample size.

**Does NOT yet show:**
- Medium and large results (98.6–100%) are on the same held-out-split basis
  and carry the same limitation as everything else here: not cross-site, not
  calibrated to physical size.
- The false-positive rate roughly doubled between runs (0.0101 → 0.0214).
  Both are small in absolute terms, but with no per-day conversion available
  this is not comparable to the AC's ceiling, and two points is not a trend.
- Light/weather stratification still not run; metadata CSV still not located.

**Next planned session:** enlarge the small-object evaluation set — either by
raising `test_frac` for the small bucket specifically or by extending to
FOD-A's small-fastener classes (Bolt, Nut, Washer, Screw, BoltWasher,
BoltNutSet), which are both more numerous and more operationally
representative than the current five-class subset. Repeat runs at n=46 will
not narrow the interval.

---

**Next planned session (from 2026-08-20):** investigate the small-object miss cases
specifically (which of Wrench/Hammer/Screwdriver/SodaCan/Wood account for
the 22 misses, and at what size/confidence) to decide whether more
epochs, different augmentation, or more small-object training data is the
right next lever — rather than assuming any one fix. Continue toward the
MKS pilot test and the light/weather metadata CSV as separately planned.

---

## 2026-08-28 — Vendor collateral reviewed; one gap claim withdrawn

**Done:**
- Reviewed primary-source vendor material obtained at the AAAE Airport
  Operations & Technology Symposium: Illuminex AI's InspectEx platform
  overview, its FOD product sheet (fod.ai), and its PIDS product sheet;
  plus Moog's Tarsier FOD product page and the Moog/QinetiQ licensing
  announcement. Until now the landscape table rested on secondary sources.

**Corrections made to `docs/GAP_ANALYSIS_SUMMARY.md`:**

1. **A gap claim was wrong and has been withdrawn.** The analysis previously
   asserted that "no existing system amortizes FOD detection cost across
   other safety functions" and that "every vendor reviewed sells FOD
   detection as a standalone purchase." Illuminex AI's InspectEx is exactly
   such a platform: FOD AI, PIDS AI, SnowPro AI, EdgeGuard, and a
   forthcoming Surface AI share sensor and cloud infrastructure, with
   "additional plug and play applications" marketed as the expansion path.
   The claim is struck through in place rather than deleted. It had already
   propagated into outreach documents, which is why it is recorded here
   rather than quietly fixed.

2. **The surviving differentiator is narrower.** Every reviewed system,
   Illuminex included, requires dedicated sensor hardware and delivers
   coverage as a periodic inspection pass. Illuminex lowers the vehicle cost
   by mounting on vehicles the airport already owns, but the sensors are
   still a new purchase. The defensible ASSIS distinction is *existing fixed
   CCTV/PTZ cameras and continuous coverage* versus *new sensors and
   inspection passes* — not multi-function amortization, which Illuminex
   also does.

3. **The AC-terms claim needed a qualifier.** Vendors do publish performance
   figures; Moog's Tarsier page advertises "100% detection out to 3,168
   feet" and a best-in-class ranking in FAA testing. That is a range claim
   with no object size class, no false-alarm rate, and no location accuracy,
   so it does not answer AC 150/5220-24's questions and is not comparable
   across systems. The gap is the absence of a *common measure*, not the
   absence of published numbers. Reworded accordingly.

4. **Tarsier attribution updated.** QinetiQ developed it; Moog holds an
   exclusive license and commercializes it through Moog Digital Airfield
   Solutions across Europe, Asia Pacific and the Americas.

5. **Illuminex maturity understated.** Previously logged as "2026 trials."
   It is a productized platform with Standard and Premium tiers (50 ft
   inspection width at up to 25 mph; 100 ft at up to 50 mph, expandable with
   LiDAR and thermal). Updated.

**Does NOT change:**
- The environmental-stratification gap: re-confirmed against the new
  collateral. No reviewed vendor publishes results broken out by lighting or
  weather condition.
- The small-object result (50.0%, 95% CI [40.0%, 60.0%], n = 92) and every
  limitation recorded on 2026-08-23.

**Action required outside this repository:** the withdrawn claim appears in
outreach and petition materials and must be corrected there before any of
them are sent.

---

## 2026-08-31 — Run 3: reproducible split built, training started

**Done:**
- Built the run 3 split with the corrected enumeration ordering and the
  enlarged small-object held-out fraction. Source: FOD-A Pascal VOC mirror,
  33,793 XML files, 0 malformed, 31 distinct class names confirmed against
  the live dataset. `configs/fod.yaml` restricts training to the same
  five-class subset used in runs 1 and 2 (Wrench, Hammer, Screwdriver,
  SodaCan, Wood), giving 5,295 labelled images.

  | Bucket | Source images | Held out | Fraction |
  |---|---|---|---|
  | small | 309 | **123** | 0.40 |
  | medium | 1,481 | 222 | 0.15 |
  | large | 3,505 | 525 | 0.15 |
  | **total** | **5,295** | **870** | |

  Train base 4,425, plus 558 oversampled small-object images (186 × 3).

- **Split fingerprints — the point of this run.** Anyone rebuilding from the
  same source with `--seed 42 --test-frac 0.15 --small-test-frac 0.40` must
  obtain these digests. Aggregate counts agreeing is *not* sufficient: that
  is exactly what agreed across runs 1 and 2 while the membership differed.

  ```
  train        f39b2c6315530d27f1c180ac1a712b94900672d449aebec0de5e441a6db8d484
  test         ce1efeeb3899ebe5111bf7092cef4115db19aa0323b0f5433d7f29bf90f78e7f
  test_small   476227dd9dec9d26bf9d2a529754ac6bce2c49fd802cf4f1328e7cf835aab321
  test_medium  92aab47ea0ba6be548da7e22437aaafe99be67e6165c024e8e21702da06e0546
  test_large   8b9488036a2bbac3cd02d21d662a5e0c81fe03f862f3bbffe5133825e5b9a741
  ```

- Medium (222) and large (525) held-out counts are **identical to runs 1 and
  2**, as intended: `--test-frac` was unchanged and the source composition is
  fixed, so the only deliberate difference is the small bucket.

**What this changes about precision:**
- At n = 123 and p ≈ 0.5 the 95% interval is roughly ±8.8 points, against
  ±14 at n = 46. The result will still be an interval and must still be
  quoted with the interval attached — a wider sample does not license a
  point estimate.

**Defects found and fixed while running this (notebook, not pipeline):**
- `voc_to_yolo.py` defaults `--config` to the relative path
  `configs/fod.yaml`, which resolves against the working directory rather
  than the repository. Two notebook cells omitted the argument and failed.
- The image-copy step used `cp {IMAGES}/*.jpg ... 2>/dev/null`. A ~34,000-file
  glob overflows the shell argument list, and redirecting stderr hid the
  failure until Step 6 reported an empty dataset three cells later. Replaced
  with a per-file `shutil.copy2` loop that prints its count. **Suppressing
  stderr on a step whose failure surfaces downstream is the error here, not
  the glob.**
- `src/colab_helpers.py` had never been pushed to the public repository, so a
  fresh clone could not import `find_voc_root` or `find_latest_run`. Now
  committed with its 13 tests.

**Does NOT yet show:**
- Training is in progress; no run 3 accuracy or benchmark figures exist yet.
  Every detection figure currently quoted in `README.md` and elsewhere still
  derives from runs 1 and 2 (small 50.0%, 95% CI [40.0%, 60.0%], n = 92) and
  remains the figure of record until run 3's benchmark completes.
- The run was interrupted once and resumed from the Drive checkpoint at epoch
  5. An interrupted run is not bit-identical to an uninterrupted one, because
  dataloader shuffling state does not survive the restart. This does not bear
  on the reproducibility claim, which concerns the *split*, not the training
  trajectory. Runs 1 and 2 were likewise resumed multiple times.
- Light/weather stratification not yet run. The categorization annotations
  ship with FOD-A's original-format distribution, not the Pascal VOC mirror
  used here; `notebooks/ASSIS_FOD_Run3_Reproducible.py` Steps 10–10c download
  it, search for the CSV, and validate its counts against the paper before
  any stratified result is produced.

**Next:** run the FAA benchmark against run 3's weights, commit the output to
`docs/benchmark_results/`, and propagate the corrected figures to `README.md`,
`app/streamlit_app.py`, the two outreach PDFs, the expert-letter template, and
the RFE cover letter in a single pass.

---

## 2026-08-31 (session 2) — Run 3 benchmarked; AC thresholds verified against
## the primary source; stratification blocked and characterised

**Run 3 FAA benchmark** (`docs/benchmark_results/benchmark_20260831T195805Z.*`),
run 3 weights on run 3's held-out split, imgsz 640, conf 0.35, IoU 0.5:

| Bucket | Detected | Rate | 95% Wilson CI | n |
|---|---|---|---|---|
| Small | 64 | **52.0%** | **[43.3%, 60.7%]** | 123 |
| Medium | 220 | 99.1% | | 222 |
| Large | 523 | 99.6% | | 525 |

False positives per image 0.0092. Mean localization error 0.284% of frame
diagonal.

**This is now the figure of record: 52.0%, 95% CI [43.3%, 60.7%], n = 123.**
The interval is ±8.7 points against ±13.9 for either earlier run.

**Not pooled with runs 1 and 2, deliberately.** All three splits draw from the
same 309 small-object source images, so they are not independent samples;
pooling would understate the interval. Run 3 supersedes them as the reported
figure because it is reproducible (fingerprinted) and has the largest n. Runs
1 and 2 remain in this log as earlier, consistent measurements on splits that
cannot be rebuilt.

**The three measurements agree.** 52.2%, 47.8%, 52.0%. Run 3 sits 0.29 standard
errors from the earlier pooled figure — indistinguishable. The small-object
shortfall is a stable property of the model, not sampling noise.

**Overall mAP fell slightly and this is expected**, not a regression: mAP50
0.988 / mAP50-95 0.951 against 0.995/0.964 (run 1) and 0.993/0.958 (run 2). The
test set now holds 123 small-object images instead of 46, so it is harder. A
higher score on a harder test set would have been the suspicious result.

**Secondary metrics are best-of-three but are NOT claimed as improvements.**
False positives per image (0.0092 vs 0.0101, 0.0214) and localization error
(0.284% vs 0.296%, 0.322%) are single measurements on different test sets with
no intervals attached. Best-of-three is what chance produces a third of the
time.

---

**AC 150/5220-24 thresholds verified against the primary document.** Until now
`configs/fod.yaml` carried these values from secondary reading. The AC itself
(dated 09/30/2009) was obtained and read on this date. All five values are
correct; each now carries its section reference in the config file.

- 3.2.b(1)(c) — 90% of a specified group of objects, each no larger than 4 in
  (10 cm) in any dimension, within a 100 ft square.
- 3.2.b(2) — location information within 16 ft (5.0 m).
- 3.2.b(1)(d) — two objects no more than 10 ft (3 m) apart identified separately.
- 3.2.b(7)(a) — false alarms not to exceed one per day (visual) or three per day
  (non-visual), averaged over any 90-day period.

**One qualification now recorded rather than glossed:** the AC's 90% applies to
a specified group of TEN object types placed in a 100 ft square, not to
arbitrary small debris. This module applies the figure to a bounding-box-area
"small" bucket. Related, not identical, and stated as such in
`src/benchmark_faa.py`'s limitations block.

**Two AC requirements found that strengthen the module's framing:**

- 3.2.b(6)(c): *"All systems must demonstrate detection performance during
  daylight, nighttime, and dawn/dusk operations."*
- 3.2.b(6): systems *"must demonstrate the detection performance under both clear
  and inclement weather conditions"*, with site-specific specifications for
  clear weather, inclement weather, and post-storm recovery time.

The lighting and weather stratification this module reports is therefore
demonstrating something the Advisory Circular **requires**, not a reporting
refinement invented here. Section 2.2.c adds that *"Dark-colored items made up
nearly 50% of the FOD collected"*, so lighting performance is not a corner
case. Section c also supplies primary-source backing for the small-object
scope: *"over 60% of the FOD items were made of metal, followed by 18% ...
rubber"* and *"Common FOD dimensions can be 1 in. by 1 in. (3 cm by 3 cm) or
smaller."* The 18% rubber figure independently supports the tire-fragment gap.

---

**Environmental stratification: located, decoded, and blocked.**

The categorization annotations were found. They ship with FOD-A's
ORIGINAL-format distribution (8.9 GB as downloaded), not the Pascal VOC mirror
used for training here:
`FullDatasetV.2.1-400x400/All_Dataset_Utility_Files/FOD_categorization_annotations.csv`
— 33,863 rows, columns File / Weather / Light, values as INTEGER codes.

**Code mapping, confirmed two independent ways:**

| | 0 | 1 | 2 |
|---|---|---|---|
| Weather | Dry (26,647) | Wet (7,216) | — |
| Light | Bright (17,012) | Dim (12,464) | Dark (4,387) |

Confirmed by matching row counts to the FOD-A paper's Table I, and against the
dataset's own `category_information.txt` ("Weather: [Dry,Wet,] Light:
[Bright,Dim,Dark,]"). **Note that 0 is Bright and 2 is Dark** — an assumed
ordering would invert the finding and report best performance in darkness.

**The join is not possible with the mirror in hand, and the stratification was
therefore NOT run.** The VOC mirror contains 33,793 images numbered
contiguously 000000–033792 with zero gaps: 70 images were dropped and the
remainder renumbered, so the correspondence to the original ordering cannot be
recovered from filenames. Assigning labels on a guessed alignment would
produce a plausible-looking result that was wrong for most of the dataset.

The earlier prediction of a ~70-row discrepancy in the CSV was wrong in
detail: the ORIGINAL distribution matches the paper exactly at 33,863, and it
is the Pascal VOC mirror that carries 33,793.

**Closing this requires retraining from the original-format distribution**,
where filenames and metadata correspond. That is a specific, characterised
next step rather than an open question — which is a better position than this
gap was in yesterday, even though no stratified number was produced.

**Also added:** `src/benchmark_faa.py` now accepts `--imgsz` and records it in
its output, since inference resolution materially changes small-object recall
and a benchmark that does not state its own resolution cannot be compared with
another one.

---

## 2026-08-31 (session 3) — Inference-resolution sweep: negative result, and
## what it rules out

**Done:** benchmarked run 3's weights against run 3's held-out split at three
inference resolutions. Same weights, same 123 small-object images, same
confidence and IoU thresholds — only `--imgsz` varied.

| imgsz | Small (n=123) | Medium (n=222) | Large (n=525) |
|---|---|---|---|
| 640 | 64/123 = 52.0% | 99.1% | 99.6% |
| 960 | 64/123 = 52.0% | 99.1% | 99.6% |
| 1280 | 64/123 = 52.0% | 99.1% | **95.8%** |

**Small-object detection is unchanged at every resolution — literally the same
64 detections.** Raising inference resolution does not help here, and this
lever is now ruled out.

**Why, and it is not a null finding:** the Pascal VOC mirror used for training
and evaluation contains **300x300** images. Ultralytics already upscales those
to fit imgsz 640, so 960 and 1280 interpolate further from a 300-pixel source.
There is no additional detail to recover because it was never present in this
distribution. Resolution cannot be the constraint when the images are already
being enlarged past their native size.

**The sweep was not a no-op.** Large-object detection fell from 99.6% to 95.8%
at 1280, which confirms `--imgsz` reached the model and had an effect. It also
shows that over-upscaling costs accuracy rather than being merely neutral.

**This converges with the stratification blocker recorded earlier today.** Two
independent problems now point at the same cause and the same fix:

1. The light/weather metadata cannot be joined to the VOC mirror, because that
   mirror renumbered its images contiguously after dropping 70.
2. Inference resolution cannot help, because that mirror is 300x300
   downsampled from the original distribution's 400x400.

**Next run, therefore: train from `FullDatasetV.2.1-400x400`** — the
original-format distribution, already downloaded (8.9 GB). That gives higher
resolution source imagery AND filenames that join to
`FOD_categorization_annotations.csv`, unlocking the environmental
stratification. One run addresses both.

**What remains open on small-object detection after this:** more small-object
training data (only 186 small images are in the current training set),
extension to the small-fastener classes (Bolt 3,300 / Washer 2,139 / Nut 1,303
/ BoltWasher 1,017 / BoltNutSet 514 / Screw 157 objects, all currently excluded
by the five-class config), and tiled inference. None of these are ruled out;
resolution now is.

**Benchmark artifacts from this sweep were written to /content/imgsz_sweep and
are NOT committed** — they are three runs of the same weights differing only in
one flag, and the table above records the result. The committed artifacts under
`docs/benchmark_results/` remain the runs of record.
