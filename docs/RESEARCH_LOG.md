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

---

## 2026-09-01 — Anecdotal deployed-app failure: closed taxonomy forces a wrong
## label on an out-of-distribution object

**Not a benchmark measurement — a single manual test against the deployed
Streamlit demo, recorded because it is a clean illustration of two limitations
already named in this log, not because n=1 changes any reported rate.**

Uploaded a professional aviation stock photo (shallow depth of field, an
out-of-focus fighter jet filling the background) of a "48 FW – Golden Bolt"
novelty/award bolt prop lying in the foreground. Run 3's weights returned one
detection: class "Wrench," confidence 0.76, with the bounding box drawn over
the blurred aircraft in the background — not over the bolt in the foreground.

**Two compounding causes, not one bug:**

1. **Closed five-class taxonomy with no reject option.** Run 3 is trained on
   only Wrench, Hammer, Screwdriver, SodaCan, Wood (see the 2026-08-31 sweep
   entry above). A bolt is not one of them — and notably, FOD-A itself *does*
   have a Bolt class (3,300 objects), it is simply one of the classes this
   config currently excludes. A closed-set classifier cannot output "unknown
   object"; it is forced onto its closest known label regardless of fit. The
   0.76 confidence reflects certainty in that forced choice, not correctness
   of the label.
2. **Severe domain shift from FOD-A's image style.** FOD-A training/eval
   images are flat, uncalibrated, airfield-camera-style shots. This upload was
   a staged studio-quality stock photo with heavy background bokeh. The
   mislocalized box (on the blurred jet, not the sharp bolt) is consistent
   with the model keying on shape/texture cues that correlated with "tool" in
   training data, without the training distribution to anchor it to the
   actually-salient object when the image looks nothing like FOD-A.

For contrast: an earlier upload the same session — a handheld wrench, flat
lighting, plain pavement, phone-camera framing much closer to FOD-A's own
style — was detected correctly (class and box both right). The difference
between the two results tracks distribution shift, not random noise.

**Why this is worth recording rather than discarding:** it is a concrete,
reproducible illustration of two gaps this log already documents in the
aggregate (small/excluded-class coverage; no validation outside one dataset's
image style) rather than a new finding, and it is a useful caution against
over-trusting a single confident-looking detection in the demo — exactly the
risk `app/streamlit_app.py`'s own measured-performance panel is there to
guard against.

**Not evidence of anything quantitative.** One image is feasibility evidence
of a failure mode, not a rate, and is not added to any benchmark table.

**What would actually address this, for the record:** (a) extending the
training config to include Bolt and the other currently-excluded small-
fastener classes already present in FOD-A (see 2026-08-31 sweep entry), which
run 4 does not yet do but could be scoped to; (b) evaluating on imagery that
matches the deployment's actual expected input style, not just FOD-A's own
held-out split, before claiming any cross-domain robustness — currently
undone and already listed as a limitation in the app itself.

---

## 2026-09-02 — SAHI sliced inference: negative result on FOD-A, as predicted

**Done:** benchmarked run 3's weights against run 3's exact held-out split
(fingerprints reproduced and verified byte-for-byte against the 2026-08-31
entry before trusting anything below) under two conditions: a plain
full-image pass, and SAHI sliced inference (512x512 tiles, 20% overlap).
`src/benchmark_faa.py` gained a `--sahi` flag for this comparison; everything
downstream of the prediction call (IoU matching, bucketing, the FAA-threshold
report) is identical between the two runs, so they are directly comparable.

| Condition | Small (n=123) | Medium (n=222) | Large (n=525) | FP/image | Mean loc. error |
|---|---|---|---|---|---|
| Plain | 64/123 = 52.0% | 220/222 = 99.1% | 523/525 = 99.6% | 0.0092 | 0.284% |
| SAHI (512px, 20% overlap) | 63/123 = 51.2% | 220/222 = 99.1% | 523/525 = 99.6% | 0.0080 | 0.288% |

**No real difference.** One fewer small-object detection under SAHI (63 vs.
64) is noise at n=123, not a signal — medium and large are byte-identical
between conditions. This is a negative result, in the same category as the
2026-08-31 inference-resolution sweep, and it was the predicted outcome
going in, stated explicitly in this repo before the run: FOD-A's Pascal VOC
mirror images are already only 300x300, and tiling an already-small source
image does not manufacture pixel detail that was never captured. SAHI had
nothing to recover here.

**This does not rule out SAHI in general — it rules it out for FOD-A's own
benchmark specifically.** SAHI's plausible use case remains higher-resolution
real-world deployment photos, where a small object is a tiny fraction of a
much larger frame — a case this held-out split, by construction, does not
represent. That remains untested. Do not read this entry as "SAHI doesn't
work"; read it as "SAHI has nothing to add on a 300x300 source," which is a
narrower and now-confirmed claim.

**Converges with everything else pointing at run 4.** Three independent
findings now trace to the same root cause (FOD-A's Pascal VOC mirror is a
downsampled, unstratifiable 300x300 mirror of a 400x400 original): the
inference-resolution sweep, the light/weather stratification block, and now
this SAHI result. Retraining from `FullDatasetV.2.1-400x400` (run 4, written
2026-08-31, not yet executed) remains the single next step most likely to
move more than one of these at once.

**Environment note:** this run required rebuilding `/content/fod-a-yolo` and
reinstalling `ultralytics`/`sahi` from a fresh Colab VM — neither survives a
VM recycle, only Google Drive artifacts (weights, split fingerprints) do.
The rebuilt split's fingerprints matched the 2026-08-31 recorded ones
exactly, confirming the rebuild reproduced run 3's actual split rather than
a similar-looking new one.

**Benchmark artifacts** written to
`docs/benchmark_results/sahi_experiment/` (both JSON+MD reports) and to
`/content/drive/MyDrive/ASSIS_FOD_sahi_experiment/` on Drive.

---

## 2026-09-03 — Run 4 executed: original-format retrain, and the light/weather
## stratification actually produced (first time, after fixing a real bug)

**Done.** `notebooks/ASSIS_FOD_Run4_Original400.py` (written 2026-08-31) was
run for the first time, on Colab Pro (T4/L4), against the original-format
FOD-A distribution (`FullDatasetV.2.1-400x400`, 8.9 GB, gdown file id
`1lLBJXXaQCWaFa-1MeLAANPpSwMhCJqGh`) rather than the Pascal VOC mirror runs
1-3 used.

**Real layout, confirmed by looking, not assumed.** The original-format
distribution is organised per object type (`Battery1/`, `cutter2/`,
`ClampPart2/`, ...), each with its own `Annotations/*.xml` (VOC format,
image size read from the XML itself) + `frame/*.PNG` pair — not the flat
`Annotations/`+`JPEGImages/` pair the VOC mirror used. `src/voc_to_yolo.py`'s
`--voc-dir` flag was built for the flat layout, so a new conversion step was
written (Step 5b, not present in the 2026-08-31 draft) to walk each
per-object folder and convert it. **Collision risk found and handled, not
assumed away:** every object folder reuses `frame_000000`, `frame_000001`...
independently, so converting into one flat output directory keyed only by
frame number would silently overwrite files across different object types.
Every converted image/label is instead named `{object_folder}__{frame_stem}`
(e.g. `cutter2__frame_000094`), with an assertion that raises loudly on any
collision rather than allowing a silent overwrite. Verified against
synthetic data shaped exactly like the real layout before running it against
the real 33,863-file dataset.

**Training.** YOLOv8n, 100 epochs, the same 5-class scope as run 3 (Wrench,
Hammer, Screwdriver, SodaCan, Wood), imgsz 640, seed 42. Completed in 0.811
hours on an L4. Per-class validation: Wrench mAP50 0.998 / mAP50-95 0.801,
Hammer 0.998/0.986, Screwdriver 0.998/0.984, SodaCan 0.998/0.991, Wood
0.995/[email protected] — consistent with run 3's per-class shape.

**FAA-bucket benchmark:** small 52.0% (64/123), medium 99.5% (221/222),
large 99.8% (524/525). This is within one detection of run 3's published
figures (523/525 large, 220/222 medium, 64/123 small identical) despite a
different image source (400x400 original vs. 300x300 mirror) and an
independently-rebuilt split. Read together with the 2026-08-31
inference-resolution sweep and the 2026-09-02 SAHI result, this is a fourth
independent line of evidence that the small-object ceiling is a real
detection-quality limit, not an artifact of the VOC mirror's downsampling —
moving to a higher-resolution source did not move this number.

**A real bug found and fixed: the metadata join silently matched nothing on
the first attempt.** The stratified benchmark call reported
`"[metadata] Loaded light/weather labels for 33863 images"` (the CSV parsed
fine) but returned `results_by_light_level: null` and
`results_by_weather: null` — the join to actual test images was silently
producing zero matches. Root cause, confirmed against the live CSV: its
`File` column uses Windows-style backslash paths
(`Battery1\frame\frame_000000.PNG`), which `pathlib` does not split on
under Linux/Colab — the old code's `Path(raw_name).stem` left the
backslashes embedded in the lookup key instead of extracting a clean
filename, and separately discarded which object folder a frame came from,
even though that folder is required to disambiguate the row (see the
collision note above — the same `frame_000094` name exists under many
different object folders with different weather/light values). Fixed in
`src/benchmark_faa.py`'s `parse_metadata_csv`: split on both `/` and `\`,
and key metadata by `{object_folder}__{filename_stem}` — exactly the naming
scheme Step 5b's conversion uses for its output images, so the two now
line up. Also translated the CSV's raw numeric condition codes (0/1/2) to
the documented labels (bright/dim/dark, dry/wet) instead of leaving them as
digits in the report. Regression test added
(`test_parse_metadata_csv_handles_windows_backslash_object_folder_paths`)
pinning the real row shape. All 70 tests pass.

**After the fix, the join matched 100% of the test set** (489+286+95 = 870
for light, 705+165 = 870 for weather — both equal to `n_images_evaluated`
exactly, versus zero before). Stratified result:

| Light level | Detection rate | n |
|---|---|---|
| Bright | 87.5% (428/489) | 489 |
| Dim | 100% (286/286) | 286 |
| Dark | 100% (95/95) | 95 |

| Weather | Detection rate | n |
|---|---|---|
| Dry | 91.3% (644/705) | 705 |
| Wet | 100% (165/165) | 165 |

This is, as far as this project has found, the first FOD detection system —
commercial or research — to report performance broken out by lighting and
weather condition in the terms AC 150/5220-24 section 3.2.b(6)/(6)(c)
actually asks for (see `docs/GAP_ANALYSIS_SUMMARY.md`'s archived analysis
for why no reviewed vendor does this).

**Open question, stated rather than resolved either way:** detection is
*lower* in bright light than in dim or dark, which is the opposite of a
naive expectation. Two explanations are both plausible and this benchmark
cannot yet distinguish them: (a) a real effect — glare/overexposure hurting
detection more than low light does, or (b) a size-distribution confound —
if the dim/dark buckets happen to be dominated by medium/large objects
(already ~99%+ regardless of lighting) and contain few of the hard small
objects, 100% would follow from bucket composition alone, not from the
model handling darkness well. Distinguishing these requires a joint
size-bucket x light-level breakdown, which `src/benchmark_faa.py` does not
currently produce. **Do not cite "the model performs better in the dark"
as a finding until that cross-tabulation is done** — recorded here as the
immediate next step for this specific result, not claimed.

**Split fingerprints** (SHA-256 over sorted membership, from
`split_manifest.json`):

```
train        c895a6537dfddb39e526c818f070504c478a6d889b6a45d1d39c17290b3232da
test         a08ef0b577488cc3f93b7df0f86f0431bb1f009977a216a985c17c8192d84c5c
test_large   f673a26e46a8e609f1c5491de52dc5b340ff2d814b94ed90993b8f31a31b41ac
test_medium  de41936ebf540e9a9a86cd1c2a3ebd99807b05945e42b1a7ced902dc20c7c0f0
test_small   79a9e094e1b4fe07921f875594e02e9d9e72bd763befa126aa54883ffed86ec5
```

Confirmed distinct from run 3's recorded fingerprints (`train
f39b2c6315530d27f1c180ac1a712b94900672d449aebec0de5e441a6db8d484`, `test
ce1efeeb3899ebe5111bf7092cef4115db19aa0323b0f5433d7f29bf90f78e7f`, `test_small
476227dd9dec9d26bf9d2a529754ac6bce2c49fd802cf4f1328e7cf835aab321`) — positive
evidence run 4 built its own independent split from the original-format
source, not a coincidental reuse of run 3's VOC-mirror split.

**Benchmark artifacts:** `docs/benchmark_results/benchmark_20260903T194949Z.*`
(plain) and `docs/benchmark_results/benchmark_20260903T195007Z.*` (with
metadata, post-fix) — committed alongside this entry. The earlier
`benchmark_20260903T185941Z.*` / `T190001Z.*` pair from before the
metadata-join fix is not committed; it reflects the same detection numbers
but a `null` (broken) stratified breakdown, and would only be confusing
alongside the corrected pair.

---

## 2026-09-04 — Streamlit app rebuilt on real data; Run 5 prepared (class scope extended to small fasteners)

**Done — Streamlit app:**
- Rebuilt `app/streamlit_app.py` as a multi-page app (Dashboard, Image
  Detection, Benchmark Performance, Environmental Conditions, Methodology,
  Limitations & Roadmap) per the project's UX/UI specification, implementing
  only the pages the repository can actually support with real data.
- Removed the previous hard-coded `MEASURED_PERFORMANCE` tuple (stale run-3
  numbers no longer matching any committed benchmark file) and added
  `src/benchmark_report.py`, a new pure, tested module that reads the
  committed `docs/benchmark_results/*.json` reports directly. Every number
  the app shows is sourced from those files; a `metadata_accounting()`
  cross-check surfaces unmatched light/weather-metadata counts explicitly
  (870/870 matched on the current run 4 report) rather than hiding them.
- Verified with `streamlit.testing.v1.AppTest` against all six pages (no
  exceptions) and spot-checked every displayed figure against the source
  JSON byte-for-byte.
- Removed a specific named-airport (CHS) reference from the Limitations
  page at the user's request — the no-CCTV-authorization statement now
  reads as a general policy rather than naming one airport, since naming
  it there could misleadingly imply it is a specific candidate under
  consideration.
- Real-world spot check: the user tested the live app with an uploaded
  photo of loose hardware (nuts, bolts, washers, rocks). The model
  classified a bolt as "SodaCan" (0.81 confidence). This is not a code
  defect — the model's trained class list (5 classes, see below) has no
  fastener category, so an out-of-scope object is necessarily forced into
  the nearest of the 5 it does know. This single-image observation is
  recorded here as the concrete motivation for the class-scope extension
  below, not as a validated accuracy finding.

**Done — Run 5 preparation (not yet executed):**
- Extended `configs/fod.yaml`'s `classes:` list from 5 to 12: added Bolt,
  Washer, Nut, BoltWasher, BoltNutSet, Screw, and Nail — FOD-A's
  small-fastener categories, which are the actual small-object gap this
  module targets (see docs/GAP_ANALYSIS_SUMMARY.md), and which directly
  address the SodaCan misclassification above.
- **Provenance caveat, stated explicitly in `configs/fod.yaml` itself:**
  these 7 class names were confirmed present via
  `src/voc_to_yolo.py --list-classes-only` on 2026-08-19, but that scan was
  run against the Pascal VOC MIRROR distribution
  (`FODPascalVOCFormat-V.2.1/VOC2007/`), not the ORIGINAL-format 400x400
  distribution Run 5 will actually train on. The two distributions are
  known to differ (the mirror dropped 70 images and renumbered the rest —
  see the 2026-08-31 findings above), so the class vocabulary is NOT
  assumed to carry over unchecked. `notebooks/ASSIS_FOD_Run5_Fasteners.py`
  Step 4b re-runs the class scan directly against the original-format
  distribution before conversion proceeds, and the notebook explicitly
  instructs stopping if any of the 7 new classes comes back with a zero
  count there.
- Added `convert_original_format_distribution()` to `src/voc_to_yolo.py` —
  a real, tested replacement for the ad hoc conversion cell that was
  pasted directly into the live Colab session during run 4 and never
  became part of the committed, tested source tree (a gap explicitly
  flagged after run 4). It writes both images and collision-safe-named
  labels (`{object_folder}__{filename_stem}`, matching
  `benchmark_faa.py`'s existing convention), and is exercised by a new
  regression test reproducing the exact real defect class this exists to
  prevent: two different object folders (`Bolt1/`, `Nut3/`) both containing
  a `frame_000000` file, asserting both survive conversion under distinct
  names rather than one silently overwriting the other. A second new test
  covers an in-scope object with no matching image file, and an
  out-of-scope object being counted rather than silently dropped.
- Added a CLI mode, `--original-format-root`, to `src/voc_to_yolo.py`
  alongside the existing `--voc-dir` mode, so the new function is usable
  from the notebook the same way the existing flat-VOC conversion is.
- Wrote `notebooks/ASSIS_FOD_Run5_Fasteners.py`, following run 4's
  structure (same split parameters — seed 42, test_frac 0.15,
  small_test_frac 0.40 — held constant so the split METHOD doesn't change,
  only the class scope) with the new Step 4b verification and Step 5b
  conversion call added, plus a Step 1b that fails fast if the repository's
  `configs/fod.yaml` on GitHub doesn't yet have the 12-class list.
- Test suite: 77 passed (10 for `benchmark_report.py`, 2 new for
  `voc_to_yolo.py`'s new function; unchanged tests for `convert_directory`,
  `parse_metadata_csv`, and everything else still pass).

**Not yet done / explicitly not claimed:**
- Run 5 has NOT been executed. No training has happened, no benchmark
  numbers exist for the 12-class model, and none should be assumed.
- Run 5's numbers, once they exist, will NOT be directly comparable to run
  4's: run 4 was scored over 5 classes, run 5 over 12, with a different
  underlying labeled-image set and therefore a different split. State this
  wherever a run 5 number is eventually reported.
- The Nail class's real object count in this dataset has not been tallied
  anywhere in this log (only the other 6 fastener classes' counts were,
  in the 2026-08-31 entries above) — Step 4b of the run 5 notebook will
  report the real number before training starts; it is not guessed here.
