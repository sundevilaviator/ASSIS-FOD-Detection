# ASSIS — Foreign Object Debris (FOD) Detection Module

**Phase 2 of the AI-Integrated Airport Safety and Security Intelligence System (ASSIS)**

This repository contains the Phase 2 FOD detection module of ASSIS. Phase 1 (PPE compliance detection) is at [`sundevilaviator/ASSIS-PPE-Detection`](https://github.com/sundevilaviator/ASSIS-PPE-Detection). This module targets a specific, documented gap in the FOD detection literature and commercial market rather than reproducing existing capability — see [`docs/GAP_ANALYSIS_SUMMARY.md`](docs/GAP_ANALYSIS_SUMMARY.md) for the full competitive landscape this scope was derived from.

## Why this scope

Commercial FOD detection (QinetiQ Tarsier, Xsight FODetect, Stratech iFerret) is fixed radar/EO infrastructure costing $1M–$8M+, built for large hubs. A newer wave of vehicle-mounted camera+AI systems — notably Illuminex AI's FOD AI, delivered on its InspectEx platform alongside perimeter-intrusion, snowbank, edge-light and (announced) pavement-condition modules — lowers cost by mounting on vehicles an airport already operates, and does amortize development across several airfield applications. It still requires dedicated sensor hardware, and it covers the airfield in periodic inspection passes rather than continuously. **No reviewed system performs FOD scanning using an airport's already-installed security CCTV/PTZ cameras** — which is the design principle ASSIS uses for its other modules (PPE, badge misuse, fall detection). This module applies that same principle to FOD:

- **Small-object focus.** The most widely used public benchmark, [FOD-A](https://www.kaggle.com/datasets/kilogrand/foreign-object-debris-in-airports-fod-a-dataset), is dominated by large, easily-detected debris. Real airfield FOD is predominantly small items (nuts, bolts, screws, metal fragments) under 4 cm — the size class where detection models measurably degrade. This module explicitly builds and evaluates a small-object-weighted training split.
- **FAA-standard benchmarking.** Every FOD detection claim is ultimately measured against [FAA AC 150/5220-24](https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_150_5220-24.pdf). None of the commercial systems reviewed publish results in the AC's own terms (object size class, false-alarm rate per 90-day average, location accuracy). This repo includes a benchmark script (`src/benchmark_faa.py`) that reports exactly those metrics against model output, so performance claims are checkable, not asserted.
- **No new hardware.** The training/inference pipeline is designed for existing fixed or PTZ camera feeds — not a new radar tower or a dedicated inspection vehicle. This is a cost/deployability differentiator, not a size ceiling: it lowers the marginal cost of adding FOD coverage at *any* airport that already has ASSIS's other modules (PPE, badge misuse, fall detection) running on the same camera infrastructure — the incremental cost of one more detection head on an existing feed, versus a new capital purchase. Stated precisely: multi-function amortization on shared infrastructure is **not** unique to ASSIS — Illuminex's InspectEx does this too, across its own sensor hardware. The narrower distinction that does hold is *which* infrastructure is shared: already-installed fixed cameras giving continuous coverage, versus purchased sensors giving inspection-pass coverage. See `docs/GAP_ANALYSIS_SUMMARY.md`, where the broader claim is struck through and withdrawn.
- **Human-in-the-loop.** Consistent with the rest of ASSIS, this module classifies and localizes candidate FOD for operator review; it does not trigger any automated physical response.

## What's in this repository

| Path | Purpose |
|---|---|
| `notebooks/ASSIS_FOD_Colab_Full.py` | **The complete pipeline, one file.** Cell-delimited (`# %%`) — paste into Colab or run as a plain script. Download → confirm real class names → VOC→YOLO conversion → small-object split → train → FAA benchmark (with environmental breakdown) → save. Has two explicit manual-confirmation checkpoints (see below) rather than guessing dataset internals. |
| `notebooks/ASSIS_FOD_Training_Colab.ipynb` | Same pipeline as a Colab-native notebook, for anyone who prefers `.ipynb` over a cell-delimited `.py`. |
| `src/voc_to_yolo.py` | Converts FOD-A's native Pascal VOC XML annotations to YOLO format. Includes `--list-classes-only` to read the dataset's real class names directly rather than assuming them. |
| `src/data_prep.py` | Downloads FOD-A, and re-splits a YOLO-format dataset by bounding-box size, producing a small-object-weighted training set and a held-out size-stratified test set. |
| `src/train.py` | CLI wrapper around Ultralytics YOLOv8 training, parameterized by `configs/fod.yaml`. |
| `src/infer.py` | CLI batch inference over a folder of images or a video; writes annotated output and a JSON detection log. |
| `src/benchmark_faa.py` | Scores a model's predictions against ground truth using FAA AC 150/5220-24 criteria: detection rate by object-size bucket, false-alarm rate (per-image proxy), localization error, and — if a FOD-A light/weather metadata CSV is supplied — a breakdown by lighting and weather condition. Produces `docs/benchmark_results/*.md`. |
| `app/streamlit_app.py` | Local demo app: upload an image, run the trained model, view detections with FAA size-class labels and a mock operations "alert" panel (structured record: time, camera ID, size class, confidence). |
| `configs/fod.yaml` | Class list, size-bucket thresholds, augmentation, and training hyperparameters in one place. |
| `docs/RESEARCH_LOG.md` | Dated log of what was run, on what data, with what result — kept in the format used across ASSIS project documentation to keep "done / in progress / planned" distinct. |
| `docs/FAA_AC_150_5220-24_BENCHMARK.md` | Explains the benchmark methodology and its known limitations (see below). |
| `docs/GAP_ANALYSIS_SUMMARY.md` | Condensed version of the competitive/gap analysis that scoped this module — verification status of its vendor claims is noted at the top of that file. |

## Status

This is an active research module, not a certified or deployed product. See `docs/RESEARCH_LOG.md` for the current, dated state. As of this commit:

- [x] Repository scaffold, VOC→YOLO conversion, data pipeline, training/inference/benchmark scripts, and demo app
- [x] Every script's logic unit-tested against hand-computed values; the full pipeline (conversion → split → train → infer → benchmark, including the environmental-metadata breakdown) has been run end to end against synthetic data and confirmed to execute without error
- [x] Model trained and evaluated on the **real** FOD-A small-object split — two independent 100-epoch YOLOv8 runs (Colab A100, 2026-08-20 and 2026-08-23), each benchmarked on its own held-out split
- [x] Results reported in FAA AC 150/5220-24 terms against real data — see `docs/benchmark_results/`. Detection by size bucket: large 99.8–100%, medium 98.6–99.5%, **small 50.0% pooled across both runs, 95% CI [40.0%, 60.0%], n = 92**. The small-object result sits entirely below the Advisory Circular's referenced 90% threshold; that shortfall is the finding, and it is reported rather than tuned away. Individual-run figures (52.2% and 47.8%) differ by 0.42 standard errors and should not be quoted separately — at 46 small-object instances per run, a single run's interval is roughly ±14 points, so any specific decimal implies a precision this sample does not support
- [x] Split reproducibility defect found and fixed (2026-08-23) — `find_labeled_images()` fed filesystem-ordered enumeration into a seeded shuffle, so `--seed 42` produced different splits on different machines. Found by comparing per-class test composition across the two runs; invisible in the split manifest, whose aggregate counts matched. Fixed by sorting before the shuffle, with five regression tests (`tests/test_split_reproducibility.py`). **Splits built before that commit — including both runs above — are not reproducible across machines.** The two runs are therefore two measurements on different held-out splits, not a repeat of one measurement
- [ ] Multi-airport / cross-site validation
- [ ] Thermal/RGB fusion, open-world generalization (roadmap, not in scope for this module's first release)
- [ ] Camera/CCTV integration — **not started, and not authorized.** This module trains and evaluates against the FOD-A dataset only. No camera survey, testing, or deployment work has been done or is assumed as a next step.
- [ ] Tire-fragment / rubber-debris detection — **not started.** Not covered by FOD-A's 31 classes; no public labeled dataset for this identified yet. See "known limitations" below.

**A known limitation, stated plainly:** FOD-A images do not carry real-world physical scale (no calibrated camera geometry), so "object size" here is a bounding-box-area proxy, not a measured centimeter size. The benchmark script documents this explicitly rather than reporting FAA-standard centimeter thresholds as if they were directly verified — closing that gap requires either a calibrated camera setup or a dataset with known object-to-pixel scale, which is flagged as follow-on work.

**A second known limitation:** FOD-A's exact internal folder layout and its light/weather metadata CSV's column names were not independently verified against the live dataset while this repository was built (no network access to Kaggle in that environment). `notebooks/ASSIS_FOD_Colab_Full.py` has two explicit "CONFIRM BEFORE CONTINUING" checkpoints for exactly this reason — don't skip them on a first run.

**A third known limitation: tire-fragment / rubber-debris FOD is not covered.** FOD-A's 31 object classes (confirmed directly against the live dataset: Bolt, Pliers, Wrench, Washer, Wire, PlasticPart, LuggageTag, Cutter, Label, Nut, Nail, Battery, BoltWasher, MetalPart, PaintChip, SodaCan, ClampPart, Screwdriver, Hammer, LuggagePart, Rock, FuelCap, AdjustableClamp, BoltNutSet, Pen, AdjustableWrench, MetalSheet, Hose, Wood, Screw, Tape) do not include burst tire fragments or rubber debris left on taxiways/runways — a real, documented airfield FOD hazard (engine-ingestion and secondary tire-damage risk). This is a gap in the field's own standard public benchmark dataset, not something this module currently solves. Closing it would require a separately sourced and hand-annotated tire-fragment image set (no public dataset for this was identified), which has not been started. Scoped as a future phase; see `docs/GAP_ANALYSIS_SUMMARY.md`.

## Quickstart

### Option A — the complete pipeline file (recommended)
Open `notebooks/ASSIS_FOD_Colab_Full.py` in Colab (paste each `# %%` block into a cell) or run it as a plain script once dependencies are installed. Stop at both "CONFIRM BEFORE CONTINUING" checkpoints and check the printed dataset structure / class names / metadata filename against what the script assumes before letting it continue.

### Option B — step by step, locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Get the raw dataset (requires a Kaggle account + API token: ~/.kaggle/kaggle.json).
#    FOD-A ships Pascal VOC XML annotations, not YOLO — this downloads the raw dataset only.
python src/data_prep.py --download --dataset kilogrand/foreign-object-debris-in-airports-fod-a-dataset --out data/fod-a-raw

# 2. CONFIRM real class names before trusting configs/fod.yaml's `classes:` list
python src/voc_to_yolo.py --voc-dir data/fod-a-raw/Annotations --list-classes-only
#    (adjust the --voc-dir path once you've confirmed the real folder name from step 1's output)

# 3. Convert VOC -> YOLO labels, then assemble an images/+labels/ root
python src/voc_to_yolo.py --voc-dir data/fod-a-raw/Annotations --out data/fod-a-yolo-labels
mkdir -p data/fod-a-yolo/images data/fod-a-yolo/labels
cp data/fod-a-raw/JPEGImages/*.jpg data/fod-a-yolo/images/
cp data/fod-a-yolo-labels/*.txt data/fod-a-yolo/labels/

# 4. Build the small-object-weighted split
python src/data_prep.py --build-split --source data/fod-a-yolo --out data/fod-a-split --small-object-max-area-pct 0.5

# 5. Train
python src/train.py --config configs/fod.yaml --data data/fod-a-split/data.yaml

# 6. Benchmark against FAA AC 150/5220-24 criteria (add --metadata-csv for the environmental breakdown)
python src/benchmark_faa.py --weights runs/detect/train/weights/best.pt --data data/fod-a-split/data.yaml --out docs/benchmark_results

# 7. Run the demo app
streamlit run app/streamlit_app.py -- --weights runs/detect/train/weights/best.pt
```

### Deploying the demo app (hosted)

Trained weights are deliberately **not** committed to git (`.gitignore` excludes
`*.pt` — a 20MB+ binary does not belong in source history), so a hosted
deployment has no weights file to load and will show an empty state until one
is provided. To deploy:

1. Publish `best.pt` as a **GitHub release asset** on this repository
   (Releases → Draft a new release → attach the file). This keeps the binary
   out of git history while still giving it a stable public URL.
2. In Streamlit Community Cloud, open the app's **Settings → Secrets** and add:
   ```toml
   weights_url = "https://github.com/<owner>/<repo>/releases/download/<tag>/best.pt"
   ```
   (Equivalently, set an `ASSIS_FOD_WEIGHTS_URL` environment variable.)

The app downloads the weights once on first run and caches them. A local
`--weights` path, when supplied, always takes precedence over the URL.

`packages.txt` pins the one system library (`libgl1`) that OpenCV — pulled in
by `ultralytics` — needs at import time. Streamlit Community Cloud's base
image does not include it, and without this file model loading fails with
`libGL.so.1: cannot open shared object file`. Do **not** also list
`libglib2.0-0`: on that platform's Debian 11 image it pulls an uninstallable
`libffi7` and breaks the whole dependency install.

## Relationship to the ASSIS platform

This module produces the same structured output format described in the ASSIS Technical Report §5.1 (time, location/camera ID, classification, confidence) so it can plug into the same reporting/SMS-integration layer as the PPE, badge-misuse, and fall-detection modules, rather than existing as a standalone tool.

## License

AGPL-3.0 — see `LICENSE`, matching the license used for Phase 1
([`ASSIS-PPE-Detection`](https://github.com/sundevilaviator/ASSIS-PPE-Detection)).
This means anyone who runs a modified version of this code as a network
service must make that modified source available to users of the service,
not just to people they distribute the software to directly. The FOD-A
dataset has its own license/attribution terms; see the
[dataset page](https://www.kaggle.com/datasets/kilogrand/foreign-object-debris-in-airports-fod-a-dataset)
before redistributing any derived data.
