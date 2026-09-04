# ASSIS — Foreign Object Debris (FOD) Detection Module

Phase 2 of the AI-Integrated Airport Safety and Security Intelligence System (ASSIS)

This repository contains the Phase 2 FOD detection module of ASSIS. Phase 1 (PPE compliance detection) is at `sundevilaviator/ASSIS-PPE-Detection`. This module targets a specific, documented gap in publicly benchmarked FOD detection rather than reproducing existing capability.

**Live demo:** [fodscan.streamlit.app](https://fodscan.streamlit.app/)

## Why this scope

This module is scoped around a documented gap in publicly benchmarked FOD detection: evaluating performance in the terms [FAA AC 150/5220-24](https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_150_5220-24.pdf) actually uses (object size class, false-alarm rate per 90-day average, location accuracy), with particular attention to small-object detection, where FOD is hardest to catch and least commonly benchmarked. `src/benchmark_faa.py` reports exactly those metrics against model output, so performance claims here are checkable, not asserted.

* **Small-object focus.** The most widely used public benchmark, [FOD-A](https://www.kaggle.com/datasets/kilogrand/foreign-object-debris-in-airports-fod-a-dataset), is dominated by large, easily-detected debris. Real airfield FOD is predominantly small items (nuts, bolts, screws, metal fragments) under 4 cm — the size class where detection models measurably degrade. This module explicitly builds and evaluates a small-object-weighted training split.
* **Existing-camera-native design.** The training/inference pipeline targets standard fixed or PTZ camera feeds rather than purpose-built sensor hardware, consistent with the design principle ASSIS uses across its other modules (PPE, badge misuse, fall detection).
* **Human-in-the-loop.** Consistent with the rest of ASSIS, this module classifies and localizes candidate FOD for operator review; it does not trigger any automated physical response.

## Status

This is an active research module, not a certified or deployed product. See `docs/RESEARCH_LOG.md` for the current, dated state. As of this commit:

* Repository scaffold, VOC→YOLO conversion, data pipeline, training/inference/benchmark scripts, and demo app
* Every script's logic unit-tested against hand-computed values; the full pipeline (conversion → split → train → infer → benchmark, including the environmental-metadata breakdown) has been run end to end against synthetic data and confirmed to execute without error
* **Split reproducibility defect found and fixed (2026-08-23)** — `find_labeled_images()` fed filesystem-ordered enumeration into a seeded shuffle, so `--seed 42` produced different splits on different machines. Found by comparing per-class test composition across two early runs; invisible in the split manifest, whose aggregate counts matched. Fixed by sorting before the shuffle, with five regression tests (`tests/test_split_reproducibility.py`). Splits built before that commit are not reproducible across machines and are superseded by the run below.
* **Model trained and evaluated on the real FOD-A small-object split — run 3** (Colab A100, post-fix, reproducible split with recorded SHA-256 fingerprints). Results reported in FAA AC 150/5220-24 terms against real data — see `docs/benchmark_results/`. Detection by size bucket: large 99.6% (523/525), medium 99.1% (220/222), small 52.0% (64/123), 95% CI [43.3%, 60.7%]. The small-object result sits entirely below the Advisory Circular's referenced 90% threshold; that shortfall is the finding, and it is reported rather than tuned away.
* **Two negative results, both logged rather than discarded:** an inference-resolution sweep and a SAHI sliced-inference experiment (`--sahi` flag in `src/benchmark_faa.py`) — neither improved small-object detection. See `docs/RESEARCH_LOG.md` for full reasoning.
* Several areas remain open (cross-site validation, additional environmental robustness, expanded object coverage) — tracked internally rather than itemized here.
* **Camera/CCTV integration — not started, and not authorized.** This module trains and evaluates against the FOD-A dataset only. No camera survey, testing, or deployment work has been done or is assumed as a next step.

A known limitation, stated plainly: FOD-A images do not carry real-world physical scale (no calibrated camera geometry), so "object size" here is a bounding-box-area proxy, not a measured centimeter size. The benchmark script documents this explicitly rather than reporting FAA-standard centimeter thresholds as if they were directly verified — closing that gap requires either a calibrated camera setup or a dataset with known object-to-pixel scale, which is flagged as follow-on work.

A second known limitation: FOD-A's exact internal folder layout and its light/weather metadata CSV's column names were not independently verified against the live dataset while this repository was built (no network access to Kaggle in that environment). `notebooks/ASSIS_FOD_Run3_Reproducible.py` has explicit "CONFIRM BEFORE CONTINUING" checkpoints for exactly this reason — don't skip them on a first run.

A third known limitation: some real-world FOD categories (e.g. tire-fragment / rubber debris) are not covered by FOD-A's object classes, and this module does not currently address that gap.

## Quickstart

Option A — the complete pipeline file (recommended)
Open `notebooks/ASSIS_FOD_Run3_Reproducible.py` in Colab (paste each `# %%` block into a cell) or run it as a plain script once dependencies are installed. Stop at both "CONFIRM BEFORE CONTINUING" checkpoints and check the printed dataset structure / class names / metadata filename against what the script assumes before letting it continue. This is the run of record — its weights and benchmark are what `docs/benchmark_results/` and the figures above are drawn from.

`notebooks/ASSIS_FOD_Run4_Original400.py` retrains from FOD-A's original-format (400x400) distribution instead of the Pascal VOC mirror, so the light/weather categorization CSV can be joined by filename within one archive instead of across two. **Executed 2026-09-03** — small/medium/large detection rates match run 3 within noise (52.0%/99.5%/99.8%), and the light/weather stratification the earlier runs couldn't produce is now real: bright 87.5%, dim 100%, dark 100%, dry 91.3%, wet 100%. See `docs/RESEARCH_LOG.md` for the full result, an open question about the bright-vs-dark gap that isn't resolved yet, and a metadata-matching bug found and fixed along the way.

```
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
# 6. Benchmark against FAA AC 150/5220-24 criteria (add --metadata-csv for the environmental breakdown, --sahi for sliced inference)
python src/benchmark_faa.py --weights runs/detect/train/weights/best.pt --data data/fod-a-split/data.yaml --out docs/benchmark_results
# 7. Run the demo app
streamlit run app/streamlit_app.py -- --weights runs/detect/train/weights/best.pt
```

## Deploying the demo app (hosted)

Trained weights are deliberately not committed to git (`.gitignore` excludes `*.pt` — a 20MB+ binary does not belong in source history), so a hosted deployment has no weights file to load and will show an empty state until one is provided. To deploy:

1. Publish `best.pt` as a GitHub release asset on this repository (Releases → Draft a new release → attach the file). This keeps the binary out of git history while still giving it a stable public URL.
2. In Streamlit Community Cloud, open the app's Settings → Secrets and add:

```
weights_url = "https://github.com/<owner>/<repo>/releases/download/<tag>/best.pt"
```

(Equivalently, set an `ASSIS_FOD_WEIGHTS_URL` environment variable.)

The app downloads the weights once on first run and caches them. A local `--weights` path, when supplied, always takes precedence over the URL.

`packages.txt` pins the system libraries OpenCV — pulled in by `ultralytics` — needs at import time: `libgl1` and `libglib2.0-0t64`. Streamlit Community Cloud's current base image does not include either, and without this file model loading fails with `libGL.so.1: cannot open shared object file`. Note: `libglib2.0-0t64`, not `libglib2.0-0` — the platform's base image renamed this package (a Debian time_t transition), and the old name pulls an uninstallable `libffi7` on the current image, breaking the whole dependency install. This is confirmed working against the live deployment above.

## Relationship to the ASSIS platform

This module produces the same structured output format described in the ASSIS Technical Report §5.1 (time, location/camera ID, classification, confidence) so it can plug into the same reporting/SMS-integration layer as the PPE, badge-misuse, and fall-detection modules, rather than existing as a standalone tool.

## License

AGPL-3.0 — see `LICENSE`, matching the license used for Phase 1 (`ASSIS-PPE-Detection`). This means anyone who runs a modified version of this code as a network service must make that modified source available to users of the service, not just to people they distribute the software to directly. The FOD-A dataset has its own license/attribution terms; see the [dataset page](https://www.kaggle.com/datasets/kilogrand/foreign-object-debris-in-airports-fod-a-dataset) before redistributing any derived data.
