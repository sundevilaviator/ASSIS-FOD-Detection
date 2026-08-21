#!/usr/bin/env python3
# %% [markdown]
# # ASSIS — FOD Detection: Complete Training Pipeline (Phase 2)
#
# One file, run top to bottom, in Colab (paste into cells at the `# %%`
# markers) or as a plain script (`python notebooks/ASSIS_FOD_Colab_Full.py`).
# It calls the same tested modules in `src/` — this file does not
# reimplement their logic, so a fix in `src/` is picked up here automatically
# on the next `git pull`. What this file adds on top of `src/` is just
# orchestration: the order of operations, and the two manual-confirmation
# checkpoints marked CONFIRM BEFORE CONTINUING below.
#
# ## Why this exists / what it targets
# Built from a competitive review of 8 commercial and government-funded FOD
# detection systems (`docs/GAP_ANALYSIS_SUMMARY.md`). Every one of them is
# either a $1M-$8M+ fixed radar/EO installation or a newer vehicle/drone-
# mounted camera system — none reuse an airport's already-installed
# fixed/PTZ security CCTV, and none publish results in FAA AC 150/5220-24's
# own terms. This pipeline targets both gaps directly: small-object-focused
# training (`src/data_prep.py`'s oversampling) and an FAA-AC-formatted
# benchmark (`src/benchmark_faa.py`), including — new in this version — a
# breakdown by FOD-A's own light-level/weather metadata, which is not
# something any reviewed vendor or paper publishes either.
#
# This is a cost/deployability differentiator applicable at any airport
# running ASSIS's other camera-based modules (PPE, badge misuse, fall
# detection) on the same feed — NOT a small-airport-only tool. See
# `README.md` for the full positioning statement.
#
# ## Two things this file does NOT assert
# 1. FOD-A's raw class names and its light/weather CSV column names were not
#    independently confirmed against the live dataset when this file was
#    written (no network access to Kaggle/GitHub in that environment). Step
#    1b and Step 4 below are explicit manual-confirmation checkpoints for
#    exactly that reason — don't skip them.
# 2. CHS camera access has not been authorized. Nothing in this file touches
#    a live camera feed; it trains and evaluates against the FOD-A dataset
#    only. Camera integration is out of scope here.

# %%
# --------------------------------------------------------------------------
# Step 0 — Setup
# --------------------------------------------------------------------------
# If running in Colab from a fresh clone:
#   !git clone https://github.com/sundevilaviator/ASSIS-FOD-Detection.git
#   %cd ASSIS-FOD-Detection
#   !pip install -q -r requirements.txt
#
# Runtime -> Change runtime type -> GPU, before running the training step.

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd()
CONFIG = REPO / "configs" / "fod.yaml"


def run(cmd: list) -> None:
    """Run a src/ script as a subprocess and stream its output, so this file
    stays a thin orchestrator rather than a second copy of the logic."""
    print("\n$ " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


# %%
# --------------------------------------------------------------------------
# Step 1 — Kaggle credentials + download FOD-A
# --------------------------------------------------------------------------
import os  # noqa: E402

try:
    from google.colab import files  # type: ignore

    print("Colab detected — upload your kaggle.json (Kaggle account -> Settings -> Create New Token).")
    uploaded = files.upload()
    kaggle_dir = Path("~/.kaggle").expanduser()
    kaggle_dir.mkdir(exist_ok=True)
    for fname, content in uploaded.items():
        (kaggle_dir / fname).write_bytes(content)
    os.chmod(kaggle_dir / "kaggle.json", 0o600)
except ImportError:
    print("Not running in Colab. Set KAGGLE_USERNAME / KAGGLE_KEY env vars, "
          "or place ~/.kaggle/kaggle.json manually, before continuing.")

RAW_VOC_DIR = REPO / "data" / "fod-a-raw"
run([
    sys.executable, "src/data_prep.py", "--download",
    "--dataset", "kilogrand/foreign-object-debris-in-airports-fod-a-dataset",
    "--out", str(RAW_VOC_DIR),
])

# %%
# --------------------------------------------------------------------------
# Step 1b — CONFIRM BEFORE CONTINUING: inspect the real downloaded structure
# --------------------------------------------------------------------------
# Print the directory tree so YOU confirm where images vs. annotations live
# before the next steps guess at it. Common Pascal VOC convention is
# JPEGImages/ + Annotations/, but this was not independently verified
# against the live dataset when this script was written — check the printed
# tree against the values of IMAGES_SUBDIR / ANNOTATIONS_SUBDIR below and
# EDIT THEM if they don't match what you actually see.

for p in sorted(RAW_VOC_DIR.rglob("*"))[:60]:
    print(p.relative_to(RAW_VOC_DIR))

# EDIT THESE TWO if the printed tree above doesn't match:
IMAGES_SUBDIR = "JPEGImages"
ANNOTATIONS_SUBDIR = "Annotations"

RAW_IMAGES_DIR = RAW_VOC_DIR / IMAGES_SUBDIR
RAW_ANNOTATIONS_DIR = RAW_VOC_DIR / ANNOTATIONS_SUBDIR
assert RAW_IMAGES_DIR.exists(), f"{RAW_IMAGES_DIR} not found — fix IMAGES_SUBDIR above."
assert RAW_ANNOTATIONS_DIR.exists(), f"{RAW_ANNOTATIONS_DIR} not found — fix ANNOTATIONS_SUBDIR above."

# %%
# --------------------------------------------------------------------------
# Step 2 — Confirm the REAL class names before trusting configs/fod.yaml
# --------------------------------------------------------------------------
# Don't assume the class list in configs/fod.yaml matches the live dataset —
# check it against what's actually in the XML files first.

run([sys.executable, "src/voc_to_yolo.py", "--voc-dir", str(RAW_ANNOTATIONS_DIR),
     "--list-classes-only", "--config", str(CONFIG)])

print(
    "\nCompare the class names printed above against configs/fod.yaml's "
    "`classes:` list (currently Wrench, Hammer, Screwdriver, SodaCan, Wood — "
    "a deliberately narrow starting subset). If names differ (e.g. casing, "
    "spelling), edit configs/fod.yaml before continuing, not this script."
)

# %%
# --------------------------------------------------------------------------
# Step 3 — VOC -> YOLO conversion
# --------------------------------------------------------------------------
YOLO_LABELS_DIR = REPO / "data" / "fod-a-yolo-labels"
run([sys.executable, "src/voc_to_yolo.py",
     "--voc-dir", str(RAW_ANNOTATIONS_DIR), "--out", str(YOLO_LABELS_DIR),
     "--config", str(CONFIG)])

# Assemble the images/ + labels/ root src/data_prep.py expects.
import shutil  # noqa: E402

SOURCE_ROOT = REPO / "data" / "fod-a-yolo"
(SOURCE_ROOT / "images").mkdir(parents=True, exist_ok=True)
(SOURCE_ROOT / "labels").mkdir(parents=True, exist_ok=True)
for img in RAW_IMAGES_DIR.glob("*.jpg"):
    dest = SOURCE_ROOT / "images" / img.name
    if not dest.exists():
        shutil.copy2(img, dest)
for lbl in YOLO_LABELS_DIR.glob("*.txt"):
    shutil.copy2(lbl, SOURCE_ROOT / "labels" / lbl.name)
print(f"Assembled YOLO-format source at {SOURCE_ROOT}")

# %%
# --------------------------------------------------------------------------
# Step 4 — CONFIRM BEFORE CONTINUING: locate the light/weather metadata CSV
# --------------------------------------------------------------------------
# FOD-A ships light-level (bright/dim/dark) and weather (dry/wet)
# categorization separately from the bounding-box annotations (Munyer et al.
# 2021). This step is what makes the FAA benchmark's environmental
# breakdown possible — but the CSV's exact filename and column names were
# not independently verified against the live dataset when this script was
# written. Find it in the printed tree from Step 1b (likely a .csv near the
# annotations) and set the path below. If you can't find one, set
# METADATA_CSV = None and the benchmark step will simply skip that section
# rather than fail.

METADATA_CSV = None  # e.g. RAW_VOC_DIR / "light_weather_labels.csv" — CONFIRM the real filename first

# %%
# --------------------------------------------------------------------------
# Step 5 — Build the small-object-weighted split
# --------------------------------------------------------------------------
SPLIT_DIR = REPO / "data" / "fod-a-split"
run([
    sys.executable, "src/data_prep.py", "--build-split",
    "--source", str(SOURCE_ROOT), "--out", str(SPLIT_DIR),
    "--small-object-max-area-pct", "0.5", "--oversample-factor", "3",
    "--config", str(CONFIG),
])

# %%
# --------------------------------------------------------------------------
# Step 6 — Train
# --------------------------------------------------------------------------
# yolov8s / imgsz=960 / copy_paste=0.3 / hsv_v=0.5 come from configs/fod.yaml
# — applied from the start (not bolted on after a problem shows up), same as
# the Phase 1 lesson about underrepresented classes never recovering once
# training starts without enough of them in view.
run([
    sys.executable, "src/train.py",
    "--config", str(CONFIG), "--data", str(SPLIT_DIR / "data.yaml"),
])

weights_candidates = sorted((REPO / "runs" / "detect").glob("train*/weights/best.pt"))
assert weights_candidates, "No weights produced — check the training step's output above for errors."
WEIGHTS = weights_candidates[-1]
print(f"\nTrained weights: {WEIGHTS}")

# %%
# --------------------------------------------------------------------------
# Step 7 — FAA AC 150/5220-24-style benchmark (with environmental breakdown
# if METADATA_CSV was set in Step 4)
# --------------------------------------------------------------------------
benchmark_cmd = [
    sys.executable, "src/benchmark_faa.py",
    "--weights", str(WEIGHTS), "--data", str(SPLIT_DIR / "data.yaml"),
    "--config", str(CONFIG), "--out", str(REPO / "docs" / "benchmark_results"),
]
if METADATA_CSV is not None:
    benchmark_cmd += ["--metadata-csv", str(METADATA_CSV)]
run(benchmark_cmd)

print(
    "\nRead the LIMITATIONS section the benchmark script just printed before "
    "quoting any number from it — pixel-area size buckets, per-image false "
    "alarm rate, and pixel-based localization error are proxies, not the "
    "AC's real centimeter/meter/per-day units, until a calibrated camera "
    "setup closes that gap (see docs/FAA_AC_150_5220-24_BENCHMARK.md)."
)

# %%
# --------------------------------------------------------------------------
# Step 8 — Save artifacts and record the run
# --------------------------------------------------------------------------
# In Colab, copy weights + benchmark report to Drive so they survive the
# session ending:
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#   import shutil, datetime
#   stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
#   dest = f'/content/drive/MyDrive/ASSIS-FOD-runs/{stamp}'
#   import os; os.makedirs(dest, exist_ok=True)
#   shutil.copy(WEIGHTS, dest)
#   shutil.copytree(REPO / 'docs' / 'benchmark_results', f'{dest}/benchmark_results', dirs_exist_ok=True)
#
# Then, back in your local clone (not in Colab), add a dated entry to
# docs/RESEARCH_LOG.md with the real numbers from this run — done/pending/
# planned, same as every other entry in that file. That log entry plus these
# artifacts is what turns this run into citable, verifiable progress rather
# than an unlogged experiment.

print("Pipeline complete. Next: record this run in docs/RESEARCH_LOG.md with the real numbers above.")
