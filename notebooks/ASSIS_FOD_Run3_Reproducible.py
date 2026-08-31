#!/usr/bin/env python3
# %% [markdown]
# # ASSIS FOD — Run 3: reproducible split, enlarged small-object test set
#
# Why this run exists
# -------------------
# Runs 1 (2026-08-20) and 2 (2026-08-23) were built with splits made BEFORE
# the ordering fix, so neither is reproducible from `--seed 42` on another
# machine. Every figure published from them — 50.0%, 95% CI [40.0%, 60.0%],
# n = 92 — rests on splits nobody else can rebuild. This run fixes that.
#
# It also enlarges the small-object test set. At n = 46 per run the interval
# is roughly +/-14 points. Repeating at that size narrows nothing; holding out
# more small-bucket images does. `--small-test-frac 0.40` raises the small
# held-out set without touching the large/medium buckets, which are already
# near 99% and do not need more test data.
#
# Every cell re-derives its own paths. A Colab VM can be recycled between
# cells, and cells that depend on earlier variables fail with NameError when
# that happens — which is what went wrong before.

# %%
# ---- Step 1: environment ---------------------------------------------------
!pip -q install ultralytics kagglehub

import os, sys, json, subprocess
from pathlib import Path

REPO = Path("/content/ASSIS-FOD-Detection")
REPO_URL = "https://github.com/sundevilaviator/ASSIS-FOD-Detection.git"
# subprocess rather than a `!git` magic: a `!` line inside an if/else branch
# is valid only under IPython, and this file is also meant to run as a plain
# script. Same reason the conditional logic lives here and not in a magic.
if not REPO.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "pull"], check=False)

sys.path.insert(0, str(REPO))
print("repo:", REPO, "| exists:", REPO.exists())

# %%
# ---- Step 2: Drive checkpointing, BEFORE any training ----------------------
# Set up first. Five disconnects across the previous two runs; a checkpoint
# written only to the VM's local disk is gone when the VM is recycled.
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive", force_remount=False)
DRIVE_RUNS = Path("/content/drive/MyDrive/ASSIS_FOD_runs")
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

LOCAL_RUNS = Path("/content/runs")
if LOCAL_RUNS.is_symlink() or LOCAL_RUNS.exists():
    print("runs/ already set up ->", os.path.realpath(LOCAL_RUNS))
else:
    LOCAL_RUNS.symlink_to(DRIVE_RUNS)
    print("symlinked /content/runs ->", DRIVE_RUNS)

# %%
# ---- Step 3: dataset ------------------------------------------------------
from pathlib import Path
import kagglehub, shutil

RAW = Path("/content/fod-a-raw")
if not RAW.exists() or not any(RAW.iterdir()):
    cached = Path(kagglehub.dataset_download(
        "kilogrand/foreign-object-debris-in-airports-fod-a-dataset"))
    shutil.copytree(cached, RAW, dirs_exist_ok=True)
print("raw:", RAW)

# %%
# ---- Step 4: locate the VOC layout (search, don't assume) ------------------
import sys
from pathlib import Path
sys.path.insert(0, "/content/ASSIS-FOD-Detection")
from src.colab_helpers import find_voc_root

ANNOT, IMAGES = find_voc_root(Path("/content/fod-a-raw"))
print("annotations:", ANNOT)
print("images     :", IMAGES)
print("xml files  :", sum(1 for _ in ANNOT.glob('*.xml')))

# %%
# ---- Step 5: VOC -> YOLO ---------------------------------------------------
# CONFIRM BEFORE CONTINUING: check the printed class list against
# configs/fod.yaml. 31 classes expected.
import sys
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")
sys.path.insert(0, str(REPO))
from src.colab_helpers import find_voc_root

ANNOT, IMAGES = find_voc_root(Path("/content/fod-a-raw"))
!python {REPO}/src/voc_to_yolo.py --voc-dir {ANNOT} --list-classes-only --config {REPO}/configs/fod.yaml

# %%
import sys
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")
sys.path.insert(0, str(REPO))
from src.colab_helpers import find_voc_root

ANNOT, IMAGES = find_voc_root(Path("/content/fod-a-raw"))
YOLO_ROOT = Path("/content/fod-a-yolo")
(YOLO_ROOT / "images").mkdir(parents=True, exist_ok=True)
(YOLO_ROOT / "labels").mkdir(parents=True, exist_ok=True)

!python {REPO}/src/voc_to_yolo.py --voc-dir {ANNOT} --out {YOLO_ROOT}/labels --config {REPO}/configs/fod.yaml

# shutil, not `cp *.jpg`: a 34k-file glob overflows the shell argument list,
# and the old `2>/dev/null` hid that failure until Step 6 reported an empty
# dataset three cells later. Never suppress stderr on a step whose failure is
# invisible downstream.
import shutil
n = 0
for _p in IMAGES.iterdir():
    if _p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        shutil.copy2(_p, YOLO_ROOT / 'images' / _p.name)
        n += 1
print('copied', n, 'images')

print("images:", len(list((YOLO_ROOT / 'images').glob('*.jpg'))))
print("labels:", len(list((YOLO_ROOT / 'labels').glob('*.txt'))))

# %%
# ---- Step 6: build the split (corrected ordering + enlarged small bucket) --
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")
SPLIT = Path("/content/fod-a-split-run3")

!python {REPO}/src/data_prep.py --build-split \
    --source /content/fod-a-yolo \
    --out {SPLIT} \
    --small-object-max-area-pct 0.5 \
    --test-frac 0.15 \
    --small-test-frac 0.40 \
    --seed 42 \
    --config {REPO}/configs/fod.yaml

# %%
# ---- Step 7: RECORD THE FINGERPRINTS --------------------------------------
# This is the point of the run. Paste these into docs/RESEARCH_LOG.md.
# Anyone rebuilding from the same source with --seed 42 must get the same
# digests. Aggregate counts matching is NOT sufficient — that is precisely
# what agreed across runs 1 and 2 while the membership differed.
import json
from pathlib import Path

m = json.loads((Path("/content/fod-a-split-run3") / "split_manifest.json").read_text())
print("small held out :", m["test_bucket_counts"]["small"],
      " (runs 1-2 had 46; target ~96 for +/-10 points)")
print("test total     :", m["n_test"])
print("seed           :", m["seed"],
      "| test_frac:", m["test_frac"], "| small_test_frac:", m["small_test_frac"])
print("\nFINGERPRINTS — record these:")
for k, v in m["fingerprints"].items():
    print(f"  {k:14s} {v}")

# %%
# ---- Step 8: train ---------------------------------------------------------
# Resumes from the most recent checkpoint BY MODIFICATION TIME, not by name.
# Name ordering is what silently resumed a finished `train` over a partial
# `train-2` and exited at 100/100 without training.
import sys
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")
sys.path.insert(0, str(REPO))
from src.colab_helpers import describe_runs, find_latest_run
from ultralytics import YOLO

RUNS_DETECT = Path("/content/runs/detect")
print(describe_runs(RUNS_DETECT))

latest = find_latest_run(RUNS_DETECT)
if latest is not None:
    print(f"\nRESUMING from {latest}")
    model = YOLO(str(latest / "weights" / "last.pt"))
    results = model.train(resume=True)
else:
    print("\nSTARTING a new run")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(Path("/content/fod-a-split-run3") / "data.yaml"),
        epochs=100, imgsz=640, batch=16, seed=42,
        project="/content/runs/detect", name="run3", exist_ok=False,
    )

# %%
# ---- Step 9: FAA benchmark -------------------------------------------------
# Small-object detection rate here is the figure that bounds every claim made
# about this model. With ~96 small instances the 95% interval is roughly
# +/-10 points at p ~ 0.5, versus +/-14 at n = 46. It will still be an
# interval, not a point estimate — quote it with the interval attached.
import sys
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")
sys.path.insert(0, str(REPO))
from src.colab_helpers import find_latest_run

latest = find_latest_run(Path("/content/runs/detect"))
assert latest is not None, "no completed run found"
weights = latest / "weights" / "best.pt"
print("benchmarking:", weights)

!python {REPO}/src/benchmark_faa.py \
    --weights {weights} \
    --data /content/fod-a-split-run3/data.yaml \
    --config {REPO}/configs/fod.yaml \
    --out {REPO}/docs/benchmark_results

# %%
# ---- Step 10: light/weather stratification -------------------------------
# The Kaggle mirror is the Pascal VOC distribution: bounding boxes only. The
# FOD-A paper states the categorization annotations ship with the dataset's
# ORIGINAL format (8.3 GB, 400x400), not the Pascal VOC one (412 MB, 300x300)
# — which is why the earlier search came back empty. Both are linked from
# github.com/FOD-UNOmaha/FOD-data and hosted on Google Drive.
#
# This cell downloads the original format, SEARCHES it for the categorization
# CSV (no filename or column layout is assumed), and validates the contents
# against the counts published in the paper's Table I before anything
# downstream uses it.
#
# Paper Table I:  Dry 26,647 / Wet 7,216 ; Dark 4,387 / Dim 12,464 / Bright
# 17,012 — both summing to 33,863. The VOC mirror in hand has 33,793
# annotation files, so a small revision difference is expected and will be
# reported rather than reconciled away.
!pip -q install gdown

import sys, subprocess
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")
sys.path.insert(0, str(REPO))

ORIG = Path("/content/fod-a-original")
ORIG.mkdir(parents=True, exist_ok=True)
FILE_ID = "1lLBJXXaQCWaFa-1MeLAANPpSwMhCJqGh"   # FOD-A v2.1 original format

archive = ORIG / "fod-a-original.zip"
if not archive.exists() and not any(ORIG.glob("*/")):
    subprocess.run(["gdown", "--id", FILE_ID, "-O", str(archive)], check=False)
    if archive.exists():
        subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(ORIG)], check=False)

print("downloaded to:", ORIG)
print("top level:", sorted(p.name for p in ORIG.iterdir())[:20])

# %%
# ---- Step 10b: find and validate the categorization CSV -------------------
import sys
from pathlib import Path
sys.path.insert(0, "/content/ASSIS-FOD-Detection")
from src.fod_metadata import (
    describe_scan, read_categorization, scan_for_metadata, validate_against_paper,
)

files = scan_for_metadata(Path("/content/fod-a-original"))
print("CSV files found:")
print(describe_scan(files))

cat = [f for f in files if f.is_categorization]
if not cat:
    print("\nNo categorization CSV found in the original-format download.")
    print("Do NOT run a stratified benchmark. Record in docs/RESEARCH_LOG.md "
          "that the metadata could not be located, and leave the "
          "environmental-stratification gap open rather than reporting an "
          "unstratified result as if it were stratified.")
else:
    meta = cat[0]
    print(f"\nUsing: {meta.path}")
    print(f"  weather column: {meta.weather_column}")
    print(f"  light column  : {meta.light_column}")
    print(f"  path column   : {meta.path_column}")
    weather, light = read_categorization(meta)
    report = validate_against_paper(weather, light)
    print("\n" + str(report))
    print("\nRecord the OBSERVED counts above in docs/RESEARCH_LOG.md — cite "
          "those, not the paper's, if they differ.")

# %%
# ---- Step 10c: stratified benchmark ---------------------------------------
# Only run this once Step 10b printed a categorization file and a report you
# are willing to stand behind.
import sys
from pathlib import Path
sys.path.insert(0, "/content/ASSIS-FOD-Detection")
from src.colab_helpers import find_latest_run
from src.fod_metadata import scan_for_metadata

REPO = Path("/content/ASSIS-FOD-Detection")
cat = [f for f in scan_for_metadata(Path("/content/fod-a-original")) if f.is_categorization]
assert cat, "no categorization CSV — see Step 10b before running this"
latest = find_latest_run(Path("/content/runs/detect"))
assert latest is not None, "no completed run"

!python {REPO}/src/benchmark_faa.py \
    --weights {latest}/weights/best.pt \
    --data /content/fod-a-split-run3/data.yaml \
    --config {REPO}/configs/fod.yaml \
    --metadata-csv {cat[0].path} \
    --out {REPO}/docs/benchmark_results

# %%
# ---- Step 11: save what matters back to Drive ------------------------------
import shutil
from pathlib import Path

DEST = Path("/content/drive/MyDrive/ASSIS_FOD_run3_artifacts")
DEST.mkdir(parents=True, exist_ok=True)
shutil.copy2("/content/fod-a-split-run3/split_manifest.json", DEST)
for p in Path("/content/ASSIS-FOD-Detection/docs/benchmark_results").glob("*"):
    shutil.copy2(p, DEST)
print("saved to", DEST)
print(sorted(p.name for p in DEST.iterdir()))
