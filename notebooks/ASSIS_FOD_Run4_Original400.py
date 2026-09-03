#!/usr/bin/env python3
# %% [markdown]
# # ASSIS FOD — Run 4: train from the ORIGINAL 400x400 distribution
#
# Why this run exists
# -------------------
# Two independent findings on 2026-08-31 both point at the same cause:
#
#   1. The light/weather categorization CSV cannot be joined to the Pascal VOC
#      mirror. That mirror dropped 70 images and renumbered the rest
#      contiguously (000000-033792, zero gaps), so the correspondence to the
#      original ordering is unrecoverable from filenames.
#   2. Inference resolution does nothing for small objects on that mirror -
#      52.0% at imgsz 640, 960 AND 1280, the same 64 detections each time -
#      because the mirror is 300x300 and is already being upscaled to fit 640.
#
# The original-format distribution (FullDatasetV.2.1-400x400) fixes both: it is
# 400x400, and its filenames are exactly the File column of
# FOD_categorization_annotations.csv.
#
# WHAT THIS RUN IS AND IS NOT COMPARABLE TO
# -----------------------------------------
# Run 4 trains on DIFFERENT source imagery from runs 1-3. Its numbers are a new
# baseline, not a continuation of the 52.0% [43.3%, 60.7%] series. Do not
# present a run 4 figure as an improvement on run 3 - they are measured on
# different data. Say so wherever the number appears.
#
# Every cell re-derives its own paths, for the same reason as run 3.

# %%
# ---- Step 1: environment and repo -------------------------------------------
!pip -q install ultralytics

import os, sys, subprocess
from pathlib import Path

REPO = Path("/content/ASSIS-FOD-Detection")
REPO_URL = "https://github.com/sundevilaviator/ASSIS-FOD-Detection.git"
if not REPO.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
else:
    subprocess.run(["git", "-C", str(REPO), "pull"], check=False)
sys.path.insert(0, str(REPO))
print("repo ready:", REPO.exists())

# %%
# ---- Step 2: Drive checkpointing, BEFORE training ---------------------------
import os
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive", force_remount=False)
DRIVE_RUNS = Path("/content/drive/MyDrive/ASSIS_FOD_runs")
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)
LOCAL_RUNS = Path("/content/runs")
if not (LOCAL_RUNS.is_symlink() or LOCAL_RUNS.exists()):
    LOCAL_RUNS.symlink_to(DRIVE_RUNS)
print("runs ->", os.path.realpath(LOCAL_RUNS))

# %%
# ---- Step 3: confirm the original distribution is present -------------------
# It is 8.9 GB. If the VM was recycled since it was downloaded, re-run run 3's
# Step 10 to fetch it again before continuing.
from pathlib import Path

ORIG = Path("/content/fod-a-original/FullDatasetV.2.1-400x400")
assert ORIG.exists(), (
    "Original-format distribution not found. Re-run run 3 Step 10 to download "
    "it (gdown file id 1lLBJXXaQCWaFa-1MeLAANPpSwMhCJqGh), then return here."
)
print("top level:", sorted(p.name for p in ORIG.iterdir())[:20])

# %%
# ---- Step 4: MAP THE LAYOUT. Confirm before continuing. ---------------------
# The original distribution is organised per object (e.g. Battery1/frame/...),
# NOT as a flat Pascal VOC pair. Its annotation format must be established by
# looking, not assumed - the same discipline that found the VOC mirror's
# renumbering. Read this output before running anything after it.
from collections import Counter
from pathlib import Path

ORIG = Path("/content/fod-a-original/FullDatasetV.2.1-400x400")
exts = Counter()
sample_by_ext = {}
for p in ORIG.rglob("*"):
    if p.is_file():
        e = p.suffix.lower()
        exts[e] += 1
        sample_by_ext.setdefault(e, p)

print("file types under the distribution:")
for e, n in exts.most_common(15):
    print(f"  {e or '(none)':10s} {n:8d}   e.g. {sample_by_ext[e].relative_to(ORIG)}")

print("\ntop-level entries:")
for p in sorted(ORIG.iterdir())[:15]:
    print("  ", p.name, "(dir)" if p.is_dir() else "(file)")

print("\nSTOP. Read the above before continuing. The conversion step after this "
      "assumes an annotation format; confirm which one is actually present.")

# %%
# ---- Step 5: load and verify the categorization metadata --------------------
# Codes were confirmed twice on 2026-08-31: against the FOD-A paper's Table I
# row counts, and against the distribution's own category_information.txt.
# NOTE the light ordering - 0 is Bright and 2 is Dark. An assumed ordering
# would invert the finding and report best performance in darkness.
import csv
from collections import Counter
from pathlib import Path

CSV = Path("/content/fod-a-original/FullDatasetV.2.1-400x400/"
           "All_Dataset_Utility_Files/FOD_categorization_annotations.csv")
WEATHER = {"0": "dry", "1": "wet"}
LIGHT = {"0": "bright", "1": "dim", "2": "dark"}

with CSV.open(newline="", encoding="utf-8", errors="replace") as fh:
    rows = list(csv.DictReader(fh))

w = Counter(WEATHER[r["Weather"].strip()] for r in rows)
l = Counter(LIGHT[r["Light"].strip()] for r in rows)
print("rows:", len(rows))
print("weather:", dict(w))
print("light  :", dict(l))

# Paper Table I. A mismatch means the distribution differs from the published
# one and the mapping must be re-derived rather than assumed.
assert len(rows) == 33863, f"expected 33,863 rows, got {len(rows)}"
assert w == Counter({"dry": 26647, "wet": 7216}), w
assert l == Counter({"bright": 17012, "dim": 12464, "dark": 4387}), l
print("\nPASS: matches the FOD-A paper's Table I exactly; code mapping confirmed.")

# %%
# ---- Step 6: build the split ------------------------------------------------
# Same parameters as run 3 so the SPLIT METHOD is held constant even though the
# source imagery differs: seed 42, test_frac 0.15, small_test_frac 0.40.
# Record the fingerprints - they will differ from run 3's, correctly, because
# the source is different.
#
# NOTE: --source must point at a YOLO-format images/+labels/ root. If Step 4
# showed the original distribution does not carry YOLO or Pascal VOC labels
# directly, a conversion step belongs here first. Do not skip past this.
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")

!python {REPO}/src/data_prep.py --build-split \
    --source /content/fod-a-400-yolo \
    --out /content/fod-a-split-run4 \
    --small-object-max-area-pct 0.5 \
    --test-frac 0.15 \
    --small-test-frac 0.40 \
    --seed 42 \
    --config {REPO}/configs/fod.yaml

# %%
# ---- Step 7: record the fingerprints ----------------------------------------
import json
from pathlib import Path

m = json.loads(Path("/content/fod-a-split-run4/split_manifest.json").read_text())
print("small held out :", m["test_bucket_counts"]["small"], " (run 3 had 123)")
print("test total     :", m["n_test"], " (run 3 had 870)")
print("\nFINGERPRINTS — record in docs/RESEARCH_LOG.md:")
for k, v in m["fingerprints"].items():
    print(f"  {k:14s} {v}")
print("\nThese SHOULD differ from run 3's. Different source imagery, so a "
      "different split is correct, not a defect.")

# %%
# ---- Step 8: train ----------------------------------------------------------
import sys
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")
sys.path.insert(0, str(REPO))
from src.colab_helpers import describe_runs, find_latest_run
from ultralytics import YOLO

RUNS = Path("/content/runs/detect")
print(describe_runs(RUNS))

# Resume only a run4 checkpoint. find_latest_run() picks the newest run of ANY
# name, which would happily resume run3 into this experiment.
run4 = RUNS / "run4-orig400"
if (run4 / "weights" / "last.pt").exists():
    print(f"\nRESUMING {run4}")
    model = YOLO(str(run4 / "weights" / "last.pt"))
    results = model.train(resume=True)
else:
    print("\nSTARTING run4-orig400")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data="/content/fod-a-split-run4/data.yaml",
        epochs=100, imgsz=640, batch=16, seed=42,
        project=str(RUNS), name="run4-orig400", exist_ok=False,
    )

# %%
# ---- Step 9: benchmark, and THEN the stratified benchmark -------------------
# The second call is the one this whole run exists for: filenames in the 400x400
# distribution match the CSV's File column, so the light/weather breakdown can
# finally be produced. AC 150/5220-24 section 3.2.b(6)(c) requires demonstrating
# performance across daylight, nighttime and dawn/dusk; section 3.2.b(6) across
# clear and inclement weather. No reviewed vendor publishes either.
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")
W = Path("/content/runs/detect/run4-orig400/weights/best.pt")
CSV = ("/content/fod-a-original/FullDatasetV.2.1-400x400/"
       "All_Dataset_Utility_Files/FOD_categorization_annotations.csv")

!python {REPO}/src/benchmark_faa.py --weights {W} --data /content/fod-a-split-run4/data.yaml --config {REPO}/configs/fod.yaml --out {REPO}/docs/benchmark_results

!python {REPO}/src/benchmark_faa.py --weights {W} --data /content/fod-a-split-run4/data.yaml --config {REPO}/configs/fod.yaml --metadata-csv {CSV} --out {REPO}/docs/benchmark_results

# CHECK the "[metadata] Loaded light/weather labels for N images" line against
# the actual test-set size. If it loaded 0, or a number far below the test set,
# the join failed and the stratified breakdown must NOT be reported.

# %%
# ---- Step 10: save everything that matters ----------------------------------
import shutil
from pathlib import Path

DEST = Path("/content/drive/MyDrive/ASSIS_FOD_run4_artifacts")
DEST.mkdir(parents=True, exist_ok=True)
shutil.copy2("/content/fod-a-split-run4/split_manifest.json", DEST)
for p in Path("/content/ASSIS-FOD-Detection/docs/benchmark_results").glob("*"):
    shutil.copy2(p, DEST)
print("saved to", DEST)
print(sorted(p.name for p in DEST.iterdir()))
