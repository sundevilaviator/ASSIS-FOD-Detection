#!/usr/bin/env python3
# %% [markdown]
# # ASSIS FOD — SAHI sliced-inference experiment
#
# What this tests, and why it might NOT help
# -------------------------------------------
# SAHI (Slicing Aided Hyper Inference) tiles an image into overlapping crops,
# runs the SAME trained weights on each tile, and merges detections back into
# full-image coordinates. The idea: a small object that's a tiny fraction of
# a full frame becomes a much larger fraction of a tile, giving the model
# more effective pixels-on-target.
#
# BUT: the 2026-08-31 inference-resolution sweep (docs/RESEARCH_LOG.md,
# session 3) found that raising --imgsz does NOTHING for small-object
# detection on run 3's held-out split (52.0% at 640/960/1280, identical),
# because FOD-A's Pascal VOC mirror images are already only 300x300 — there
# is no extra detail to recover from an already-small source.
#
# Tiling a 300x300 image does not manufacture pixels that were never
# captured either. So the honest prediction going in is: this benchmark
# will probably show little or no improvement on FOD-A's own held-out split,
# for the SAME reason as the resolution sweep. SAHI's more plausible benefit
# is on higher-resolution real-world photos (a phone photo, a security
# camera frame) where a small object is a tiny fraction of a much bigger
# frame — a case this benchmark does not represent.
#
# This notebook runs BOTH conditions (plain vs. --sahi) against the SAME
# run 3 weights and SAME held-out split, so whichever way it comes out is a
# real, comparable number — not an assumption. A negative result here is
# still useful: it would confirm the prediction above rather than leave it
# untested, exactly like the resolution sweep did.
#
# Every cell re-derives its own paths, for the same reason as runs 3/4.

# %%
# ---- Step 1: environment and repo -------------------------------------------
!pip -q install ultralytics sahi

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
# ---- Step 2: Drive access (run 3's weights and split live there) -----------
import os
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive", force_remount=False)
DRIVE_RUNS = Path("/content/drive/MyDrive/ASSIS_FOD_runs")
WEIGHTS = DRIVE_RUNS / "detect" / "run3" / "weights" / "best.pt"
SPLIT = Path("/content/fod-a-split-run3")
assert WEIGHTS.exists(), (
    f"{WEIGHTS} not found. If this Colab VM was recycled since run 3, you may "
    "need to re-run run 3's Step 1-9 (or just its split-building step) before "
    "this notebook has a data.yaml and held-out images to benchmark against."
)
print("weights:", WEIGHTS, WEIGHTS.stat().st_size, "bytes")

# %%
# ---- Step 3: confirm the held-out split is present (rebuild if not) --------
# If /content/fod-a-split-run3 doesn't exist on this VM, rebuild it exactly as
# run 3 did — same seed/fractions, so it is the SAME split, not a new one.
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")
SPLIT = Path("/content/fod-a-split-run3")

if not (SPLIT / "data.yaml").exists():
    print("Split not found on this VM — rebuilding with run 3's exact parameters.")
    !python {REPO}/src/data_prep.py --build-split \
        --source /content/fod-a-yolo \
        --out {SPLIT} \
        --small-object-max-area-pct 0.5 \
        --test-frac 0.15 \
        --small-test-frac 0.40 \
        --seed 42 \
        --config {REPO}/configs/fod.yaml
else:
    print("Found existing split:", SPLIT)

# %%
# ---- Step 4: baseline (plain) benchmark — should reproduce 52.0% ----------
# This is a sanity check before trusting anything from --sahi below: if this
# doesn't reproduce run 3's known 52.0% small-object rate, something about
# this VM's data/weights differs from run 3 and the SAHI comparison would not
# be trustworthy either.
from pathlib import Path
REPO = Path("/content/ASSIS-FOD-Detection")
WEIGHTS = Path("/content/drive/MyDrive/ASSIS_FOD_runs/detect/run3/weights/best.pt")
SPLIT = Path("/content/fod-a-split-run3")

!python {REPO}/src/benchmark_faa.py \
    --weights {WEIGHTS} \
    --data {SPLIT}/data.yaml \
    --config {REPO}/configs/fod.yaml \
    --out {REPO}/docs/benchmark_results/sahi_experiment

# CHECK: the small-object detection rate printed above should be 52.0%
# (64/123). If it isn't, stop and figure out why before continuing — do not
# treat a --sahi result as meaningful against a baseline that doesn't match.

# %%
# ---- Step 5: SAHI sliced-inference benchmark, same weights/split/split ----
!python {REPO}/src/benchmark_faa.py \
    --weights {WEIGHTS} \
    --data {SPLIT}/data.yaml \
    --config {REPO}/configs/fod.yaml \
    --out {REPO}/docs/benchmark_results/sahi_experiment \
    --sahi \
    --sahi-slice-size 512 \
    --sahi-overlap 0.2

# %%
# ---- Step 6: compare the two reports side by side --------------------------
import json
from pathlib import Path

OUT = Path("/content/ASSIS-FOD-Detection/docs/benchmark_results/sahi_experiment")
reports = sorted(OUT.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime)
assert len(reports) >= 2, f"Expected at least 2 reports in {OUT}, found {len(reports)}"
plain, sahi = json.loads(reports[-2].read_text()), json.loads(reports[-1].read_text())

def small(r):
    return r["results_by_size_bucket"]["small"]["detection_rate"]

print(f"{'condition':<12} {'small det. rate':<18} {'n':<6} {'fp/image':<10}")
for label, r in [("plain", plain), ("sahi", sahi)]:
    rate = small(r)
    rate_str = f"{rate:.1%}" if rate is not None else "n/a"
    n = r["results_by_size_bucket"]["small"]["n_ground_truth"]
    fp = r["false_positives_per_image"]
    print(f"{label:<12} {rate_str:<18} {n:<6} {fp:.3f}")

print(
    "\nIf these two rates are close (within noise for n=123), that CONFIRMS the "
    "prediction in this notebook's header: tiling a 300x300 source adds no "
    "detail SAHI can recover, so this lever is ruled out for FOD-A specifically "
    "— record that in docs/RESEARCH_LOG.md exactly like the imgsz sweep. If SAHI "
    "clearly wins here despite the reasoning above, that is a genuinely "
    "surprising and worth-investigating result — check for a bug (e.g. class "
    "mismatch, box-merging duplicates counted as extra true positives) before "
    "trusting it."
)

# %%
# ---- Step 7: (optional) test on a higher-resolution photo, not FOD-A ------
# This is the case SAHI is actually built for. Upload a real photo (NOT from
# FOD-A) that is larger than 300x300 and see whether --sahi finds a small
# object a plain pass misses. This step has no ground truth to score against
# automatically — it's a visual check, not a benchmark number — upload a
# photo via the Colab file browser and point PHOTO at it.
from pathlib import Path

PHOTO = Path("/content/your_test_photo.jpg")  # <-- set this after uploading
if PHOTO.exists():
    from ultralytics import YOLO
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    model = YOLO(str(WEIGHTS))
    plain_pred = model.predict(source=str(PHOTO), conf=0.35, verbose=False)[0]
    print(f"Plain pass: {len(plain_pred.boxes)} detection(s)")

    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(WEIGHTS), confidence_threshold=0.35,
        device="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
    )
    sahi_result = get_sliced_prediction(
        str(PHOTO), sahi_model, slice_height=512, slice_width=512,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
    )
    print(f"SAHI pass: {len(sahi_result.object_prediction_list)} detection(s)")
    sahi_result.export_visuals(export_dir="/content/sahi_visual_check")
    print("Annotated SAHI result saved to /content/sahi_visual_check/prediction_visual.png")
else:
    print(f"Skipped — {PHOTO} does not exist. Upload a photo and set PHOTO above to run this step.")

# %%
# ---- Step 8: save everything that matters ----------------------------------
import shutil
from pathlib import Path

DEST = Path("/content/drive/MyDrive/ASSIS_FOD_sahi_experiment")
DEST.mkdir(parents=True, exist_ok=True)
OUT = Path("/content/ASSIS-FOD-Detection/docs/benchmark_results/sahi_experiment")
for p in OUT.glob("*"):
    shutil.copy2(p, DEST)
print("saved to", DEST)
print(sorted(p.name for p in DEST.iterdir()))
