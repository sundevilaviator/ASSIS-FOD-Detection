#!/usr/bin/env python3
"""Helpers for running the FOD pipeline in a hosted notebook (Colab).

These live in `src/` rather than inline in the notebook file for one reason:
they are the two pieces that actually went wrong in practice, so they need
tests. Everything else in the notebook is orchestration.

WHAT WENT WRONG, AND WHAT THESE FIX
-----------------------------------
1. `find_voc_root()` — the notebook previously hardcoded a flat
   `JPEGImages/` + `Annotations/` layout. The real FOD-A download nests them
   under `FODPascalVOCFormat-V.2.1/VOC2007/`. Hardcoding the corrected path
   would just move the failure to the next dataset revision, since the
   version string is in it. This searches instead.

2. `find_latest_run()` — checkpoints were selected with
   `sorted(glob.glob(...))[-1]`. `glob.glob` returns strings, and in raw
   string order '-' (0x2D) precedes '/' (0x2F), so with run directories
   named `train` and `train-2` the path 'train-2/weights/last.pt' sorts
   BEFORE 'train/weights/last.pt' and `[-1]` returns `train`. Resuming then
   loaded a *completed* run and exited at 100/100 instead of continuing the
   partial one.

   `Path.glob()` would have compared by parts tuple and returned `train-2`
   here — but only incidentally: it still orders `train-10` before
   `train-9`. No lexical ordering is correct for this. Modification time is.
"""
from __future__ import annotations

from pathlib import Path

IMAGE_DIR_NAMES = ("JPEGImages", "images", "Images", "JPEGImage")
ANNOT_DIR_NAMES = ("Annotations", "annotations", "Annotation")


def find_voc_root(raw_dir: Path) -> tuple[Path, Path]:
    """Locate the Pascal VOC images/annotations pair under `raw_dir`.

    Returns (annotations_dir, images_dir). Searches rather than assuming a
    layout, because the archive nests them under a version-stamped folder
    whose name will change with the next dataset revision.

    Raises FileNotFoundError with the actual directory tree in the message,
    so a failure tells you what IS there instead of only what is missing.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"{raw_dir} does not exist — did the download step run?")

    for annot_name in ANNOT_DIR_NAMES:
        for annot in sorted(raw_dir.rglob(annot_name)):
            if not annot.is_dir():
                continue
            if not any(annot.glob("*.xml")):
                continue
            for img_name in IMAGE_DIR_NAMES:
                images = annot.parent / img_name
                if images.is_dir() and any(
                    images.glob(f"*{ext}") for ext in (".jpg", ".jpeg", ".png")
                ):
                    return annot, images

    found = sorted({p.parent.relative_to(raw_dir) for p in raw_dir.rglob("*") if p.is_file()})
    raise FileNotFoundError(
        f"No Pascal VOC annotations+images pair found under {raw_dir}.\n"
        f"Directories containing files:\n  "
        + "\n  ".join(str(d) for d in found[:25])
    )


def find_latest_run(runs_detect_dir: Path) -> Path | None:
    """Return the most recently modified YOLO run directory, or None.

    Ordered by mtime, NOT by name. Name ordering is what caused a partially
    trained `train-2` to lose to a finished `train`.
    """
    runs_detect_dir = Path(runs_detect_dir)
    if not runs_detect_dir.exists():
        return None
    candidates = [
        d for d in runs_detect_dir.iterdir()
        if d.is_dir() and (d / "weights" / "last.pt").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / "weights" / "last.pt").stat().st_mtime)


def describe_runs(runs_detect_dir: Path) -> str:
    """Human-readable listing of every run directory and its checkpoint age.

    Printed before resuming so the choice is visible rather than implicit —
    the silent wrong-directory pick is precisely the failure mode here.
    """
    runs_detect_dir = Path(runs_detect_dir)
    if not runs_detect_dir.exists():
        return f"(no runs directory at {runs_detect_dir})"
    rows = []
    latest = find_latest_run(runs_detect_dir)
    for d in sorted(runs_detect_dir.iterdir()):
        if not d.is_dir():
            continue
        last = d / "weights" / "last.pt"
        best = d / "weights" / "best.pt"
        if not last.exists():
            rows.append(f"  {d.name:<16} (no checkpoint)")
            continue
        mark = "  <-- most recent" if d == latest else ""
        rows.append(
            f"  {d.name:<16} last.pt {last.stat().st_size / 1e6:6.1f} MB"
            f"{'  best.pt present' if best.exists() else ''}{mark}"
        )
    return "\n".join(rows) if rows else f"(no run directories in {runs_detect_dir})"
