"""Tests for the two notebook helpers that previously failed in practice.

Both bugs these cover cost real time in a live session:

  * a hardcoded flat JPEGImages/Annotations layout that did not match the
    actual FOD-A archive (nested under FODPascalVOCFormat-V.2.1/VOC2007/);
  * checkpoint selection by `sorted(...)[-1]`, which returns `train` rather
    than `train-2` because '-' sorts before '/' in the full path string, so
    resuming a partial run silently loaded a finished one and exited.

Both are verified here against hand-built fixtures, including the exact
directory names that produced the original failures.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.colab_helpers import (  # noqa: E402
    describe_runs,
    find_latest_run,
    find_voc_root,
)


# --------------------------------------------------------------------------
# find_voc_root
# --------------------------------------------------------------------------

def _voc_tree(root: Path, nest: str = "", images="JPEGImages", annots="Annotations"):
    base = root / nest if nest else root
    (base / annots).mkdir(parents=True, exist_ok=True)
    (base / images).mkdir(parents=True, exist_ok=True)
    (base / annots / "000000.xml").write_text("<annotation></annotation>")
    (base / images / "000000.jpg").write_bytes(b"\xff\xd8\xff")
    return base


def test_finds_real_fod_a_nested_layout(tmp_path):
    """The exact layout the live download produced."""
    raw = tmp_path / "fod-a-raw"
    _voc_tree(raw, nest="FODPascalVOCFormat-V.2.1/VOC2007")
    annot, images = find_voc_root(raw)
    assert annot.name == "Annotations"
    assert images.name == "JPEGImages"
    assert "VOC2007" in str(annot)


def test_finds_flat_layout(tmp_path):
    """The layout originally assumed must still work."""
    raw = tmp_path / "fod-a-raw"
    _voc_tree(raw)
    annot, images = find_voc_root(raw)
    assert annot.parent == raw


def test_survives_a_version_bump_in_the_folder_name(tmp_path):
    """The whole point of searching: the version string will change."""
    raw = tmp_path / "fod-a-raw"
    _voc_tree(raw, nest="FODPascalVOCFormat-V.9.9/VOC2031")
    annot, images = find_voc_root(raw)
    assert "V.9.9" in str(annot)


def test_ignores_an_annotations_dir_with_no_xml(tmp_path):
    """A decoy empty Annotations/ must not win over the real pair."""
    raw = tmp_path / "fod-a-raw"
    (raw / "decoy" / "Annotations").mkdir(parents=True)
    (raw / "decoy" / "JPEGImages").mkdir(parents=True)
    _voc_tree(raw, nest="real/VOC2007")
    annot, images = find_voc_root(raw)
    assert "real" in str(annot)


def test_ignores_annotations_without_a_sibling_image_dir(tmp_path):
    raw = tmp_path / "fod-a-raw"
    lone = raw / "lonely" / "Annotations"
    lone.mkdir(parents=True)
    (lone / "a.xml").write_text("<annotation></annotation>")
    _voc_tree(raw, nest="real/VOC2007")
    annot, _ = find_voc_root(raw)
    assert "real" in str(annot)


def test_missing_dir_raises_with_a_useful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        find_voc_root(tmp_path / "nope")


def test_error_lists_what_was_actually_found(tmp_path):
    """A failure should say what IS there, not only what is missing."""
    raw = tmp_path / "fod-a-raw"
    (raw / "SomethingElse").mkdir(parents=True)
    (raw / "SomethingElse" / "readme.txt").write_text("hi")
    with pytest.raises(FileNotFoundError) as e:
        find_voc_root(raw)
    assert "SomethingElse" in str(e.value)


# --------------------------------------------------------------------------
# find_latest_run
# --------------------------------------------------------------------------

def _run_dir(root: Path, name: str, mtime: float | None = None, best=False):
    w = root / name / "weights"
    w.mkdir(parents=True, exist_ok=True)
    (w / "last.pt").write_bytes(b"x" * 100)
    if best:
        (w / "best.pt").write_bytes(b"x" * 100)
    if mtime is not None:
        os.utime(w / "last.pt", (mtime, mtime))
    return root / name


def test_picks_most_recent_not_alphabetically_last(tmp_path):
    """The exact bug: 'train' vs 'train-2'.

    The failing code used `glob.glob(...)`, which returns STRINGS. In raw
    string order '-' (0x2D) precedes '/' (0x2F), so 'train-2/weights/last.pt'
    sorts before 'train/weights/last.pt' and [-1] returns the finished
    'train'. Resuming then loaded a completed run and exited at 100/100.

    Note `Path.glob()` would compare by parts tuple and happen to return
    'train-2' here — but only by luck: it still orders 'train-10' before
    'train-9'. mtime is correct in both cases, which is why it is used.
    """
    import glob as globmod

    runs = tmp_path / "detect"
    now = time.time()
    _run_dir(runs, "train", mtime=now - 10_000)     # older, completed
    _run_dir(runs, "train-2", mtime=now)            # newer, in progress

    # Reproduce the original defect exactly: string sort, as glob.glob gives.
    naive = sorted(globmod.glob(str(runs / "train*" / "weights" / "last.pt")))[-1]
    assert Path(naive).parent.parent.name == "train", (
        "fixture no longer reproduces the string-sort bug"
    )

    assert find_latest_run(runs).name == "train-2"


def test_mtime_ordering_also_survives_double_digit_run_numbers(tmp_path):
    """Lexical ordering of any kind fails on train-10 vs train-9."""
    runs = tmp_path / "detect"
    now = time.time()
    _run_dir(runs, "train-9", mtime=now - 5_000)
    _run_dir(runs, "train-10", mtime=now)           # newest
    assert sorted(["train-10", "train-9"])[-1] == "train-9", "lexical order picks wrong"
    assert find_latest_run(runs).name == "train-10"


def test_ignores_directories_without_a_checkpoint(tmp_path):
    runs = tmp_path / "detect"
    _run_dir(runs, "train", mtime=time.time() - 500)
    (runs / "train-3").mkdir(parents=True)  # no weights/
    assert find_latest_run(runs).name == "train"


def test_returns_none_when_nothing_present(tmp_path):
    assert find_latest_run(tmp_path / "detect") is None
    (tmp_path / "empty").mkdir()
    assert find_latest_run(tmp_path / "empty") is None


def test_describe_runs_marks_the_selected_one(tmp_path):
    runs = tmp_path / "detect"
    now = time.time()
    _run_dir(runs, "train", mtime=now - 10_000, best=True)
    _run_dir(runs, "train-2", mtime=now)
    out = describe_runs(runs)
    assert "train-2" in out and "most recent" in out
    line = [l for l in out.splitlines() if "most recent" in l][0]
    assert "train-2" in line, "the 'most recent' marker is on the wrong row"


def test_describe_runs_handles_missing_directory(tmp_path):
    assert "no runs directory" in describe_runs(tmp_path / "nope")
