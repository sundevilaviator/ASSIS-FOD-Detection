"""Tests for split fingerprinting and the small-bucket test fraction.

Why these exist
---------------
The 2026-08-23 defect (filesystem enumeration order feeding a seeded shuffle)
produced two runs whose split *counts* matched exactly while the actual
membership differed. The manifest agreed with itself and hid the problem.
Worse, the manifest's `test_bucket_counts` key never reported test counts at
all — it reported source bucket totals, which are identical across runs by
construction. So the one field that looked like a check was structurally
incapable of catching the error.

Two changes are covered here:

  * `split_fingerprints()` — SHA-256 over sorted membership, so "same source,
    same seed, same split" is verifiable by comparing strings rather than
    trusting aggregates;
  * `--small-test-frac` — a larger held-out fraction for the small bucket
    only, because the small-object detection rate is the figure that bounds
    every claim made about this model, and its interval is set by the number
    of small instances held out.

Every expected value below is computed by hand in the test, not read back
from the implementation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep import LabeledImage, build_split, split_fingerprints  # noqa: E402


def _item(name: str, w: float, h: float) -> LabeledImage:
    """A LabeledImage with one box of the given normalized width/height.

    max_box_area_pct() == w * h * 100, so the caller controls the bucket
    directly and the arithmetic stays hand-checkable.
    """
    return LabeledImage(Path(f"/src/{name}"), Path(f"/src/{Path(name).stem}.txt"),
                        [(0, 0.5, 0.5, w, h)])


# --------------------------------------------------------------------------
# split_fingerprints
# --------------------------------------------------------------------------

def test_fingerprint_matches_a_hand_computed_sha256():
    """Pin the exact digest, computed here independently of the source."""
    items = [_item("b.jpg", 0.1, 0.1), _item("a.jpg", 0.1, 0.1)]
    expected = hashlib.sha256("a.jpg\nb.jpg".encode("utf-8")).hexdigest()
    got = split_fingerprints(items, [], {})["train"]
    assert got == expected, f"{got} != hand-computed {expected}"


def test_fingerprint_is_independent_of_input_order():
    """The whole point: enumeration order must not change the digest."""
    a, b, c = _item("a.jpg", 0.1, 0.1), _item("b.jpg", 0.1, 0.1), _item("c.jpg", 0.1, 0.1)
    one = split_fingerprints([a, b, c], [], {})["train"]
    two = split_fingerprints([c, a, b], [], {})["train"]
    assert one == two


def test_fingerprint_changes_when_membership_changes():
    """A digest that never changed would pass every other test and be useless."""
    a, b, c = _item("a.jpg", 0.1, 0.1), _item("b.jpg", 0.1, 0.1), _item("c.jpg", 0.1, 0.1)
    assert split_fingerprints([a, b], [], {})["train"] != split_fingerprints([a, c], [], {})["train"]


def test_fingerprint_would_have_caught_the_2026_08_23_defect():
    """Reproduce the defect's signature: same counts, different membership.

    Six images, three held out either way. Aggregate counts are identical —
    which is exactly what the old manifest compared, and why it agreed.
    """
    pool = {n: _item(f"{n}.jpg", 0.1, 0.1) for n in "abcdef"}
    run1_test = [pool[n] for n in "abc"]
    run2_test = [pool[n] for n in "abd"]
    assert len(run1_test) == len(run2_test), "fixture must hold counts equal"

    f1 = split_fingerprints([], run1_test, {"small": run1_test})
    f2 = split_fingerprints([], run2_test, {"small": run2_test})
    assert f1["test"] != f2["test"]
    assert f1["test_small"] != f2["test_small"]


def test_per_bucket_fingerprints_are_emitted():
    small = [_item("s1.jpg", 0.05, 0.05)]
    large = [_item("l1.jpg", 0.9, 0.9)]
    fp = split_fingerprints([], small + large, {"small": small, "large": large})
    assert {"train", "test", "test_small", "test_large"} <= set(fp)
    assert fp["test_small"] != fp["test_large"]


# --------------------------------------------------------------------------
# build_split: --small-test-frac and the corrected manifest
# --------------------------------------------------------------------------

def _make_source(tmp_path: Path, n_small: int, n_large: int) -> Path:
    """Write a real YOLO-format source tree so build_split can copy files."""
    from PIL import Image

    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir(parents=True)

    def write(stem: str, w: float, h: float):
        Image.new("RGB", (64, 64), (128, 128, 128)).save(src / "images" / f"{stem}.jpg")
        (src / "labels" / f"{stem}.txt").write_text(f"0 0.5 0.5 {w} {h}\n")

    # area_pct = w*h*100. small bucket is <= 0.5, large is > 2.0.
    for i in range(n_small):
        write(f"small{i:03d}", 0.05, 0.05)   # 0.25 -> small
    for i in range(n_large):
        write(f"large{i:03d}", 0.9, 0.9)     # 81.0 -> large
    return src


def _build(tmp_path, **kw) -> dict:
    out = tmp_path / "out"
    build_split(
        source=_make_source(tmp_path, kw.pop("n_small", 100), kw.pop("n_large", 100)),
        out=out,
        small_object_max_area_pct=0.5,
        oversample_factor=1,
        class_names=["obj"],
        seed=42,
        **kw,
    )
    return json.loads((out / "split_manifest.json").read_text())


def test_small_test_frac_enlarges_only_the_small_bucket(tmp_path):
    """Hand-computed: 100 small at 0.40 -> 40 held out; 100 large at 0.15 -> 15."""
    m = _build(tmp_path, test_frac=0.15, small_test_frac=0.40)
    assert m["test_bucket_counts"]["small"] == 40
    assert m["test_bucket_counts"]["large"] == 15
    assert m["n_test"] == 55


def test_omitting_small_test_frac_leaves_behaviour_unchanged(tmp_path):
    """Hand-computed: both buckets at 0.15 -> 15 each, 30 total."""
    m = _build(tmp_path, test_frac=0.15)
    # `medium` is reported as 0 rather than omitted — an empty bucket is a
    # fact about the split, and dropping the key would make two manifests
    # with different bucket structure look alike.
    assert m["test_bucket_counts"] == {"small": 15, "medium": 0, "large": 15}
    assert m["n_test"] == 30


def test_manifest_reports_test_counts_not_source_totals(tmp_path):
    """Regression for the mislabelled key that hid the ordering defect.

    `test_bucket_counts` used to report every image in each bucket. With 100
    small and 100 large at test_frac=0.15 the honest answer is 15/15, not
    100/100 — and the two are now reported under separate names.
    """
    m = _build(tmp_path, test_frac=0.15)
    assert m["source_bucket_counts"] == {"small": 100, "medium": 0, "large": 100}
    assert m["test_bucket_counts"] == {"small": 15, "medium": 0, "large": 15}
    assert m["test_bucket_counts"] != m["source_bucket_counts"], (
        "test counts must not equal source totals — that was the defect"
    )


def test_same_seed_gives_identical_fingerprints(tmp_path):
    a = _build(tmp_path / "a", test_frac=0.15, small_test_frac=0.40)
    b = _build(tmp_path / "b", test_frac=0.15, small_test_frac=0.40)
    assert a["fingerprints"] == b["fingerprints"]


def test_different_seed_changes_the_fingerprint(tmp_path):
    """Guards against a 'fix' that sorts everything and never shuffles."""
    out_a, out_b = tmp_path / "a" / "out", tmp_path / "b" / "out"
    for out, seed in ((out_a, 42), (out_b, 7)):
        build_split(
            source=_make_source(out.parent, 100, 100), out=out,
            small_object_max_area_pct=0.5, test_frac=0.15, oversample_factor=1,
            class_names=["obj"], seed=seed, small_test_frac=0.40,
        )
    a = json.loads((out_a / "split_manifest.json").read_text())
    b = json.loads((out_b / "split_manifest.json").read_text())
    assert a["fingerprints"]["test"] != b["fingerprints"]["test"]
    assert a["test_bucket_counts"] == b["test_bucket_counts"], (
        "counts should match while membership differs — that is the point"
    )


def test_manifest_records_the_fractions_used(tmp_path):
    """The manifest must say what it was built with, or it can't be audited."""
    m = _build(tmp_path, test_frac=0.15, small_test_frac=0.40)
    assert m["test_frac"] == 0.15
    assert m["small_test_frac"] == 0.40
    assert m["seed"] == 42
