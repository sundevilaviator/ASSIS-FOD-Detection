"""
Unit tests for src/data_prep.py's dataset-discovery and size-bucketing logic,
using a small synthetic YOLO-format dataset built in a tmp_path fixture — no
real dataset or network access required.
"""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_prep import LabeledImage, find_labeled_images  # noqa: E402

THRESHOLDS = {
    "small": {"max_area_pct": 0.5},
    "medium": {"max_area_pct": 2.0},
    "large": {"max_area_pct": None},
}


def _make_dataset(root: Path) -> None:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # One small-object image (tiny bbox), one large-object image.
    Image.new("RGB", (640, 640), color="gray").save(images_dir / "small_case.jpg")
    (labels_dir / "small_case.txt").write_text("0 0.5 0.5 0.02 0.02\n")  # area = 0.04% of frame

    Image.new("RGB", (640, 640), color="gray").save(images_dir / "large_case.jpg")
    (labels_dir / "large_case.txt").write_text("1 0.5 0.5 0.4 0.4\n")  # area = 16% of frame

    # Unlabeled image should be skipped entirely.
    Image.new("RGB", (640, 640), color="gray").save(images_dir / "no_label.jpg")


def test_find_labeled_images_skips_unlabeled(tmp_path: Path):
    _make_dataset(tmp_path)
    items = find_labeled_images(tmp_path)
    names = sorted(i.image_path.name for i in items)
    assert names == ["large_case.jpg", "small_case.jpg"]


def test_size_bucket_classification(tmp_path: Path):
    _make_dataset(tmp_path)
    items = {i.image_path.name: i for i in find_labeled_images(tmp_path)}

    small_item = items["small_case.jpg"]
    large_item = items["large_case.jpg"]

    assert small_item.max_box_area_pct() == pytest.approx(0.04, abs=1e-6)
    assert small_item.size_bucket(THRESHOLDS) == "small"

    assert large_item.max_box_area_pct() == pytest.approx(16.0, abs=1e-6)
    assert large_item.size_bucket(THRESHOLDS) == "large"


def test_labeled_image_with_no_boxes_is_zero_area():
    item = LabeledImage(Path("x.jpg"), Path("x.txt"), boxes=[])
    assert item.max_box_area_pct() == 0.0
    assert item.size_bucket(THRESHOLDS) == "small"
