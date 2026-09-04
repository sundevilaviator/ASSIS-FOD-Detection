"""
Unit tests for the pure-logic pieces of src/benchmark_faa.py (IoU matching,
size bucketing) using synthetic data — no model weights or dataset required.
These import cleanly without ultralytics/PIL because benchmark_faa.py only
imports those inside run_benchmark(), not at module load time.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark_faa import bucket_for_area_pct, iou, parse_metadata_csv, resolve_labels_dir, yolo_to_xyxy  # noqa: E402

THRESHOLDS = {
    "small": {"max_area_pct": 0.5},
    "medium": {"max_area_pct": 2.0},
    "large": {"max_area_pct": None},
}


def test_iou_identical_boxes_is_one():
    box = (10.0, 10.0, 50.0, 50.0)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 100.0, 110.0, 110.0)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    # intersection = 5x5 = 25, union = 100+100-25 = 175
    assert abs(iou(a, b) - 25 / 175) < 1e-9


def test_yolo_to_xyxy_center_box():
    # A box centered at (0.5, 0.5) covering half the image in each dimension,
    # on a 100x200 image, should span x:[25,75], y:[50,150].
    x1, y1, x2, y2 = yolo_to_xyxy(0.5, 0.5, 0.5, 0.5, img_w=100, img_h=200)
    assert (x1, y1, x2, y2) == (25.0, 50.0, 75.0, 150.0)


def test_bucket_for_area_pct_boundaries():
    assert bucket_for_area_pct(0.1, THRESHOLDS) == "small"
    assert bucket_for_area_pct(0.5, THRESHOLDS) == "small"       # inclusive boundary
    assert bucket_for_area_pct(0.51, THRESHOLDS) == "medium"
    assert bucket_for_area_pct(2.0, THRESHOLDS) == "medium"      # inclusive boundary
    assert bucket_for_area_pct(2.01, THRESHOLDS) == "large"
    assert bucket_for_area_pct(99.0, THRESHOLDS) == "large"


def test_parse_metadata_csv_autodetects_common_column_names():
    csv_text = (
        "filename,light_level,weather\n"
        "img001.jpg,bright,dry\n"
        "img002.jpg,dim,wet\n"
        "img003.png,dark,dry\n"
    )
    metadata, warnings = parse_metadata_csv(csv_text)
    assert warnings == []
    assert metadata["img001"] == {"light": "bright", "weather": "dry"}
    assert metadata["img002"] == {"light": "dim", "weather": "wet"}
    # stem is taken regardless of extension
    assert metadata["img003"] == {"light": "dark", "weather": "dry"}


def test_parse_metadata_csv_explicit_column_override():
    csv_text = "img,cond_light,cond_weather\nfoo.jpg,bright,dry\n"
    metadata, warnings = parse_metadata_csv(
        csv_text, image_col="img", light_col="cond_light", weather_col="cond_weather"
    )
    assert warnings == []
    assert metadata["foo"] == {"light": "bright", "weather": "dry"}


def test_parse_metadata_csv_missing_image_column_warns_and_returns_empty():
    csv_text = "light_level,weather\nbright,dry\n"
    metadata, warnings = parse_metadata_csv(csv_text)
    assert metadata == {}
    assert len(warnings) == 1
    assert "image/filename column" in warnings[0]


def test_parse_metadata_csv_missing_condition_columns_warns_and_returns_empty():
    csv_text = "filename,unrelated_column\nimg001.jpg,foo\n"
    metadata, warnings = parse_metadata_csv(csv_text)
    assert metadata == {}
    assert len(warnings) == 1
    assert "light-level or weather column" in warnings[0]


def test_parse_metadata_csv_partial_light_only_still_works():
    csv_text = "filename,light_level\nimg001.jpg,bright\n"
    metadata, warnings = parse_metadata_csv(csv_text)
    assert warnings == []
    assert metadata["img001"] == {"light": "bright"}


def test_parse_metadata_csv_handles_windows_backslash_object_folder_paths():
    """Regression test for a real bug (found 2026-09-03, run 4): FOD-A's
    original-format categorization CSV writes its File column as
    OBJECT_FOLDER\\frame\\FILENAME.PNG using Windows-style backslashes.
    pathlib does not split on backslash under Linux/Colab, so a plain
    Path(raw_name).stem left the backslashes embedded in the "stem" -- a key
    that could never match a real image -- and separately dropped which
    per-object folder a frame came from. That object folder is required, not
    optional context: every object folder in this distribution reuses
    frame_000000, frame_000001... independently, so two different objects'
    rows can carry the identical bare filename. This asserts the exact real
    row shape confirmed against the live dataset, and that numeric condition
    codes are translated to the documented labels (0=bright/dry,
    1=dim/wet, 2=dark), not left as raw digits."""
    csv_text = (
        "File,Weather,Light\n"
        "Battery1\\frame\\frame_000000.PNG,1,1\n"
        "cutter2\\frame\\frame_000094.PNG,0,0\n"
    )
    metadata, warnings = parse_metadata_csv(csv_text)
    assert warnings == []
    assert metadata["Battery1__frame_000000"] == {"light": "dim", "weather": "wet"}
    assert metadata["cutter2__frame_000094"] == {"light": "bright", "weather": "dry"}
    # No literal backslashes should ever survive into a key.
    assert not any("\\" in k for k in metadata)


def test_resolve_labels_dir_matches_data_prep_output_layout():
    """Regression test for a real bug: an earlier version went up two parent
    levels (landing on the dataset root) instead of one (landing on the
    split's own labels/ dir), which silently produced zero ground truth for
    every image instead of raising an error. This asserts the exact
    transformation data_prep.py's --build-split output actually needs."""
    images_dir = Path("/data/fod-a-split/test/images")
    assert resolve_labels_dir(images_dir) == Path("/data/fod-a-split/test/labels")

    train_images_dir = Path("/data/fod-a-split/train/images")
    assert resolve_labels_dir(train_images_dir) == Path("/data/fod-a-split/train/labels")
