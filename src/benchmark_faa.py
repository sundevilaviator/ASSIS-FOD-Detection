#!/usr/bin/env python3
"""
ASSIS FOD Module — FAA AC 150/5220-24-style benchmark.

Reports model performance in the terms FAA Advisory Circular 150/5220-24
uses to evaluate FOD detection equipment, rather than a generic mAP number,
because that is what none of the commercial systems reviewed in the Phase 2
gap analysis publish. Specifically:

  - Detection rate (recall) broken out by object-size bucket, with the
    small-object bucket checked against the AC's >=90% requirement
    (section 3.2.b(1)(c)).
  - False positives per image, as a proxy for the AC's false-alarm-rate
    limit (see the LIMITATIONS section below for exactly what this is and
    isn't measuring).
  - Localization error, reported in pixels / % of frame diagonal — again a
    proxy for the AC's 5-meter requirement (section 3.2.b(2)), because FOD-A (and most public
    FOD imagery) carries no calibrated ground-sample-distance.

This script is intentionally honest about what it can and can't claim. Read
the LIMITATIONS block it prints (and writes into the report) before quoting
any number from this script's output as a real-world FAA-compliance claim.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Resolve the default config against the REPOSITORY, not the working directory.
# `Path("configs/fod.yaml")` is relative to wherever the process happens to be
# started, so every invocation from outside the repo root failed with a bare
# FileNotFoundError naming a path the caller never typed. Three separate call
# sites hit this before the default itself was fixed.
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "fod.yaml"

# Candidate column names to auto-detect in an optional FOD-A-style metadata
# CSV (light-level / weather categorization, which FOD-A ships separately
# from its bounding-box annotations per Munyer et al. 2021). These are best
# guesses at common naming conventions, NOT confirmed against the actual
# released CSV in this session (network access to Kaggle/GitHub raw files
# was not available when this was written) — if none of these match your
# actual file, pass --metadata-image-col / --metadata-light-col /
# --metadata-weather-col explicitly to override.
IMAGE_COL_CANDIDATES = ["filename", "file", "image", "file_path", "filepath", "path", "frame", "image_name"]
LIGHT_COL_CANDIDATES = ["light", "light_level", "lighting", "light_condition"]
WEATHER_COL_CANDIDATES = ["weather", "weather_condition", "weather_level"]

LIMITATIONS = """\
LIMITATIONS (read before citing any number above):

1. Size buckets are a bounding-box PIXEL-AREA proxy for the AC's physical
   centimeter thresholds (AC 150/5220-24 section 3.2.b(1), verified against
   the primary document 2026-08-31). The AC's reference objects are a metal
   cylinder 1.2 in (3.1 cm) high by 1.5 in (3.8 cm) diameter and a 1.7 in
   (4.3 cm) sphere; its 90% group test covers items no larger than 4 in
   (10 cm) in any dimension. Its 90% figure also applies to a SPECIFIED
   GROUP of ten object types in a 100 ft square, not to arbitrary small
   debris - so this bucket test is related to, but not the same as, the
   AC's test.
   Public FOD imagery (including FOD-A) does not carry calibrated
   ground-sample-distance, so pixel area cannot be converted to real-world
   size without knowing camera height, angle, and focal length for each
   shot. Treat "small/medium/large" here as relative-size buckets within
   this dataset, not as verified centimeter classes.

2. False-alarm rate is reported as false positives PER IMAGE, not per day.
   The AC's false-alarm ceiling (section 3.2.b(7)(a): <=1/day visual,
   <=3/day non-visual, each averaged over any 90 day period) is
   defined against a full runway scan cycle at a real airport. Converting
   this script's per-image rate into a per-day claim requires knowing the
   actual deployment's scan cadence (images per patrol / per camera cycle),
   which is deployment-specific and not assumed here.

3. Localization error is reported in pixels and as a percent of the frame
   diagonal, not meters, for the same calibration reason as (1). A 5-meter
   claim requires a calibrated camera setup.

4. This benchmark scores against a held-out split of the SAME source
   dataset the model was trained on (after de-duplication of augmented
   copies). It is not a cross-site validation — see docs/RESEARCH_LOG.md
   for that as a planned, not yet completed, step.

5. If --metadata-csv is supplied, light-level/weather breakdowns are
   matched to images by filename stem. FOD-A's categorization file was
   inspected directly on 2026-08-31: it is
   All_Dataset_Utility_Files/FOD_categorization_annotations.csv in the
   ORIGINAL-format distribution (not the Pascal VOC mirror), 33,863 rows,
   columns File/Weather/Light, with INTEGER codes - Weather 0=Dry 1=Wet,
   Light 0=Bright 1=Dim 2=Dark. That mapping was confirmed two independent
   ways: by matching row counts to the FOD-A paper's Table I, and against
   the dataset's own category_information.txt. Note the light ordering:
   0 is Bright and 2 is Dark, so an assumed ordering would invert the
   result.

   The Pascal VOC mirror CANNOT be joined to this file. It contains 33,793
   images numbered contiguously 000000-033792 with no gaps, meaning 70
   images were dropped and the remainder renumbered; the correspondence to
   the original ordering is unrecoverable from filenames. Stratified
   results therefore require training from the original-format
   distribution. Do not stratify a VOC-trained model against this CSV.

6. If --sahi is set, inference runs as tiled/sliced predictions (SAHI) rather
   than one full-image pass. Because FOD-A's own images are already only
   300x300 (the same fact behind the 2026-08-31 inference-resolution
   sweep's negative result), tiling them creates no new detail and is NOT
   expected to change this benchmark's small-object number. SAHI's
   plausible benefit is on higher-resolution real-world imagery where a
   small object is a tiny fraction of a much larger frame — a case FOD-A's
   own held-out split does not represent. A --sahi run against FOD-A tells
   you whether that reasoning holds; it is not, by itself, evidence about
   real deployment photos one way or the other.

Bottom line: this script produces a consistent, reproducible way to track
progress against the AC's structure over time. It does not, by itself,
constitute a certified FAA compliance test.
"""


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2


def bucket_for_area_pct(area_pct: float, thresholds: dict) -> str:
    if area_pct <= thresholds["small"]["max_area_pct"]:
        return "small"
    if area_pct <= thresholds["medium"]["max_area_pct"]:
        return "medium"
    return "large"


def _find_column(header: list[str], candidates: list[str]) -> str | None:
    lower_map = {h.lower().strip(): h for h in header}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def parse_metadata_csv(
    csv_text: str,
    image_col: str | None = None,
    light_col: str | None = None,
    weather_col: str | None = None,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Parse a light-level/weather metadata CSV into {image_stem: {"light":
    ..., "weather": ...}}. Column names are auto-detected from
    IMAGE_COL_CANDIDATES / LIGHT_COL_CANDIDATES / WEATHER_COL_CANDIDATES
    unless explicitly overridden. Returns (metadata, warnings) — warnings is
    a list of human-readable strings for anything that couldn't be resolved,
    so the caller can print them and continue in degraded mode instead of
    crashing the whole benchmark run over an optional feature.

    Pure function, no file I/O, so it's directly unit-testable against a
    hand-built CSV string.
    """
    warnings: list[str] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    header = reader.fieldnames or []

    resolved_image_col = image_col or _find_column(header, IMAGE_COL_CANDIDATES)
    resolved_light_col = light_col or _find_column(header, LIGHT_COL_CANDIDATES)
    resolved_weather_col = weather_col or _find_column(header, WEATHER_COL_CANDIDATES)

    if resolved_image_col is None:
        warnings.append(
            f"Could not find an image/filename column in {header}. "
            "Pass --metadata-image-col explicitly. Metadata will not be used."
        )
        return {}, warnings

    if resolved_light_col is None and resolved_weather_col is None:
        warnings.append(
            f"Could not find a light-level or weather column in {header}. "
            "Pass --metadata-light-col / --metadata-weather-col explicitly. Metadata will not be used."
        )
        return {}, warnings

    metadata: dict[str, dict[str, str]] = {}
    for row in reader:
        raw_name = row.get(resolved_image_col, "")
        if not raw_name:
            continue
        stem = Path(raw_name.strip()).stem
        entry = {}
        if resolved_light_col and row.get(resolved_light_col):
            entry["light"] = row[resolved_light_col].strip()
        if resolved_weather_col and row.get(resolved_weather_col):
            entry["weather"] = row[resolved_weather_col].strip()
        if entry:
            metadata[stem] = entry

    if not metadata:
        warnings.append("Metadata CSV parsed but produced zero usable rows — check the file contents.")

    return metadata, warnings


def resolve_labels_dir(images_dir: Path) -> Path:
    """Map a split's images/ directory to its sibling labels/ directory, per
    the layout src/data_prep.py's --build-split actually writes:
    <out>/<split>/images and <out>/<split>/labels side by side. That means
    ONE parent level up from images_dir, then into "labels" —
    e.g. .../test/images -> .../test/labels. An earlier version of this
    function used two parent levels (landing on the dataset root instead of
    the split's labels dir) and was only caught by an end-to-end smoke test
    that used data_prep.py's real output layout, not a hand-rolled one —
    this function exists specifically so that exact path transformation is
    now unit-tested directly, matching this project's rule that numeric/path
    transformations get a hand-checked regression test."""
    return images_dir.parent / "labels"


def _bucket_stats_to_results(stats: dict[str, dict[str, int]]) -> dict[str, dict]:
    results = {}
    for key, s in stats.items():
        n = s["tp"] + s["fn"]
        results[key] = {"tp": s["tp"], "fn": s["fn"], "n_ground_truth": n, "detection_rate": (s["tp"] / n) if n else None}
    return results


def load_gt_labels(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        xc, yc, w, h = float(xc), float(yc), float(w), float(h)
        boxes.append(
            {
                "class": int(cls),
                "xyxy": yolo_to_xyxy(xc, yc, w, h, img_w, img_h),
                "area_pct": w * h * 100,
            }
        )
    return boxes


def _predict_boxes_plain(model, img_path: Path, conf: float, imgsz: int | None) -> list[tuple[float, float, float, float]]:
    """Standard single-pass inference — the existing, already-benchmarked path."""
    _kw = {"imgsz": imgsz} if imgsz else {}
    pred = model.predict(source=str(img_path), conf=conf, verbose=False, **_kw)[0]
    return [tuple(v.item() for v in b) for b in pred.boxes.xyxy] if len(pred.boxes) else []


def _predict_boxes_sahi(sahi_model, img_path: Path, slice_size: int, overlap: float) -> list[tuple[float, float, float, float]]:
    """Sliced inference via SAHI: tile the image, run the same weights per
    tile, merge tile-level detections back into full-image coordinates.

    Why this exists: FOD-A's Pascal VOC mirror images are already only
    300x300 (see docs/RESEARCH_LOG.md, 2026-08-31 session 3 — the
    inference-resolution sweep found raising --imgsz does nothing because
    there is no extra detail to recover from an already-small source).
    Tiling a 300x300 source image does not manufacture pixels that were
    never captured either, so this is NOT expected to move the FOD-A
    benchmark number for the same reason. It exists for the case that sweep
    did not test: real, higher-resolution deployment photos where a small
    object occupies a tiny fraction of a large frame. Run this benchmark
    against FOD-A's held-out split to see whether that prediction holds, and
    treat any FOD-A-benchmark improvement (or lack of one) as informative
    either way — this is an experiment, not an assumed win.
    """
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        str(img_path),
        sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=0,
    )
    boxes = []
    for obj in result.object_prediction_list:
        b = obj.bbox
        boxes.append((float(b.minx), float(b.miny), float(b.maxx), float(b.maxy)))
    return boxes


def run_benchmark(
    weights: Path,
    data_yaml: Path,
    config: Path,
    out_dir: Path,
    iou_thresh: float,
    conf: float,
    imgsz: int | None = None,
    metadata_csv: Path | None = None,
    metadata_image_col: str | None = None,
    metadata_light_col: str | None = None,
    metadata_weather_col: str | None = None,
    sahi: bool = False,
    sahi_slice_size: int = 512,
    sahi_overlap: float = 0.2,
) -> dict:
    try:
        from ultralytics import YOLO
        from PIL import Image
    except ImportError as e:
        raise SystemExit("Requires ultralytics and pillow: pip install -r requirements.txt") from e

    cfg = yaml.safe_load(config.read_text())
    thresholds = cfg["size_buckets"]
    faa = cfg["faa_ac_150_5220_24"]

    data_cfg = yaml.safe_load(data_yaml.read_text())
    dataset_root = Path(data_cfg["path"])
    val_rel = data_cfg.get("val", "test/images")
    images_dir = dataset_root / val_rel
    labels_dir = resolve_labels_dir(images_dir)

    metadata: dict[str, dict[str, str]] = {}
    metadata_warnings: list[str] = []
    if metadata_csv is not None:
        metadata, metadata_warnings = parse_metadata_csv(
            metadata_csv.read_text(),
            image_col=metadata_image_col,
            light_col=metadata_light_col,
            weather_col=metadata_weather_col,
        )
        for w in metadata_warnings:
            print(f"[metadata] WARNING: {w}")
        if metadata:
            print(f"[metadata] Loaded light/weather labels for {len(metadata)} images from {metadata_csv}")

    model = YOLO(str(weights))

    sahi_model = None
    if sahi:
        try:
            from sahi import AutoDetectionModel
        except ImportError as e:
            raise SystemExit(
                "Requires sahi: pip install sahi. --sahi was passed but the "
                "package is not installed."
            ) from e
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(weights),
            confidence_threshold=conf,
            device=device,
        )
        print(
            f"[sahi] Sliced inference enabled: slice={sahi_slice_size}x{sahi_slice_size}, "
            f"overlap={sahi_overlap}, device={device}. Expected to matter most on images "
            "larger than FOD-A's native 300x300 — see docstring on _predict_boxes_sahi."
        )

    stats = {b: {"tp": 0, "fn": 0} for b in ("small", "medium", "large")}
    light_stats: dict[str, dict[str, int]] = {}
    weather_stats: dict[str, dict[str, int]] = {}
    total_false_positives = 0
    total_images = 0
    loc_errors_pct_diag: list[float] = []

    image_paths = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    for img_path in image_paths:
        total_images += 1
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        diag = math.hypot(img_w, img_h)

        img_meta = metadata.get(img_path.stem, {})
        light_key = img_meta.get("light")
        weather_key = img_meta.get("weather")

        gt_boxes = load_gt_labels(labels_dir / (img_path.stem + ".txt"), img_w, img_h)
        # imgsz is passed explicitly when set. Inference resolution materially
        # changes small-object recall, so a benchmark that does not state the
        # resolution it ran at cannot be compared against another one.
        if sahi_model is not None:
            pred_boxes = _predict_boxes_sahi(sahi_model, img_path, sahi_slice_size, sahi_overlap)
        else:
            pred_boxes = _predict_boxes_plain(model, img_path, conf, imgsz)

        matched_pred = set()
        for gt in gt_boxes:
            best_iou, best_idx = 0.0, -1
            for i, pb in enumerate(pred_boxes):
                if i in matched_pred:
                    continue
                cur = iou(gt["xyxy"], pb)
                if cur > best_iou:
                    best_iou, best_idx = cur, i
            bucket = bucket_for_area_pct(gt["area_pct"], thresholds)
            is_tp = best_iou >= iou_thresh and best_idx >= 0
            outcome_key = "tp" if is_tp else "fn"
            stats[bucket][outcome_key] += 1
            if light_key:
                light_stats.setdefault(light_key, {"tp": 0, "fn": 0})[outcome_key] += 1
            if weather_key:
                weather_stats.setdefault(weather_key, {"tp": 0, "fn": 0})[outcome_key] += 1
            if is_tp:
                matched_pred.add(best_idx)
                gx1, gy1, gx2, gy2 = gt["xyxy"]
                px1, py1, px2, py2 = pred_boxes[best_idx]
                gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
                err = math.hypot(gcx - pcx, gcy - pcy)
                loc_errors_pct_diag.append(err / diag * 100)

        total_false_positives += max(0, len(pred_boxes) - len(matched_pred))

    results = _bucket_stats_to_results(stats)
    results_by_light = _bucket_stats_to_results(light_stats) if light_stats else None
    results_by_weather = _bucket_stats_to_results(weather_stats) if weather_stats else None

    fp_per_image = total_false_positives / total_images if total_images else None
    mean_loc_err = sum(loc_errors_pct_diag) / len(loc_errors_pct_diag) if loc_errors_pct_diag else None

    small_pass = (
        results["small"]["detection_rate"] is not None
        and results["small"]["detection_rate"] >= faa["min_detection_rate_small_object"]
    )

    report = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "data": str(data_yaml),
        "iou_threshold": iou_thresh,
        "confidence_threshold": conf,
        "inference_imgsz": imgsz,  # None = Ultralytics default (640)
        "sahi_sliced_inference": sahi,
        "sahi_slice_size": sahi_slice_size if sahi else None,
        "sahi_overlap": sahi_overlap if sahi else None,
        "n_images_evaluated": total_images,
        "results_by_size_bucket": results,
        "results_by_light_level": results_by_light,
        "results_by_weather": results_by_weather,
        "metadata_source": str(metadata_csv) if metadata_csv else None,
        "metadata_warnings": metadata_warnings,
        "false_positives_per_image": fp_per_image,
        "mean_localization_error_pct_frame_diagonal": mean_loc_err,
        "faa_reference_thresholds": faa,
        "small_object_meets_faa_90pct_threshold": small_pass,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"benchmark_{stamp}.json"
    md_path = out_dir / f"benchmark_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_markdown(report))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {json_path}\nWrote {md_path}")
    print("\n" + LIMITATIONS)
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# ASSIS FOD Module — FAA AC 150/5220-24-Style Benchmark",
        "",
        f"Run: {report['run_timestamp_utc']}  ",
        f"Weights: `{report['weights']}`  ",
        f"Data: `{report['data']}`  ",
        f"Images evaluated: {report['n_images_evaluated']}",
        "",
        "## Detection rate by size bucket",
        "",
        "| Bucket | Ground-truth objects | Detected (TP) | Missed (FN) | Detection rate | AC 90% threshold (small only) |",
        "|---|---|---|---|---|---|",
    ]
    for bucket in ("small", "medium", "large"):
        r = report["results_by_size_bucket"][bucket]
        rate = f"{r['detection_rate']:.1%}" if r["detection_rate"] is not None else "n/a (no ground truth)"
        flag = ""
        if bucket == "small" and r["detection_rate"] is not None:
            flag = "PASS" if report["small_object_meets_faa_90pct_threshold"] else "FAIL"
        lines.append(f"| {bucket} | {r['n_ground_truth']} | {r['tp']} | {r['fn']} | {rate} | {flag} |")

    def stratified_table(title: str, results_by_key: dict | None) -> list[str]:
        if not results_by_key:
            return [f"## {title}", "", "Not available — no metadata CSV supplied or no matching column found.", ""]
        out = [
            f"## {title}",
            "",
            "This breakdown is what none of the vendors/papers reviewed in the Phase 2 gap "
            "analysis publish — see LIMITATIONS item 5 for how it was matched.",
            "",
            "| Condition | Ground-truth objects | Detected (TP) | Missed (FN) | Detection rate |",
            "|---|---|---|---|---|",
        ]
        for key in sorted(results_by_key):
            r = results_by_key[key]
            rate = f"{r['detection_rate']:.1%}" if r["detection_rate"] is not None else "n/a"
            out.append(f"| {key} | {r['n_ground_truth']} | {r['tp']} | {r['fn']} | {rate} |")
        out.append("")
        return out

    fp = report["false_positives_per_image"]
    loc = report["mean_localization_error_pct_frame_diagonal"]
    lines += [
        "",
        "## False alarms and localization",
        "",
        f"- False positives per image (proxy metric, see limitations): "
        f"{fp:.3f}" if fp is not None else "- False positives per image: n/a",
        f"- Mean localization error: {loc:.2f}% of frame diagonal (proxy metric, see limitations)"
        if loc is not None else "- Mean localization error: n/a",
        "",
    ]
    lines += stratified_table("Detection rate by lighting condition", report.get("results_by_light_level"))
    lines += stratified_table("Detection rate by weather condition", report.get("results_by_weather"))
    if report.get("metadata_warnings"):
        lines += ["## Metadata warnings", ""]
        lines += [f"- {w}" for w in report["metadata_warnings"]]
        lines += [""]
    lines += [
        "## " + LIMITATIONS.splitlines()[0],
        "",
    ]
    lines += LIMITATIONS.splitlines()[1:]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="data.yaml from src/data_prep.py --build-split")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=Path("docs/benchmark_results"))
    ap.add_argument("--iou-thresh", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="Inference resolution. Raising it is the cheapest lever on "
                         "small-object recall and needs no retraining. Recorded in the "
                         "output so results at different resolutions are never confused.")
    ap.add_argument(
        "--metadata-csv", type=Path, default=None,
        help="Optional FOD-A-style light-level/weather metadata CSV, joined by filename stem. "
             "See LIMITATIONS item 5 in this script's docstring before trusting the result.",
    )
    ap.add_argument("--metadata-image-col", default=None, help="Override auto-detected image/filename column name.")
    ap.add_argument("--metadata-light-col", default=None, help="Override auto-detected light-level column name.")
    ap.add_argument("--metadata-weather-col", default=None, help="Override auto-detected weather column name.")
    ap.add_argument(
        "--sahi", action="store_true",
        help="Use SAHI sliced inference instead of a single full-image pass: tile the "
             "image, run the same weights per tile, merge results back to full-image "
             "coordinates. Requires: pip install sahi. NOT expected to change results on "
             "FOD-A's own 300x300 images (see _predict_boxes_sahi docstring) — this flag "
             "exists to test that prediction and to benchmark higher-resolution imagery "
             "where it's more likely to matter. Mutually exclusive in effect with --imgsz "
             "(imgsz is ignored when --sahi is set; SAHI controls its own tile size).",
    )
    ap.add_argument("--sahi-slice-size", type=int, default=512, help="Tile height/width in pixels for --sahi.")
    ap.add_argument("--sahi-overlap", type=float, default=0.2, help="Tile overlap ratio for --sahi (0-1).")
    args = ap.parse_args()

    run_benchmark(
        args.weights, args.data, args.config, args.out, args.iou_thresh, args.conf,
        args.imgsz,
        metadata_csv=args.metadata_csv,
        metadata_image_col=args.metadata_image_col,
        metadata_light_col=args.metadata_light_col,
        metadata_weather_col=args.metadata_weather_col,
        sahi=args.sahi,
        sahi_slice_size=args.sahi_slice_size,
        sahi_overlap=args.sahi_overlap,
    )


if __name__ == "__main__":
    main()
