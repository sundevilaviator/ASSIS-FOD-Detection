#!/usr/bin/env python3
"""
ASSIS FOD Module — Pascal VOC -> YOLO annotation converter.

FOD-A (Munyer et al. 2021, IEEE ICMLA; arXiv:2110.03072) ships annotations
as one Pascal VOC XML file per image, not YOLO txt. This converts a VOC
annotation directory into YOLO-format labels so src/data_prep.py and
everything downstream of it can work with a single label format.

Design choices, stated up front rather than left implicit:

  - The class list is NOT hardcoded from memory. It's read from
    configs/fod.yaml (single source of truth, shared with training/
    inference/benchmarking) and passed in explicitly. Any XML <object>
    whose <name> isn't in that list is skipped and counted, not silently
    dropped — the conversion summary reports how many objects were skipped
    and which class names caused it, so a class-name typo or an
    out-of-scope object doesn't disappear without a trace.
  - Malformed or unreadable XML files are skipped with a warning, not a
    hard crash on the whole batch — one bad annotation file in a
    multi-thousand-image dataset shouldn't stop the conversion.
  - Coordinates are clamped to the image bounds before normalizing, since
    a small number of public FOD-A-style annotations have bounding boxes
    that run 1-2px outside the frame.
"""
from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ConversionSummary:
    n_xml_files: int = 0
    n_converted: int = 0
    n_skipped_malformed_xml: int = 0
    n_skipped_no_objects: int = 0
    n_skipped_missing_image: int = 0
    n_objects_written: int = 0
    n_objects_skipped_unknown_class: int = 0
    unknown_class_counts: Counter = field(default_factory=Counter)
    malformed_files: list = field(default_factory=list)


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Parse one VOC XML file. Returns (img_width, img_height, objects), where
    each object is (class_name, xmin, ymin, xmax, ymax) in pixel coordinates.
    Raises ET.ParseError / ValueError on malformed input — caller handles it."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path}: missing <size> element")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"{xml_path}: non-positive image dimensions ({img_w}x{img_h})")

    objects = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bbox = obj.find("bndbox")
        if name_el is None or bbox is None:
            continue
        name = name_el.text.strip()
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        # Clamp to image bounds (a small number of public annotations run
        # slightly outside the frame) and skip degenerate zero-area boxes.
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append((name, xmin, ymin, xmax, ymax))

    return img_w, img_h, objects


def voc_box_to_yolo_line(
    class_id: int, xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
) -> str:
    """Convert one pixel-space VOC box to a single normalized YOLO label line."""
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"


def convert_directory(voc_dir: Path, out_labels_dir: Path, class_names: list[str]) -> ConversionSummary:
    class_to_id = {name: i for i, name in enumerate(class_names)}
    summary = ConversionSummary()
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(voc_dir.rglob("*.xml"))
    summary.n_xml_files = len(xml_files)

    for xml_path in xml_files:
        try:
            img_w, img_h, objects = parse_voc_xml(xml_path)
        except (ET.ParseError, ValueError, AttributeError) as e:
            summary.n_skipped_malformed_xml += 1
            summary.malformed_files.append(f"{xml_path.name}: {e}")
            continue

        lines = []
        for name, xmin, ymin, xmax, ymax in objects:
            if name not in class_to_id:
                summary.n_objects_skipped_unknown_class += 1
                summary.unknown_class_counts[name] += 1
                continue
            lines.append(voc_box_to_yolo_line(class_to_id[name], xmin, ymin, xmax, ymax, img_w, img_h))

        if not lines:
            summary.n_skipped_no_objects += 1
            continue

        out_path = out_labels_dir / (xml_path.stem + ".txt")
        out_path.write_text("\n".join(lines) + "\n")
        summary.n_converted += 1
        summary.n_objects_written += len(lines)

    return summary


def convert_original_format_distribution(
    dataset_root: Path, out_images_dir: Path, out_labels_dir: Path, class_names: list[str]
) -> ConversionSummary:
    """Convert FOD-A's per-object-folder "original format" distribution.

    This distribution (e.g. `FullDatasetV.2.1-400x400/`) is laid out very
    differently from a flat Pascal VOC directory: one subfolder per object
    type (`Bolt1/`, `Nut3/`, `cutter2/`, ...), each with its own
    `Annotations/*.xml` + `frame/*.PNG` (or other extension) pair. Critically,
    frame numbering restarts independently inside every object folder —
    `Bolt1/frame/frame_000000.PNG` and `Nut3/frame/frame_000000.PNG` are
    different images — so converting with `convert_directory()` above (which
    names outputs from the XML filename stem alone) would silently collide
    and overwrite labels/images across object folders. This was a real,
    diagnosed defect the first time this dataset distribution was used (run
    4, 2026-09-03); see docs/RESEARCH_LOG.md.

    Every output image/label pair here is instead named
    "{object_folder_name}__{original_filename_stem}" — the same
    collision-safe convention `src/benchmark_faa.py`'s `parse_metadata_csv()`
    already uses for this dataset's categorization CSV, so a benchmark run
    can join the two on that same key.

    Unlike `convert_directory()`, this also copies the matching image file
    (found by filename stem inside the object folder's `frame/` directory,
    whatever its extension actually is) into `out_images_dir`, because this
    distribution's images are not already sitting in a directory
    `src/data_prep.py` can read as-is.
    """
    class_to_id = {name: i for i, name in enumerate(class_names)}
    summary = ConversionSummary()
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    object_folders = sorted(
        p for p in dataset_root.iterdir() if p.is_dir() and (p / "Annotations").is_dir()
    )
    for folder in object_folders:
        ann_dir = folder / "Annotations"
        frame_dir = folder / "frame"
        xml_files = sorted(ann_dir.glob("*.xml"))
        summary.n_xml_files += len(xml_files)

        for xml_path in xml_files:
            try:
                img_w, img_h, objects = parse_voc_xml(xml_path)
            except (ET.ParseError, ValueError, AttributeError) as e:
                summary.n_skipped_malformed_xml += 1
                summary.malformed_files.append(f"{folder.name}/{xml_path.name}: {e}")
                continue

            lines = []
            for name, xmin, ymin, xmax, ymax in objects:
                if name not in class_to_id:
                    summary.n_objects_skipped_unknown_class += 1
                    summary.unknown_class_counts[name] += 1
                    continue
                lines.append(voc_box_to_yolo_line(class_to_id[name], xmin, ymin, xmax, ymax, img_w, img_h))

            if not lines:
                summary.n_skipped_no_objects += 1
                continue

            # Extension is not assumed: this distribution has been observed to
            # use uppercase ".PNG"; look for whatever extension is actually
            # there rather than hardcoding one.
            matches = sorted(frame_dir.glob(xml_path.stem + ".*")) if frame_dir.is_dir() else []
            if not matches:
                summary.n_skipped_missing_image += 1
                summary.malformed_files.append(
                    f"{folder.name}/{xml_path.stem}: no matching image file under {frame_dir}"
                )
                continue
            image_path = matches[0]

            out_stem = f"{folder.name}__{xml_path.stem}"
            (out_labels_dir / (out_stem + ".txt")).write_text("\n".join(lines) + "\n")
            shutil.copy2(image_path, out_images_dir / (out_stem + image_path.suffix))
            summary.n_converted += 1
            summary.n_objects_written += len(lines)

    return summary


def load_class_names(config_path: Path) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text())
    return cfg["classes"]


def list_all_classes(voc_dir: Path) -> Counter:
    """Scan every XML file under voc_dir and count every distinct <name>
    value actually present in the raw dataset. Use this BEFORE deciding what
    to put in configs/fod.yaml's classes: list — don't guess dataset class
    names from memory or documentation; read them directly from the data."""
    counts: Counter = Counter()
    xml_files = sorted(voc_dir.rglob("*.xml"))
    n_malformed = 0
    for xml_path in xml_files:
        try:
            _, _, objects = parse_voc_xml(xml_path)
        except (ET.ParseError, ValueError, AttributeError):
            n_malformed += 1
            continue
        for name, *_ in objects:
            counts[name] += 1
    print(f"Scanned {len(xml_files)} XML files ({n_malformed} malformed/skipped).")
    print(f"Found {len(counts)} distinct class names:\n")
    for name, count in counts.most_common():
        print(f"  {name:30s} {count:6d} objects")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voc-dir", type=Path,
                     help="Directory containing VOC .xml annotation files (searched recursively). "
                          "Use this for a flat Pascal-VOC-style distribution. Mutually exclusive "
                          "with --original-format-root.")
    ap.add_argument("--original-format-root", type=Path,
                     help="Root of FOD-A's per-object-folder 'original format' distribution "
                          "(e.g. FullDatasetV.2.1-400x400/), where each subfolder has its own "
                          "Annotations/ and frame/ directories and frame numbering restarts per "
                          "folder. Writes BOTH images (--out/images) and labels (--out/labels), "
                          "collision-safe-named '{object_folder}__{filename_stem}', unlike "
                          "--voc-dir which only writes labels. Mutually exclusive with --voc-dir.")
    ap.add_argument("--out", type=Path, help="Output directory. Required unless --list-classes-only. "
                                              "With --voc-dir: label .txt files go directly here. "
                                              "With --original-format-root: images/ and labels/ "
                                              "subdirectories are created here.")
    ap.add_argument("--config", type=Path, default=Path("configs/fod.yaml"))
    ap.add_argument(
        "--list-classes-only",
        action="store_true",
        help="Scan the dataset and print every distinct class name actually present, then exit "
             "without converting. Run this FIRST on a new dataset to confirm real class names "
             "before adding them to configs/fod.yaml. Works with --voc-dir only.",
    )
    args = ap.parse_args()

    if args.list_classes_only:
        if args.voc_dir is None:
            ap.error("--list-classes-only requires --voc-dir")
        list_all_classes(args.voc_dir)
        return

    if args.voc_dir is None and args.original_format_root is None:
        ap.error("Specify exactly one of --voc-dir or --original-format-root")
    if args.voc_dir is not None and args.original_format_root is not None:
        ap.error("--voc-dir and --original-format-root are mutually exclusive")
    if args.out is None:
        ap.error("--out is required unless --list-classes-only is set")

    class_names = load_class_names(args.config)
    if args.original_format_root is not None:
        summary = convert_original_format_distribution(
            args.original_format_root, args.out / "images", args.out / "labels", class_names
        )
    else:
        summary = convert_directory(args.voc_dir, args.out, class_names)

    print(f"XML files found:              {summary.n_xml_files}")
    print(f"Converted (label file written): {summary.n_converted}")
    print(f"Skipped — malformed XML:       {summary.n_skipped_malformed_xml}")
    print(f"Skipped — no in-scope objects: {summary.n_skipped_no_objects}")
    print(f"Skipped — no matching image:   {summary.n_skipped_missing_image}")
    print(f"Objects written:               {summary.n_objects_written}")
    print(f"Objects skipped (unknown class): {summary.n_objects_skipped_unknown_class}")
    if summary.unknown_class_counts:
        print("  Unknown class names encountered (not in configs/fod.yaml classes:):")
        for name, count in summary.unknown_class_counts.most_common():
            print(f"    {name}: {count}")
    if summary.malformed_files:
        print("  Malformed/missing-image files (first 10):")
        for m in summary.malformed_files[:10]:
            print(f"    {m}")


if __name__ == "__main__":
    main()
