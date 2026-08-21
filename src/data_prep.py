#!/usr/bin/env python3
"""
ASSIS FOD Module — dataset preparation.

Two independent operations, run separately so each step is inspectable:

  1. --download   Pull a Kaggle dataset (default: FOD-A) to a local folder.
  2. --build-split  Turn a YOLO-format dataset (images/ + labels/) into a
                     small-object-weighted training split plus a
                     size-stratified held-out test split, and emit a
                     data.yaml Ultralytics can train against directly.

The small-object weighting is done by oversampling (duplicating, with light
augmentation) images that contain at least one "small" bounding box per
configs/fod.yaml, rather than by touching the YOLO loss function — this is
the simplest change that measurably shifts what the model sees during
training, and it's easy to audit (you can literally count the duplicated
files). See README.md "known limitation" for why bbox area is a pixel-area
proxy, not a physical centimeter measurement.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image, ImageEnhance

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_kaggle_dataset(dataset: str, out_dir: Path) -> Path:
    """Download a Kaggle dataset using kagglehub, falling back to a clear
    error message with manual instructions if kagglehub / credentials aren't
    available (common in CI or a fresh Colab session)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import kagglehub  # imported lazily so this module has no hard
                           # dependency for users who already have the data
    except ImportError as e:
        raise SystemExit(
            "kagglehub is not installed. Run `pip install kagglehub` or "
            "download the dataset manually from "
            f"https://www.kaggle.com/datasets/{dataset} and point "
            "--build-split at it directly."
        ) from e

    try:
        path = kagglehub.dataset_download(dataset)
    except Exception as e:  # noqa: BLE001 - surface a usable message
        raise SystemExit(
            f"kagglehub download failed for '{dataset}': {e}\n"
            "Make sure ~/.kaggle/kaggle.json exists (Kaggle account -> "
            "Settings -> Create New Token) or set the KAGGLE_USERNAME / "
            "KAGGLE_KEY environment variables."
        ) from e

    src = Path(path)
    print(f"Downloaded to cache at {src}; copying into {out_dir} ...")
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{out_dir} already has content — leaving it as-is.")
    else:
        shutil.copytree(src, out_dir, dirs_exist_ok=True)
    return out_dir


# --------------------------------------------------------------------------- #
# Split building
# --------------------------------------------------------------------------- #

@dataclass
class LabeledImage:
    image_path: Path
    label_path: Path
    boxes: list[tuple[int, float, float, float, float]]  # cls, xc, yc, w, h (normalized)

    def max_box_area_pct(self) -> float:
        if not self.boxes:
            return 0.0
        return max(w * h * 100 for _, _, _, w, h in self.boxes)

    def size_bucket(self, thresholds: dict) -> str:
        area = self.max_box_area_pct()
        if area <= thresholds["small"]["max_area_pct"]:
            return "small"
        if area <= thresholds["medium"]["max_area_pct"]:
            return "medium"
        return "large"


def find_labeled_images(root: Path) -> list[LabeledImage]:
    """Locate image/label pairs under a YOLO-format dataset root. Tries the
    conventional images/ + labels/ layout first; falls back to same-directory
    pairing (image.jpg + image.txt) since public FOD datasets are not always
    laid out consistently."""
    items: list[LabeledImage] = []

    images_dir = root / "images"
    labels_dir = root / "labels"
    if images_dir.exists() and labels_dir.exists():
        candidates = [
            (p, labels_dir / (p.stem + ".txt"))
            for p in images_dir.rglob("*")
            if p.suffix.lower() in IMG_EXTS
        ]
    else:
        candidates = [
            (p, p.with_suffix(".txt"))
            for p in root.rglob("*")
            if p.suffix.lower() in IMG_EXTS
        ]

    for img_path, label_path in candidates:
        if not label_path.exists():
            continue
        boxes = []
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
        items.append(LabeledImage(img_path, label_path, boxes))

    return items


def augment_copy(src_img: Path, dst_img: Path, seed: int) -> None:
    """Cheap, label-preserving augmentation (brightness/contrast jitter only —
    anything geometric would require rewriting the label file, which is
    intentionally out of scope for this lightweight oversampling step)."""
    rng = random.Random(seed)
    im = Image.open(src_img).convert("RGB")
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.75, 1.25))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.8, 1.2))
    im.save(dst_img, quality=95)


def build_split(
    source: Path,
    out: Path,
    small_object_max_area_pct: float,
    test_frac: float,
    oversample_factor: int,
    class_names: list[str],
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    items = find_labeled_images(source)
    if not items:
        raise SystemExit(
            f"No labeled images found under {source}. Expected a YOLO-format "
            "layout (images/ + labels/, or paired image+.txt files)."
        )

    thresholds = {
        "small": {"max_area_pct": small_object_max_area_pct},
        "medium": {"max_area_pct": max(small_object_max_area_pct * 4, 2.0)},
        "large": {"max_area_pct": None},
    }

    by_bucket: dict[str, list[LabeledImage]] = {"small": [], "medium": [], "large": []}
    for it in items:
        by_bucket[it.size_bucket(thresholds)].append(it)

    print("Source dataset composition by max-box-size bucket:")
    for bucket, imgs in by_bucket.items():
        print(f"  {bucket:6s}: {len(imgs):5d} images")

    # Size-stratified test split: pull test_frac from EACH bucket so the held-out
    # set isn't accidentally skewed toward whichever bucket is most common —
    # this matters because it's exactly the set the FAA benchmark script scores.
    test_items: list[LabeledImage] = []
    train_items: list[LabeledImage] = []
    for bucket, imgs in by_bucket.items():
        imgs = imgs[:]
        rng.shuffle(imgs)
        n_test = max(1, int(len(imgs) * test_frac)) if imgs else 0
        test_items.extend(imgs[:n_test])
        train_items.extend(imgs[n_test:])

    # Oversample small-object training images (with light augmentation so
    # duplicates aren't byte-identical).
    train_out_img = out / "train" / "images"
    train_out_lbl = out / "train" / "labels"
    test_out_img = out / "test" / "images"
    test_out_lbl = out / "test" / "labels"
    for d in (train_out_img, train_out_lbl, test_out_img, test_out_lbl):
        d.mkdir(parents=True, exist_ok=True)

    def copy_pair(item: LabeledImage, img_dir: Path, lbl_dir: Path, suffix: str = "") -> None:
        stem = item.image_path.stem + suffix
        shutil.copy2(item.label_path, lbl_dir / (stem + ".txt"))
        shutil.copy2(item.image_path, img_dir / (stem + item.image_path.suffix))

    small_count = medium_count = large_count = 0
    for item in train_items:
        copy_pair(item, train_out_img, train_out_lbl)
        bucket = item.size_bucket(thresholds)
        if bucket == "small":
            small_count += 1
            for k in range(oversample_factor):
                dup_stem = f"{item.image_path.stem}_aug{k}"
                dup_img = train_out_img / (dup_stem + item.image_path.suffix)
                shutil.copy2(item.label_path, train_out_lbl / (dup_stem + ".txt"))
                augment_copy(item.image_path, dup_img, seed=seed + k)
        elif bucket == "medium":
            medium_count += 1
        else:
            large_count += 1

    for item in test_items:
        copy_pair(item, test_out_img, test_out_lbl)

    data_yaml = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "test/images",
        "test": "test/images",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))

    manifest = {
        "source": str(source),
        "n_source_images": len(items),
        "n_train_base": len(train_items),
        "n_train_small_oversampled": small_count * oversample_factor,
        "n_test": len(test_items),
        "test_bucket_counts": {b: len(v) for b, v in by_bucket.items()},
        "oversample_factor": oversample_factor,
        "small_object_max_area_pct": small_object_max_area_pct,
        "seed": seed,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote split to {out}")
    print(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def load_class_names(config_path: Path) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text())
    return cfg["classes"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/fod.yaml"))

    ap.add_argument("--download", action="store_true", help="Download a Kaggle dataset.")
    ap.add_argument("--dataset", default="kilogrand/foreign-object-debris-in-airports-fod-a-dataset")
    ap.add_argument("--out", type=Path, required=True)

    ap.add_argument("--build-split", action="store_true", help="Build the small-object-weighted split.")
    ap.add_argument("--source", type=Path, help="Path to a YOLO-format dataset (required with --build-split).")
    ap.add_argument("--small-object-max-area-pct", type=float, default=0.5)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--oversample-factor", type=int, default=3,
                     help="Number of augmented duplicates to add per small-object training image.")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if not args.download and not args.build_split:
        ap.error("Specify --download and/or --build-split.")

    if args.download:
        download_kaggle_dataset(args.dataset, args.out)

    if args.build_split:
        if args.source is None:
            ap.error("--build-split requires --source")
        class_names = load_class_names(args.config)
        build_split(
            source=args.source,
            out=args.out,
            small_object_max_area_pct=args.small_object_max_area_pct,
            test_frac=args.test_frac,
            oversample_factor=args.oversample_factor,
            class_names=class_names,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
