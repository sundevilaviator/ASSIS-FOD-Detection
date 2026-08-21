#!/usr/bin/env python3
"""
ASSIS FOD Module — training entry point.

Thin wrapper around Ultralytics YOLOv8 so that hyperparameters live in
configs/fod.yaml (one file, checked into git, diffable) instead of being
buried in a notebook cell or a shell one-liner. Every run's exact config is
also copied into the run directory by Ultralytics automatically, so a given
weights file can always be traced back to the settings that produced it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/fod.yaml"))
    ap.add_argument("--data", type=Path, required=True, help="Path to a YOLO data.yaml (see src/data_prep.py).")
    ap.add_argument("--epochs", type=int, default=None, help="Override configs/fod.yaml training.epochs.")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--model", default=None, help="Override base model, e.g. yolov8n.pt for faster CPU iteration.")
    ap.add_argument("--device", default=None, help="e.g. 0 for first GPU, 'cpu' to force CPU.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["training"]

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "ultralytics is not installed. Run `pip install -r requirements.txt` "
            "(or, on Colab, `!pip install ultralytics`)."
        ) from e

    model_name = args.model or cfg["model"]
    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs or cfg["epochs"],
        imgsz=args.imgsz or cfg["imgsz"],
        batch=args.batch or cfg["batch"],
        patience=cfg["patience"],
        optimizer=cfg["optimizer"],
        seed=cfg["seed"],
    )
    # Pass through augmentation settings only if present in the config, so
    # this script doesn't break against an older fod.yaml that predates them.
    for key in ("copy_paste", "hsv_v"):
        if key in cfg:
            train_kwargs[key] = cfg[key]
    if args.device is not None:
        train_kwargs["device"] = args.device

    print("Training with:", train_kwargs)
    results = model.train(**train_kwargs)
    print("Training complete. Best weights:", results.save_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
