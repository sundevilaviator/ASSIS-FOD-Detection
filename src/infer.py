#!/usr/bin/env python3
"""
ASSIS FOD Module — batch inference.

Runs a trained model over a folder of images (or a single video), writes
annotated images/video, and — importantly for the SMS-integration goal
described in the ASSIS Technical Report — a structured JSON detection log
(one record per detection: time, source file, camera_id, class, confidence,
bbox, size_bucket) rather than just pictures. That JSON is what a real
deployment would feed into a reporting/dashboard layer; the Streamlit app in
app/streamlit_app.py reads the same format for its "alert" panel.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def size_bucket_for(area_pct: float, thresholds: dict) -> str:
    if area_pct <= thresholds["small"]["max_area_pct"]:
        return "small"
    if area_pct <= thresholds["medium"]["max_area_pct"]:
        return "medium"
    return "large"


def run_inference(
    weights: Path,
    source: Path,
    out_dir: Path,
    config: Path,
    camera_id: str,
    conf: float,
) -> list[dict]:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "ultralytics is not installed. Run `pip install -r requirements.txt`."
        ) from e

    cfg = yaml.safe_load(config.read_text())
    thresholds = cfg["size_buckets"]
    class_names = cfg["classes"]

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    is_video = source.is_file() and source.suffix.lower() in VIDEO_EXTS
    predict_source = str(source)

    results = model.predict(
        source=predict_source,
        conf=conf,
        save=True,
        project=str(out_dir),
        name="predict",
        exist_ok=True,
        stream=is_video,
    )

    detections: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    for result in results:
        img_w, img_h = result.orig_shape[1], result.orig_shape[0]
        src_name = Path(result.path).name if result.path else "video_frame"
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf_val = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            w_pct = (x2 - x1) / img_w
            h_pct = (y2 - y1) / img_h
            area_pct = w_pct * h_pct * 100
            detections.append(
                {
                    "timestamp_utc": now,
                    "camera_id": camera_id,
                    "source_file": src_name,
                    "class": class_names[cls_id] if cls_id < len(class_names) else str(cls_id),
                    "confidence": round(conf_val, 4),
                    "bbox_xyxy_px": [round(v, 1) for v in (x1, y1, x2, y2)],
                    "bbox_area_pct_of_frame": round(area_pct, 4),
                    "size_bucket": size_bucket_for(area_pct, thresholds),
                }
            )

    log_path = out_dir / "detections.json"
    log_path.write_text(json.dumps(detections, indent=2))
    print(f"Wrote {len(detections)} detections to {log_path}")
    print(f"Annotated output saved under {out_dir / 'predict'}")
    return detections


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="Image folder, single image, or video file.")
    ap.add_argument("--out", type=Path, default=Path("runs/infer"))
    ap.add_argument("--config", type=Path, default=Path("configs/fod.yaml"))
    ap.add_argument("--camera-id", default="CAM-UNSPECIFIED")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    run_inference(args.weights, args.source, args.out, args.config, args.camera_id, args.conf)


if __name__ == "__main__":
    main()
