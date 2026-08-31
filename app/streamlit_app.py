#!/usr/bin/env python3
"""
ASSIS FOD Module — local demo app.

A small Streamlit UI to actually look at what the model does, rather than
just reading numbers off a benchmark report. It produces the same structured
record (time, camera ID, classification, confidence, size bucket) described
in the ASSIS Technical Report as the format that feeds into SMS/incident
reporting — so this app is a stand-in for that integration point, not a
finished ops tool.

Run with:
    streamlit run app/streamlit_app.py -- --weights runs/detect/train/weights/best.pt

(Everything after the bare `--` is passed through to this script's own
argparse; Streamlit swallows its own flags before that point.)
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fod.yaml"

# Trained weights are NOT committed to git (see .gitignore — a 20MB+ binary
# does not belong in source history). For a hosted deployment where there is
# no local weights file, the app fetches them once from a published release
# asset. Set this to the release-asset URL, or override at runtime with the
# ASSIS_FOD_WEIGHTS_URL environment variable / Streamlit secret.
DEFAULT_WEIGHTS_URL = ""
WEIGHTS_CACHE_DIR = Path(
    os.environ.get("ASSIS_FOD_WEIGHTS_CACHE", Path.home() / ".cache" / "assis-fod")
)


def configured_weights_url() -> str:
    """Resolve the weights URL from (in order) env var, Streamlit secret, default.

    st.secrets raises rather than returning empty when no secrets file exists
    at all, which is the normal case for a local run — so that lookup is
    guarded rather than assumed to succeed.
    """
    env_url = os.environ.get("ASSIS_FOD_WEIGHTS_URL", "").strip()
    if env_url:
        return env_url
    try:
        secret_url = str(st.secrets.get("weights_url", "")).strip()
        if secret_url:
            return secret_url
    except Exception:  # noqa: BLE001 - no secrets configured; expected locally
        pass
    return DEFAULT_WEIGHTS_URL.strip()


def download_weights(url: str, dest: Path) -> Path:
    """Download weights to `dest` if not already cached. Returns the path.

    Downloads to a temporary sibling file first and renames on success, so an
    interrupted download can never leave a truncated file that later looks
    like a valid cache hit.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - URL is operator-configured
    tmp.replace(dest)
    return dest


def parse_args() -> argparse.Namespace:
    # Streamlit passes everything after `--` to sys.argv as-is.
    argv = sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return ap.parse_args(argv)


@st.cache_resource
def load_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


@st.cache_data
def load_config(config_path: str) -> dict:
    return yaml.safe_load(Path(config_path).read_text())


def bucket_for_area_pct(area_pct: float, thresholds: dict) -> str:
    if area_pct <= thresholds["small"]["max_area_pct"]:
        return "small"
    if area_pct <= thresholds["medium"]["max_area_pct"]:
        return "medium"
    return "large"


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    colors = {"small": (255, 80, 80), "medium": (255, 180, 40), "large": (80, 160, 255)}
    for d in detections:
        x1, y1, x2, y2 = d["bbox_xyxy_px"]
        color = colors.get(d["size_bucket"], (200, 200, 200))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{d['class']} {d['confidence']:.2f} ({d['size_bucket']})"
        draw.rectangle([x1, max(0, y1 - 16), x1 + 8 * len(label), y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 15)), label, fill=(0, 0, 0))
    return annotated


# Measured performance, kept as data rather than prose so that these figures
# have exactly one definition in the app and can be checked against
# docs/RESEARCH_LOG.md. Pooled over two independent 100-epoch runs
# (2026-08-20, 2026-08-23), each benchmarked on its own held-out split.
# See docs/benchmark_results/. Do NOT quote the individual runs' small-object
# figures (52.2% / 47.8%) separately: they differ by 0.42 standard errors and
# a single run's 95% interval is roughly +/-14 points.
MEASURED_PERFORMANCE = (
    # (size bucket, pooled detection rate, 95% CI or run range, n)
    ("Large", "99.8-100%", "range across two runs", 525),
    ("Medium", "98.6-99.5%", "range across two runs", 222),
    ("Small", "50.0%", "95% CI 40.0-60.0%", 92),
)

FAA_SMALL_OBJECT_THRESHOLD_NOTE = (
    "FAA AC 150/5220-24 references a 90% detection threshold for small objects. "
    "The entire confidence interval for small-object detection above sits below "
    "that threshold. This shortfall is reported rather than omitted: small-object "
    "detection is the unsolved part of this problem, and stating where a federal "
    "standard is not met is more useful than reporting only favourable figures."
)


def _render_measured_performance() -> None:
    """Show measured benchmark results up front, including the bad one.

    This lives in the demo because the demo is the part strangers actually
    open. Someone who uploads a photo of a large object will see a confident
    detection and may reasonably over-infer from it; the small-object result
    is the number that bounds what this model can currently be trusted to do.
    """
    with st.expander("Measured performance on FOD-A (read before interpreting results)", expanded=False):
        st.markdown(
            "| Object size | Detection rate | Uncertainty | Ground-truth instances |\n"
            "|---|---|---|---|\n"
            + "\n".join(
                f"| {bucket} | {rate} | {interval} | {n} |"
                for bucket, rate, interval, n in MEASURED_PERFORMANCE
            )
        )
        st.warning(FAA_SMALL_OBJECT_THRESHOLD_NOTE)
        st.markdown(
            "**Limitations.** FOD-A images carry no calibrated camera geometry, so "
            "\"size\" here is a bounding-box-area proxy, not a measured centimetre "
            "size. Results are from a held-out split of one public dataset — not "
            "cross-site, not validated at an operating airport, and not evaluated "
            "against tire-fragment or rubber debris, which FOD-A does not cover. "
            "Full detail and dated history: `docs/RESEARCH_LOG.md`."
        )


def main() -> None:
    args = parse_args()

    st.set_page_config(page_title="ASSIS — FOD Detection Demo", layout="wide")
    st.title("ASSIS — Foreign Object Debris (FOD) Detection")
    st.caption(
        "Phase 2 of the AI-Integrated Airport Safety and Security Intelligence "
        "System (ASSIS). Research demo, not a certified or deployed product. "
        "Human-in-the-loop: detections are candidates for operator review, not "
        "automated alerts to a physical system."
    )

    _render_measured_performance()

    cfg_path = st.sidebar.text_input("Config path", str(args.config))
    weights_path = st.sidebar.text_input("Weights path", str(args.weights) if args.weights else "")
    camera_id = st.sidebar.text_input("Camera / source ID", "CAM-DEMO-01")
    conf_thresh = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)

    if not weights_path:
        url = configured_weights_url()
        if url:
            cached = WEIGHTS_CACHE_DIR / "best.pt"
            try:
                with st.spinner("Fetching trained weights (first run only)…"):
                    weights_path = str(download_weights(url, cached))
            except Exception as e:  # noqa: BLE001
                st.error(
                    f"Could not download weights from the configured URL: {e}\n\n"
                    "Enter a local weights path in the sidebar instead."
                )
                st.stop()
        else:
            st.info(
                "No weights loaded yet. Train a model first (`src/train.py`), then pass "
                "`--weights path/to/best.pt` when launching this app, or enter the path "
                "in the sidebar. For a hosted deployment, publish the weights as a "
                "release asset and set `ASSIS_FOD_WEIGHTS_URL` (or a `weights_url` "
                "Streamlit secret) so the app can fetch them automatically."
            )
            st.stop()

    cfg = load_config(cfg_path)
    thresholds = cfg["size_buckets"]

    try:
        model = load_model(weights_path)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load weights from {weights_path}: {e}")
        st.stop()

    uploaded = st.file_uploader("Upload a runway/taxiway/apron image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        st.stop()

    image = Image.open(uploaded)
    result = model.predict(source=image, conf=conf_thresh, verbose=False)[0]

    img_w, img_h = result.orig_shape[1], result.orig_shape[0]
    now = datetime.now(timezone.utc).isoformat()
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf_val = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        area_pct = (x2 - x1) / img_w * (y2 - y1) / img_h * 100
        detections.append(
            {
                "timestamp_utc": now,
                "camera_id": camera_id,
                "class": cfg["classes"][cls_id] if cls_id < len(cfg["classes"]) else str(cls_id),
                "confidence": round(conf_val, 4),
                "bbox_xyxy_px": [round(v, 1) for v in (x1, y1, x2, y2)],
                "size_bucket": bucket_for_area_pct(area_pct, thresholds),
            }
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Annotated frame")
        st.image(draw_detections(image, detections), use_container_width=True)

    with col2:
        st.subheader(f"Structured alert log ({len(detections)} detection{'s' if len(detections) != 1 else ''})")
        if detections:
            st.dataframe(
                [
                    {
                        "Class": d["class"],
                        "Size": d["size_bucket"],
                        "Confidence": d["confidence"],
                        "Camera": d["camera_id"],
                        "Time (UTC)": d["timestamp_utc"],
                    }
                    for d in detections
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "Download detections as JSON",
                data=__import__("json").dumps(detections, indent=2),
                file_name="fod_detections.json",
                mime="application/json",
            )
        else:
            st.write("No detections above the confidence threshold.")


if __name__ == "__main__":
    main()
