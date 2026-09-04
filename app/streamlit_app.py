#!/usr/bin/env python3
"""
ASSIS FOD Module — research demonstration app.

A multi-page Streamlit UI that walks a reviewer through the research story
this repository actually supports: run the detector on an image, see how it
was benchmarked against FAA AC 150/5220-24-style metrics, see how it holds
up across the FOD-A dataset's own light/weather conditions, and read the
methodology and known limitations. It produces the same structured record
(time, camera ID, classification, confidence, size bucket) described in the
ASSIS Technical Report as the format that feeds into SMS/incident
reporting — so this app is a stand-in for that integration point, not a
finished ops tool.

Every metric shown here is read from the committed benchmark reports in
docs/benchmark_results/ via src/benchmark_report.py — nothing is
hard-coded, so a stale figure cannot silently survive a re-run. Where the
underlying data or feature does not exist yet, the page says so explicitly
rather than filling the gap with a plausible-looking placeholder (see
docs/RESEARCH_LOG.md and the project's UX/UI specification for why this
matters).

Run with:
    streamlit run app/streamlit_app.py -- --weights runs/detect/train/weights/best.pt

(Everything after the bare `--` is passed through to this script's own
argparse; Streamlit swallows its own flags before that point.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fod.yaml"
BENCHMARK_RESULTS_DIR = REPO_ROOT / "docs" / "benchmark_results"
RESEARCH_LOG_PATH = REPO_ROOT / "docs" / "RESEARCH_LOG.md"

# src/ has no __init__.py (see tests/test_benchmark_faa.py for the same
# pattern) — it is imported as a plain sys.path addition, not a package.
sys.path.insert(0, str(REPO_ROOT / "src"))
from benchmark_report import (  # noqa: E402
    has_environmental_stratification,
    infer_run_label,
    load_benchmark_reports,
    metadata_accounting,
    select_latest_verified_run,
)

# Trained weights are NOT committed to git (see .gitignore — a 20MB+ binary
# does not belong in source history). For a hosted deployment where there is
# no local weights file, the app fetches them once from a published release
# asset. Set this to the release-asset URL, or override at runtime with the
# ASSIS_FOD_WEIGHTS_URL environment variable / Streamlit secret.
DEFAULT_WEIGHTS_URL = ""
WEIGHTS_CACHE_DIR = Path(
    os.environ.get("ASSIS_FOD_WEIGHTS_CACHE", Path.home() / ".cache" / "assis-fod")
)

# Palette from the project's UX/UI specification. Kept in one place so page
# functions never hand-pick colors ad hoc.
COLORS = {
    "navy": "#0B1F33",
    "blue": "#1677C8",
    "cyan": "#00A6D6",
    "green": "#2E9D62",
    "amber": "#F2A900",
    "red": "#D64545",
    "background": "#F4F7FA",
    "white": "#FFFFFF",
    "text": "#17202A",
    "muted": "#64748B",
}

NAV_PAGES = [
    "Dashboard",
    "Image Detection",
    "Benchmark Performance",
    "Environmental Conditions",
    "Methodology",
    "Limitations & Roadmap",
]


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


@st.cache_data
def load_reports() -> list[dict]:
    return load_benchmark_reports(BENCHMARK_RESULTS_DIR)


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


def _fmt_pct(x: float | None, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%" if x is not None else "—"


def _research_prototype_badge() -> None:
    st.markdown(
        f"<span style='background:{COLORS['amber']};color:{COLORS['navy']};"
        "padding:2px 10px;border-radius:3px;font-weight:600;font-size:0.8rem;'>"
        "RESEARCH PROTOTYPE</span>",
        unsafe_allow_html=True,
    )


def _page_header() -> None:
    st.title("ASSIS — Foreign Object Debris (FOD) Detection")
    st.caption("Camera-Based FOD Detection & Validation")
    _research_prototype_badge()
    st.caption(
        "Phase 2 of the AI-Integrated Airport Safety and Security Intelligence "
        "System (ASSIS). A research prototype investigating whether computer "
        "vision can perform useful FOD detection using conventional camera "
        "imagery and existing airport infrastructure — not a certified, "
        "operationally deployed, or airport-tested system. Human-in-the-loop: "
        "detections are candidates for operator review, not automated alerts "
        "to a physical system."
    )


def _measurement_scope_block() -> None:
    with st.expander("Measurement scope — read before interpreting any result", expanded=False):
        st.markdown(
            "- Object-size buckets (small/medium/large) are based on **bounding-box "
            "pixel area**, a proxy — not calibrated physical (centimeter) "
            "dimensions.\n"
            "- False positives are reported **per image**, not per day; no "
            "deployment-specific scan cadence exists to convert between the two.\n"
            "- Localization error is expressed **relative to frame diagonal**, "
            "not in meters — FOD-A carries no calibrated ground-sample distance.\n"
            "- Results apply to the specific dataset and evaluation split used "
            "and should not be generalized to airport-wide performance.\n"
            "- None of this establishes operational or regulatory (FAA) "
            "certification.\n\n"
            "Full detail and dated history: `docs/RESEARCH_LOG.md` and "
            "`docs/FAA_AC_150_5220-24_BENCHMARK.md`."
        )


def render_dashboard() -> None:
    _page_header()
    reports = load_reports()
    latest = select_latest_verified_run(reports)

    st.subheader("Latest verified benchmark")
    if latest is None:
        st.warning(
            "NO VERIFIED BENCHMARK AVAILABLE\n\n"
            "Run `src/benchmark_faa.py` to generate a result for this view. "
            "No benchmark values are displayed until a verified result exists."
        )
    else:
        size = latest.get("results_by_size_bucket") or {}
        small = size.get("small", {})
        fp_per_image = latest.get("false_positives_per_image")
        loc_err = latest.get("mean_localization_error_pct_frame_diagonal")

        c1, c2, c3 = st.columns(3)
        c1.metric("Small-object detection rate", _fmt_pct(small.get("detection_rate")))
        c2.metric("False positives / image", f"{fp_per_image:.3f}" if fp_per_image is not None else "—")
        c3.metric("Localization error (% frame diagonal)", f"{loc_err:.2f}%" if loc_err is not None else "—")

        info_cols = st.columns(4)
        info_cols[0].markdown(f"**Run**\n\n{infer_run_label(latest)}")
        info_cols[1].markdown(f"**Model**\n\nYOLOv8")
        info_cols[2].markdown(f"**Evaluated images**\n\n{latest.get('n_images_evaluated', '—')}")
        run_ts = latest.get("run_timestamp_utc", "")
        info_cols[3].markdown(f"**Run date (UTC)**\n\n{run_ts[:10] if run_ts else '—'}")

        st.markdown("**Object-size performance**")
        for bucket in ("large", "medium", "small"):
            b = size.get(bucket)
            if not b:
                continue
            rate = b.get("detection_rate") or 0.0
            st.write(f"{bucket.capitalize()} — {_fmt_pct(b.get('detection_rate'))} "
                     f"({b.get('n_ground_truth', 0)} ground-truth instances)")
            st.progress(min(max(rate, 0.0), 1.0))

        faa = latest.get("faa_reference_thresholds", {})
        threshold = faa.get("min_detection_rate_small_object")
        meets = latest.get("small_object_meets_faa_90pct_threshold")
        if threshold is not None:
            note = (
                f"FAA AC 150/5220-24 references a {threshold * 100:.0f}% detection "
                "threshold for small objects. "
            )
            if meets:
                st.success(note + "This run meets that threshold for the small-object bucket.")
            else:
                st.warning(
                    note + "This run's small-object detection rate sits below that "
                    "threshold. This shortfall is reported rather than omitted: "
                    "small-object detection is the unsolved part of this problem."
                )

        if has_environmental_stratification(latest):
            st.markdown("**Environmental robustness (this run)**")
            env_cols = st.columns(2)
            light = latest.get("results_by_light_level") or {}
            weather = latest.get("results_by_weather") or {}
            with env_cols[0]:
                for level in ("bright", "dim", "dark"):
                    if level in light:
                        st.write(f"{level.capitalize()}: {_fmt_pct(light[level].get('detection_rate'))}")
            with env_cols[1]:
                for level in ("dry", "wet"):
                    if level in weather:
                        st.write(f"{level.capitalize()}: {_fmt_pct(weather[level].get('detection_rate'))}")
        else:
            st.caption("Environmental stratification unavailable for this benchmark run.")

    _measurement_scope_block()

    st.subheader("Try the detector")
    st.write("Use the sidebar to open **Image Detection** and upload a runway/taxiway/apron image.")


def render_benchmark_performance(cfg_path: str) -> None:
    st.title("Benchmark Performance")
    st.caption(
        "Results from this project's own evaluation instrument, "
        "`src/benchmark_faa.py`, reported in FAA AC 150/5220-24 terms rather "
        "than a generic ML metric."
    )
    reports = load_reports()
    if not reports:
        st.warning(
            "NO VERIFIED BENCHMARK AVAILABLE\n\n"
            "Run `src/benchmark_faa.py` to generate a result for this view."
        )
        return

    labels = [f"{infer_run_label(r)} — {r.get('run_timestamp_utc', '')[:19].replace('T', ' ')} UTC"
              for r in reports]
    idx = st.selectbox("Select a benchmark run", options=range(len(reports)),
                        format_func=lambda i: labels[i], index=len(reports) - 1)
    report = reports[idx]

    meta_cols = st.columns(4)
    meta_cols[0].markdown(f"**Run**\n\n{infer_run_label(report)}")
    meta_cols[1].markdown(f"**Evaluated images**\n\n{report.get('n_images_evaluated', '—')}")
    meta_cols[2].markdown(f"**IoU threshold**\n\n{report.get('iou_threshold', '—')}")
    meta_cols[3].markdown(f"**Confidence threshold**\n\n{report.get('confidence_threshold', '—')}")

    size = report.get("results_by_size_bucket") or {}
    st.markdown("**Detection rate by object size**")
    st.dataframe(
        [
            {
                "Size bucket": bucket.capitalize(),
                "Detection rate": _fmt_pct(v.get("detection_rate")),
                "True positives": v.get("tp"),
                "False negatives": v.get("fn"),
                "Ground-truth instances": v.get("n_ground_truth"),
            }
            for bucket, v in size.items()
        ],
        hide_index=True,
        use_container_width=True,
    )

    fp = report.get("false_positives_per_image")
    loc = report.get("mean_localization_error_pct_frame_diagonal")
    c1, c2 = st.columns(2)
    c1.metric("False positives / image", f"{fp:.3f}" if fp is not None else "—")
    c2.metric("Mean localization error (% frame diagonal)", f"{loc:.2f}%" if loc is not None else "—")

    if report.get("metadata_warnings"):
        st.warning("Metadata warnings recorded for this run:\n\n" +
                    "\n".join(f"- {w}" for w in report["metadata_warnings"]))

    with st.expander("Raw report JSON", expanded=False):
        st.json(report)
        st.caption(f"Source file: `{Path(report['_source_file']).relative_to(REPO_ROOT)}`")

    _measurement_scope_block()


def render_environmental_conditions() -> None:
    st.title("Environmental Conditions")
    st.caption(
        "Detection performance stratified by the light/weather labels present "
        "in the FOD-A dataset's own categorization metadata."
    )
    reports = load_reports()
    stratified = [r for r in reports if has_environmental_stratification(r)]
    if not stratified:
        st.warning(
            "ENVIRONMENTAL STRATIFICATION UNAVAILABLE\n\n"
            "No committed benchmark run currently includes light/weather "
            "stratification. Re-run `src/benchmark_faa.py` with "
            "`--metadata-csv` pointed at FOD-A's categorization CSV to "
            "generate one."
        )
        return

    report = stratified[-1]
    st.markdown(f"Showing: **{infer_run_label(report)}** — "
                f"{report.get('run_timestamp_utc', '')[:19].replace('T', ' ')} UTC")

    light = report.get("results_by_light_level") or {}
    weather = report.get("results_by_weather") or {}

    if light:
        st.markdown("**By light level**")
        st.dataframe(
            [
                {
                    "Condition": level.capitalize(),
                    "Ground-truth instances": v.get("n_ground_truth"),
                    "Detection rate": _fmt_pct(v.get("detection_rate")),
                }
                for level, v in light.items()
            ],
            hide_index=True,
            use_container_width=True,
        )
        for level in ("bright", "dim", "dark"):
            if level in light:
                st.write(f"{level.capitalize()}")
                st.progress(min(max(light[level].get("detection_rate") or 0.0, 0.0), 1.0))

    if weather:
        st.markdown("**By weather**")
        st.dataframe(
            [
                {
                    "Condition": level.capitalize(),
                    "Ground-truth instances": v.get("n_ground_truth"),
                    "Detection rate": _fmt_pct(v.get("detection_rate")),
                }
                for level, v in weather.items()
            ],
            hide_index=True,
            use_container_width=True,
        )
        for level in ("dry", "wet"):
            if level in weather:
                st.write(f"{level.capitalize()}")
                st.progress(min(max(weather[level].get("detection_rate") or 0.0, 0.0), 1.0))

    st.markdown("**Data accounting**")
    acc = metadata_accounting(report)
    st.write(f"Ground-truth objects evaluated: {acc['gt_total']}")
    for axis in ("light", "weather"):
        matched = acc[f"{axis}_matched"]
        unmatched = acc[f"{axis}_unmatched"]
        if matched is None:
            st.write(f"{axis.capitalize()} metadata: not available for this run.")
        elif unmatched == 0:
            st.success(f"{axis.capitalize()} metadata matched for all {matched} ground-truth objects.")
        else:
            st.warning(f"{axis.capitalize()} metadata matched for {matched} of "
                       f"{acc['gt_total']} ground-truth objects — {unmatched} unmatched. "
                       "This project has previously shipped a silent Windows-path "
                       "metadata-matching bug (see docs/RESEARCH_LOG.md, 2026-09-03); "
                       "an unmatched count above zero is worth investigating before "
                       "trusting the stratified numbers.")
    st.caption(f"Metadata source: `{report.get('metadata_source') or '—'}`")

    _measurement_scope_block()


def render_image_detection(args: argparse.Namespace) -> None:
    st.title("Image Detection")
    st.caption("Upload a runway/taxiway/apron image and run the trained detector on it.")

    cfg_path = st.sidebar.text_input("Config path", str(args.config), key="img_cfg_path")
    weights_path = st.sidebar.text_input("Weights path", str(args.weights) if args.weights else "", key="img_weights_path")
    camera_id = st.sidebar.text_input("Camera / source ID", "CAM-DEMO-01", key="img_camera_id")
    conf_thresh = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05, key="img_conf")

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
            st.markdown("**Model status:** :warning: UNAVAILABLE — model weights could not be loaded.")
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

    st.markdown("**Model status:** :large_green_circle: LOADED (YOLOv8)")

    uploaded = st.file_uploader("Upload a runway/taxiway/apron image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        st.stop()

    image = Image.open(uploaded)
    with st.spinner("Running inference…"):
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
                data=json.dumps(detections, indent=2),
                file_name="fod_detections.json",
                mime="application/json",
            )
        else:
            st.write("No detections above the confidence threshold.")


def render_methodology() -> None:
    st.title("Methodology")
    st.caption("How a result on the other pages was actually produced.")
    st.markdown(
        "```\n"
        "FOD-A DATASET (public)\n"
        "     ↓\n"
        "VOC → YOLO LABEL CONVERSION      src/voc_to_yolo.py\n"
        "     ↓\n"
        "SMALL-OBJECT-FOCUSED SPLIT       src/data_prep.py\n"
        "     ↓\n"
        "YOLOv8 TRAINING                  src/train.py (Ultralytics)\n"
        "     ↓\n"
        "FAA AC 150/5220-24 BENCHMARK     src/benchmark_faa.py\n"
        "     ↓\n"
        "SIZE-BUCKET STRATIFICATION       results_by_size_bucket\n"
        "     ↓\n"
        "LIGHT / WEATHER STRATIFICATION   results_by_light_level, results_by_weather\n"
        "     ↓\n"
        "THIS APPLICATION                 reads the committed JSON reports directly\n"
        "```"
    )
    st.markdown(
        "Object-size buckets (small/medium/large) are defined in `configs/fod.yaml` "
        "as bounding-box pixel-area thresholds, chosen to approximate the reference "
        "object classes in FAA AC 150/5220-24 §3.2.b(1) — not as calibrated physical "
        "measurements. The FAA pass/fail reference values used throughout this app "
        "(90% small-object detection rate, false-alarm-rate limits, 5 m localization "
        "requirement) are also defined there, each with the exact AC section it was "
        "verified against."
    )
    st.markdown(
        "Full dated history of every run, bug, and fix — including the training runs "
        "that did *not* work — is kept in `docs/RESEARCH_LOG.md`, which is the "
        "authoritative record behind everything summarized on these pages."
    )


def render_limitations_and_roadmap() -> None:
    st.title("Current Limitations")
    st.markdown(
        "- This is a **research prototype**, not a certified or operationally "
        "deployed system.\n"
        "- Validation is based on the FOD-A public dataset's held-out split — "
        "not cross-site, not validated at an operating airport.\n"
        "- Results are specific to the evaluated data and methodology and should "
        "not be generalized to airport-wide performance.\n"
        "- The detector covers a deliberately narrow starting class set "
        "(`configs/fod.yaml`: Wrench, Hammer, Screwdriver, SodaCan, Wood) — not "
        "FOD-A's full label set, and not every real-world FOD category (e.g. "
        "tire-fragment or rubber debris are not covered).\n"
        "- Object-size buckets are bounding-box pixel-area proxies, not "
        "calibrated physical (centimeter) sizes.\n"
        "- False alarms are reported per image, not per day.\n"
        "- Localization error is relative to frame diagonal, not meters.\n"
        "- There is no authorized real-airport camera/CCTV deployment or testing "
        "at any airport — dataset and model work require no such authorization "
        "and none has occurred.\n"
        "- None of this establishes FAA or other aviation regulatory "
        "certification."
    )

    st.subheader("Not yet available in this application")
    st.caption(
        "Listed here rather than built as an empty-looking page, per this "
        "project's rule against implying a capability that does not exist yet."
    )
    st.markdown(
        "- **Video Analysis** — `src/infer.py` supports video sources at the "
        "CLI level, but every detection from one video currently shares a single "
        "timestamp rather than a true per-frame one; an honest detection "
        "timeline needs that fixed first.\n"
        "- **Failure Analysis** (visual false-negative/false-positive examples) "
        "— the evaluation pipeline does not yet persist which specific images "
        "produced which errors, only aggregate counts.\n"
        "- **Experiment Explorer / Run Comparison** — only one training run's "
        "benchmark reports (Run 4) are currently committed to this repository; "
        "a prior run's benchmark artifacts were referenced in the research log "
        "but never committed, so there is nothing yet to compare against.\n"
        "- **Dataset Audit / Data Distribution** — class- and condition-level "
        "distribution counts over the full dataset are not currently computed "
        "or persisted anywhere the app can read.\n"
        "- **Run Provenance / Reproducibility pages** — git-commit and "
        "configuration provenance is not currently recorded inside the "
        "benchmark JSON reports themselves (only weights/data paths and "
        "run timestamp are). The **Benchmark Performance** page's \"Raw report "
        "JSON\" panel shows everything that is recorded today."
    )


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="ASSIS — FOD Detection", layout="wide")

    st.sidebar.markdown("### ASSIS — FOD Detection")
    page = st.sidebar.radio("Navigate", NAV_PAGES, label_visibility="collapsed")
    st.sidebar.divider()

    if page == "Dashboard":
        render_dashboard()
    elif page == "Image Detection":
        render_image_detection(args)
    elif page == "Benchmark Performance":
        render_benchmark_performance(str(args.config))
    elif page == "Environmental Conditions":
        render_environmental_conditions()
    elif page == "Methodology":
        render_methodology()
    elif page == "Limitations & Roadmap":
        render_limitations_and_roadmap()


if __name__ == "__main__":
    main()
