"""Tests for src/benchmark_report.py.

Deliberately runs against the real committed files in
docs/benchmark_results/ (not synthetic fixtures) for the "latest verified
run" and "metadata accounting" checks, so a regression in either the
committed data or the parsing logic is caught the same way — this mirrors
how the Streamlit app itself will read these files. A couple of pure
unit tests for `infer_run_label` and empty-directory handling use
constructed dicts, since those specifically test the parsing logic in
isolation.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))  # matches tests/test_benchmark_faa.py convention

from benchmark_report import (  # noqa: E402
    has_environmental_stratification,
    infer_run_label,
    load_benchmark_reports,
    metadata_accounting,
    select_latest_verified_run,
)

REAL_RESULTS_DIR = REPO_ROOT / "docs" / "benchmark_results"


def test_load_benchmark_reports_finds_real_committed_files():
    reports = load_benchmark_reports(REAL_RESULTS_DIR)
    # Both run-4 reports committed 2026-09-03 must be discovered.
    assert len(reports) == 2
    assert all(r["_source_file"].endswith(".json") for r in reports)
    # Sorted oldest -> newest by run_timestamp_utc.
    assert reports[0]["run_timestamp_utc"] < reports[1]["run_timestamp_utc"]


def test_load_benchmark_reports_missing_directory_returns_empty_list():
    assert load_benchmark_reports(REAL_RESULTS_DIR / "does_not_exist") == []


def test_select_latest_verified_run_picks_the_metadata_stratified_one():
    reports = load_benchmark_reports(REAL_RESULTS_DIR)
    latest = select_latest_verified_run(reports)
    assert latest is not None
    # The later of the two run-4 timestamps (T195007Z) is the one that
    # carries the post-fix light/weather stratification.
    assert latest["run_timestamp_utc"].startswith("2026-09-03T19:50:07")
    assert has_environmental_stratification(latest)


def test_select_latest_verified_run_empty_list_returns_none():
    assert select_latest_verified_run([]) is None


def test_metadata_accounting_on_real_report_shows_full_match():
    reports = load_benchmark_reports(REAL_RESULTS_DIR)
    latest = select_latest_verified_run(reports)
    accounting = metadata_accounting(latest)
    # 123 small + 222 medium + 525 large = 870 ground-truth objects total.
    assert accounting["gt_total"] == 870
    # The 2026-09-03 metadata-join fix achieved a 100% match rate for both
    # axes on this run (recorded in docs/RESEARCH_LOG.md) — assert that
    # positive result stays true rather than silently regressing.
    assert accounting["light_matched"] == 870
    assert accounting["light_unmatched"] == 0
    assert accounting["weather_matched"] == 870
    assert accounting["weather_unmatched"] == 0


def test_metadata_accounting_on_report_without_stratification():
    reports = load_benchmark_reports(REAL_RESULTS_DIR)
    plain = reports[0]  # the T194949Z report, run without --metadata-csv
    assert not has_environmental_stratification(plain)
    accounting = metadata_accounting(plain)
    assert accounting["gt_total"] == 870
    assert accounting["light_matched"] is None
    assert accounting["light_unmatched"] is None
    assert accounting["weather_matched"] is None
    assert accounting["weather_unmatched"] is None


def test_infer_run_label_from_weights_path_with_suffix():
    report = {"weights": "/content/runs/detect/run4-orig400/weights/best.pt", "data": ""}
    assert infer_run_label(report) == "Run 4 (orig400)"


def test_infer_run_label_from_data_path_no_suffix():
    report = {"weights": "", "data": "/content/fod-a-split-run4/data.yaml"}
    assert infer_run_label(report) == "Run 4"


def test_infer_run_label_falls_back_when_no_run_token_present():
    report = {"weights": "/content/best.pt", "data": "/content/data.yaml"}
    assert infer_run_label(report) == "Unlabeled run"


def test_infer_run_label_on_real_committed_reports():
    reports = load_benchmark_reports(REAL_RESULTS_DIR)
    for report in reports:
        assert infer_run_label(report) == "Run 4 (orig400)"
