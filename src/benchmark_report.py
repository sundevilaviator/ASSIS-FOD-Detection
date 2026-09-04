#!/usr/bin/env python3
"""
ASSIS FOD Module — benchmark report loading for the Streamlit app.

The app must never hard-code detection rate / false-alarm rate / localization
error numbers: those are exactly the figures that have changed across runs in
this project's history (see docs/RESEARCH_LOG.md), and a hard-coded figure
silently goes stale the next time a benchmark is re-run. This module is the
single place that reads the committed `docs/benchmark_results/*.json` files
produced by `src/benchmark_faa.py` and turns them into the small, derived
values the UI actually displays — nothing here invents a field that is not
already present in a committed report.

Every function is pure (no Streamlit imports, no caching decorators) so it
can be unit-tested directly against the real committed report files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPORT_GLOB = "benchmark_*.json"

# Matches a "runN" token followed by an optional short suffix, e.g.
# "run4-orig400" -> ("4", "orig400"), "run3" -> ("3", "").
_RUN_TOKEN_RE = re.compile(r"run(\d+)[-_]?([a-zA-Z0-9]*)")


def load_benchmark_reports(results_dir: Path) -> list[dict]:
    """Load every committed benchmark JSON report from `results_dir`.

    Returns reports sorted oldest-to-newest by `run_timestamp_utc` (reports
    missing that field, which should not happen for anything produced by
    `src/benchmark_faa.py`, sort first). Each returned dict has a
    `"_source_file"` key added so the UI can show/link back to exactly which
    committed file a figure came from — required by the "Run Provenance"
    principle in the UX spec (every displayed number must be traceable).
    Returns an empty list if the directory does not exist or has no reports;
    callers must render an explicit empty state rather than assuming a run
    exists.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    reports = []
    for path in sorted(results_dir.glob(REPORT_GLOB)):
        try:
            report = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        report = dict(report)
        report["_source_file"] = str(path)
        reports.append(report)
    reports.sort(key=lambda r: r.get("run_timestamp_utc") or "")
    return reports


def infer_run_label(report: dict) -> str:
    """Best-effort human-readable run label, e.g. "Run 4 (orig400)".

    Derived only from the `weights`/`data` paths already present in the
    report (set by whoever ran `src/benchmark_faa.py`) — never guessed or
    hard-coded per-run. Falls back to a generic label when no "runN" token
    is present in either path, rather than fabricating a run number.
    """
    for key in ("weights", "data"):
        value = report.get(key) or ""
        m = _RUN_TOKEN_RE.search(value)
        if m:
            number, suffix = m.group(1), m.group(2)
            label = f"Run {number}"
            if suffix:
                label += f" ({suffix})"
            return label
    return "Unlabeled run"


def select_latest_verified_run(reports: list[dict]) -> dict | None:
    """The run the Dashboard should show: the most recently produced report.

    `reports` is expected pre-sorted oldest-to-newest by
    `load_benchmark_reports`; this simply takes the last one. Returns None
    when no reports exist, which callers must render as an explicit
    "no verified benchmark available" state (see UX spec) rather than a
    placeholder.
    """
    if not reports:
        return None
    return reports[-1]


def metadata_accounting(report: dict) -> dict:
    """Derive data-integrity counts already implied by a report's numbers.

    The UX spec asks for a "METADATA JOIN: rows loaded / matched / unmatched"
    block, specifically because this project has previously shipped a silent
    Windows-path metadata-matching bug (see docs/RESEARCH_LOG.md, 2026-09-03).
    `src/benchmark_faa.py`'s JSON report does not carry a row-count field
    directly, but it does carry, per stratification axis, the ground-truth
    object count matched to each light/weather label — and the unstratified
    size-bucket breakdown carries the true total. Comparing the two is a
    real cross-check computed from data already in the report, not a new
    invented field.

    Returns a dict with `gt_total` plus, for each of "light" and "weather":
    `<axis>_matched` and `<axis>_unmatched` (both None when that
    stratification is absent from the report, e.g. a benchmark run without
    a --metadata-csv).
    """
    size_results = report.get("results_by_size_bucket") or {}
    gt_total = sum(bucket.get("n_ground_truth", 0) for bucket in size_results.values())

    out: dict = {"gt_total": gt_total}
    for axis, key in (("light", "results_by_light_level"), ("weather", "results_by_weather")):
        axis_results = report.get(key)
        if not axis_results:
            out[f"{axis}_matched"] = None
            out[f"{axis}_unmatched"] = None
            continue
        matched = sum(bucket.get("n_ground_truth", 0) for bucket in axis_results.values())
        out[f"{axis}_matched"] = matched
        out[f"{axis}_unmatched"] = gt_total - matched
    return out


def has_environmental_stratification(report: dict) -> bool:
    """True if `report` has at least one non-empty light/weather breakdown."""
    return bool(report.get("results_by_light_level")) or bool(report.get("results_by_weather"))
