"""Tests for FOD-A categorization-metadata discovery and validation.

Every expected value here is hand-computed in the test. The paper's published
counts (Table I) are used as the reference for validation, and the tests
confirm that a mirror which differs from the paper is REPORTED rather than
either silently accepted or treated as fatal — which is the actual situation:
the Pascal VOC mirror in hand has 33,793 annotation files against the paper's
33,863.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fod_metadata import (  # noqa: E402
    PAPER_LIGHT_COUNTS,
    PAPER_TOTAL,
    PAPER_WEATHER_COUNTS,
    describe_scan,
    read_categorization,
    scan_for_metadata,
    validate_against_paper,
)


def _csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------
# The paper's own numbers must be internally consistent
# --------------------------------------------------------------------------

def test_paper_reference_totals_agree():
    """Both categorizations cover every image, so both must sum to the total."""
    assert sum(PAPER_WEATHER_COUNTS.values()) == PAPER_TOTAL
    assert sum(PAPER_LIGHT_COUNTS.values()) == PAPER_TOTAL
    assert PAPER_TOTAL == 33863  # hand-checked: 26647+7216 and 4387+12464+17012


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def test_finds_categorization_csv_regardless_of_filename(tmp_path):
    """Column NAMES are not a contract; the value vocabulary is."""
    _csv(tmp_path / "nested" / "anything_at_all.csv",
         ["relative_path", "col_b", "col_c"],
         [["img/000001.jpg", "dry", "bright"],
          ["img/000002.jpg", "wet", "dark"]])
    found = scan_for_metadata(tmp_path)
    assert len(found) == 1
    m = found[0]
    assert m.is_categorization
    assert m.weather_column == "col_b"
    assert m.light_column == "col_c"
    assert m.path_column == "relative_path"
    assert m.n_rows == 2


def test_a_csv_with_only_weather_is_not_categorization(tmp_path):
    """Half the dimensions would silently change the analysis being claimed."""
    _csv(tmp_path / "partial.csv", ["p", "weather"],
         [["a.jpg", "dry"], ["b.jpg", "wet"]])
    m = scan_for_metadata(tmp_path)[0]
    assert m.weather_column == "weather"
    assert m.light_column is None
    assert not m.is_categorization


def test_unrelated_csvs_are_listed_but_not_matched(tmp_path):
    """A failure should say what IS there."""
    _csv(tmp_path / "results.csv", ["epoch", "loss"], [["1", "0.5"]])
    files = scan_for_metadata(tmp_path)
    assert len(files) == 1
    assert not files[0].is_categorization
    assert "results.csv" in describe_scan(files)


def test_scan_reports_all_csvs_and_marks_the_right_one(tmp_path):
    _csv(tmp_path / "results.csv", ["epoch", "loss"], [["1", "0.5"]])
    _csv(tmp_path / "labels.csv", ["path", "w", "l"],
         [["a.jpg", "dry", "dim"]])
    out = describe_scan(scan_for_metadata(tmp_path))
    assert "results.csv" in out and "labels.csv" in out
    marked = [l for l in out.splitlines() if "categorization" in l]
    assert len(marked) == 1 and "labels.csv" in marked[0]


def test_malformed_csv_is_reported_not_raised(tmp_path):
    (tmp_path / "broken.csv").write_bytes(b"\x00\x01\x02")
    files = scan_for_metadata(tmp_path)
    assert len(files) == 1  # listed, not crashed


def test_missing_root_raises_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_for_metadata(tmp_path / "nope")


def test_case_and_whitespace_are_tolerated(tmp_path):
    _csv(tmp_path / "m.csv", ["p", "w", "l"],
         [["a.jpg", " Dry ", "BRIGHT"], ["b.jpg", "WET", " dark"]])
    m = scan_for_metadata(tmp_path)[0]
    assert m.is_categorization
    weather, light = read_categorization(m)
    assert weather == Counter({"dry": 1, "wet": 1})
    assert light == Counter({"bright": 1, "dark": 1})


# --------------------------------------------------------------------------
# Validation against the paper
# --------------------------------------------------------------------------

def test_exact_paper_counts_validate_clean():
    w = Counter(PAPER_WEATHER_COUNTS)
    l = Counter(PAPER_LIGHT_COUNTS)
    r = validate_against_paper(w, l)
    assert r.matches_paper_exactly
    assert r.total == PAPER_TOTAL
    assert r.notes == []


def test_the_actual_mirror_discrepancy_is_reported_not_fatal():
    """The real case: 33,793 rows against the paper's 33,863 — 70 fewer.

    Must produce a report saying so, not an exception and not silence.
    """
    w = Counter({"dry": 26647 - 70, "wet": 7216})
    l = Counter({"dark": 4387, "dim": 12464, "bright": 17012 - 70})
    r = validate_against_paper(w, l)
    assert not r.matches_paper_exactly
    assert r.total == 33793
    assert any("-70" in n for n in r.notes), r.notes
    assert any("RESEARCH_LOG" in n for n in r.notes)


def test_mismatched_row_counts_between_dimensions_is_flagged():
    """Every image should carry both categorizations."""
    r = validate_against_paper(Counter({"dry": 10}), Counter({"dim": 9}))
    assert any("!=" in n for n in r.notes)


def test_unexpected_category_values_are_flagged():
    r = validate_against_paper(
        Counter({"dry": 5, "foggy": 1}), Counter({"dim": 6})
    )
    assert any("foggy" in n for n in r.notes)


def test_report_renders_the_numbers():
    r = validate_against_paper(
        Counter(PAPER_WEATHER_COUNTS), Counter(PAPER_LIGHT_COUNTS)
    )
    s = str(r)
    assert "26647" in s and "17012" in s and "33863" in s
