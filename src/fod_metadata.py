#!/usr/bin/env python3
"""Locate and validate FOD-A's light-level / weather categorization annotations.

WHY THIS EXISTS
---------------
`benchmark_faa.py` can stratify results by lighting and weather, which is the
one reporting gap no reviewed vendor publishes against. That path was never
exercised because the metadata could not be found — Step 10 of the Colab
notebook searched the Kaggle download and came back empty.

The FOD-A paper (Munyer et al.) explains why. The categorization annotations
are written by the dataset's own expansion tool alongside the images, and the
dataset is distributed in two forms: an original format (8.3 GB, 400x400) and
a Pascal VOC format (412 MB, 300x300). The Kaggle mirror carries the Pascal
VOC form, which is bounding boxes only. The categorization data travels with
the original-format distribution.

So this module does not assume a filename, a location, or a column layout.
It searches, reports what it found, and validates the contents against the
counts published in the paper before anything downstream trusts it.

PUBLISHED COUNTS (FOD-A paper, Table I: "Categorization Statistics")
--------------------------------------------------------------------
    Weather      Dry 26,647   Wet 7,216                    -> 33,863
    Light-level  Dark 4,387   Dim 12,464   Bright 17,012   -> 33,863

The Pascal VOC mirror in hand has 33,793 annotation files — 70 fewer. The
mirror is therefore a slightly different revision from the one the paper
describes. That discrepancy is expected, is reported rather than silently
reconciled, and is why `validate_against_paper()` returns a report instead of
raising: a near-miss is information, not necessarily an error.
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# From the FOD-A paper, Table I. Hard-coded deliberately: these are the
# published reference values, not something to recompute from the data being
# checked against them.
PAPER_WEATHER_COUNTS = {"dry": 26647, "wet": 7216}
PAPER_LIGHT_COUNTS = {"dark": 4387, "dim": 12464, "bright": 17012}
PAPER_TOTAL = 33863

WEATHER_VALUES = frozenset(PAPER_WEATHER_COUNTS)
LIGHT_VALUES = frozenset(PAPER_LIGHT_COUNTS)


@dataclass
class MetadataFile:
    """A CSV under the dataset root, described by what it actually contains."""

    path: Path
    columns: list[str]
    n_rows: int
    weather_column: str | None = None
    light_column: str | None = None
    path_column: str | None = None

    @property
    def is_categorization(self) -> bool:
        """True only if this file carries BOTH weather and light categories.

        A file with one and not the other is not the categorization annotation
        and must not be treated as a partial substitute — stratifying on half
        the dimensions would silently produce a different analysis than the one
        being claimed.
        """
        return self.weather_column is not None and self.light_column is not None


def _classify_columns(rows: list[dict[str, str]], columns: list[str]) -> dict[str, str | None]:
    """Identify which columns hold weather, light level, and image path.

    Identification is by VALUES, not by column name. Column names in a
    community dataset are not a contract; the value vocabulary published in
    the paper is the stable thing.
    """
    found: dict[str, str | None] = {"weather": None, "light": None, "path": None}
    for col in columns:
        values = {(r.get(col) or "").strip().lower() for r in rows}
        values.discard("")
        if not values:
            continue
        if values <= WEATHER_VALUES and found["weather"] is None:
            found["weather"] = col
        elif values <= LIGHT_VALUES and found["light"] is None:
            found["light"] = col
        elif found["path"] is None and any(
            v.endswith((".jpg", ".jpeg", ".png")) for v in values
        ):
            found["path"] = col
    return found


def scan_for_metadata(root: Path, sample_rows: int = 500) -> list[MetadataFile]:
    """Find every CSV under `root` and describe it.

    Returns all CSVs, not just matching ones, so a failure tells you what IS
    present rather than only that nothing matched.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    out: list[MetadataFile] = []
    for path in sorted(root.rglob("*.csv")):
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                columns = list(reader.fieldnames or [])
                sample: list[dict[str, str]] = []
                n_rows = 0
                for i, row in enumerate(reader):
                    n_rows += 1
                    if i < sample_rows:
                        sample.append(row)
        except Exception:  # noqa: BLE001 - a malformed CSV is a finding, not a crash
            out.append(MetadataFile(path=path, columns=[], n_rows=0))
            continue

        cls = _classify_columns(sample, columns)
        out.append(
            MetadataFile(
                path=path,
                columns=columns,
                n_rows=n_rows,
                weather_column=cls["weather"],
                light_column=cls["light"],
                path_column=cls["path"],
            )
        )
    return out


def read_categorization(meta: MetadataFile) -> tuple[Counter, Counter]:
    """Return (weather_counts, light_counts) for a categorization file."""
    if not meta.is_categorization:
        raise ValueError(f"{meta.path} is not a categorization file")
    weather: Counter = Counter()
    light: Counter = Counter()
    with meta.path.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            w = (row.get(meta.weather_column) or "").strip().lower()
            l = (row.get(meta.light_column) or "").strip().lower()
            if w:
                weather[w] += 1
            if l:
                light[l] += 1
    return weather, light


@dataclass
class ValidationReport:
    weather_counts: dict[str, int]
    light_counts: dict[str, int]
    total: int
    matches_paper_exactly: bool
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"weather : {dict(sorted(self.weather_counts.items()))}",
            f"light   : {dict(sorted(self.light_counts.items()))}",
            f"total   : {self.total}  (paper: {PAPER_TOTAL})",
            f"exact match to paper: {self.matches_paper_exactly}",
        ]
        lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


def validate_against_paper(weather: Counter, light: Counter) -> ValidationReport:
    """Compare observed counts to the paper's Table I.

    Returns a report rather than raising. A revision difference is a fact to
    record, not a failure to abort on — but it must be visible, because a
    silently different dataset revision would make the stratified result
    incomparable to anything published about FOD-A.
    """
    notes: list[str] = []
    w_total = sum(weather.values())
    l_total = sum(light.values())

    if w_total != l_total:
        notes.append(
            f"weather rows ({w_total}) != light rows ({l_total}); every image "
            "should carry both categorizations"
        )

    unexpected_w = set(weather) - WEATHER_VALUES
    unexpected_l = set(light) - LIGHT_VALUES
    if unexpected_w:
        notes.append(f"unexpected weather values: {sorted(unexpected_w)}")
    if unexpected_l:
        notes.append(f"unexpected light values: {sorted(unexpected_l)}")

    exact = dict(weather) == PAPER_WEATHER_COUNTS and dict(light) == PAPER_LIGHT_COUNTS
    if not exact:
        delta = w_total - PAPER_TOTAL
        notes.append(
            f"counts differ from the paper by {delta:+d} rows overall — this "
            "mirror is a different dataset revision. Record the observed "
            "counts in docs/RESEARCH_LOG.md rather than citing the paper's."
        )

    return ValidationReport(
        weather_counts=dict(weather),
        light_counts=dict(light),
        total=w_total,
        matches_paper_exactly=exact,
        notes=notes,
    )


def describe_scan(files: list[MetadataFile]) -> str:
    """Human-readable scan result, printed before anything is trusted."""
    if not files:
        return "No CSV files found under the search root."
    rows = []
    for f in files:
        mark = "  <-- categorization" if f.is_categorization else ""
        cols = ", ".join(f.columns[:8]) or "(unreadable)"
        rows.append(f"  {f.path.name:<40} rows={f.n_rows:<8} cols=[{cols}]{mark}")
    return "\n".join(rows)
