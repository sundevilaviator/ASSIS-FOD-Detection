"""Tests for the demo app's weights-resolution helpers.

These cover the hosted-deployment path (no local weights file present, fetch
once from a published release asset). Verified against hand-constructed
known-correct cases rather than "it ran without raising":

  - a cache hit must NOT re-download (asserted by pointing the URL at a
    different payload and checking the original content survives);
  - a zero-byte file is NOT a valid cache hit (a truncated/failed earlier
    download must not permanently poison the cache);
  - an interrupted download must leave no `.partial` file behind and must
    not create the destination file at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

# streamlit is imported at module scope by the app; skip cleanly if absent.
streamlit_app = pytest.importorskip(
    "streamlit_app", reason="streamlit not installed in this environment"
)

download_weights = streamlit_app.download_weights
configured_weights_url = streamlit_app.configured_weights_url


def _file_url(p: Path) -> str:
    return p.resolve().as_uri()


def test_downloads_when_absent(tmp_path):
    src = tmp_path / "source.pt"
    src.write_bytes(b"WEIGHTS-PAYLOAD-A")
    dest = tmp_path / "cache" / "best.pt"

    returned = download_weights(_file_url(src), dest)

    assert returned == dest
    assert dest.read_bytes() == b"WEIGHTS-PAYLOAD-A"


def test_cache_hit_does_not_redownload(tmp_path):
    """An existing non-empty file must be reused, not overwritten."""
    dest = tmp_path / "best.pt"
    dest.write_bytes(b"ALREADY-CACHED")

    other = tmp_path / "different.pt"
    other.write_bytes(b"SHOULD-NOT-BE-FETCHED")

    download_weights(_file_url(other), dest)

    assert dest.read_bytes() == b"ALREADY-CACHED"


def test_zero_byte_file_is_not_a_valid_cache_hit(tmp_path):
    """A previous failed download must not permanently poison the cache."""
    dest = tmp_path / "best.pt"
    dest.write_bytes(b"")
    assert dest.stat().st_size == 0

    src = tmp_path / "source.pt"
    src.write_bytes(b"REAL-PAYLOAD")

    download_weights(_file_url(src), dest)

    assert dest.read_bytes() == b"REAL-PAYLOAD"


def test_failed_download_leaves_no_partial_or_dest_file(tmp_path):
    dest = tmp_path / "cache" / "best.pt"
    missing = tmp_path / "does-not-exist.pt"

    with pytest.raises(Exception):
        download_weights(_file_url(missing), dest)

    assert not dest.exists(), "destination must not exist after a failed download"
    partials = list(dest.parent.glob("*.partial"))
    assert partials == [], f"left behind partial files: {partials}"


def test_creates_nested_cache_directory(tmp_path):
    src = tmp_path / "source.pt"
    src.write_bytes(b"X")
    dest = tmp_path / "a" / "b" / "c" / "best.pt"

    download_weights(_file_url(src), dest)

    assert dest.exists()


def test_env_var_takes_precedence(monkeypatch):
    monkeypatch.setenv("ASSIS_FOD_WEIGHTS_URL", "https://example.invalid/from-env.pt")
    assert configured_weights_url() == "https://example.invalid/from-env.pt"


def test_env_var_whitespace_is_stripped(monkeypatch):
    monkeypatch.setenv("ASSIS_FOD_WEIGHTS_URL", "  https://example.invalid/x.pt  ")
    assert configured_weights_url() == "https://example.invalid/x.pt"


def test_blank_env_var_falls_through_rather_than_returning_blank(monkeypatch):
    """An empty env var must not shadow a configured default."""
    monkeypatch.setenv("ASSIS_FOD_WEIGHTS_URL", "   ")
    monkeypatch.setattr(streamlit_app, "DEFAULT_WEIGHTS_URL", "https://example.invalid/default.pt")
    assert configured_weights_url() == "https://example.invalid/default.pt"


def test_returns_empty_when_nothing_configured(monkeypatch):
    monkeypatch.delenv("ASSIS_FOD_WEIGHTS_URL", raising=False)
    monkeypatch.setattr(streamlit_app, "DEFAULT_WEIGHTS_URL", "")
    assert configured_weights_url() == ""
