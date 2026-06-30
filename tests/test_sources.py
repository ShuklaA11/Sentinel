"""Unit tests for source helpers. Offline; no network.

Focus: `_iso` normalizes the four posted_at formats the live fetchers emit
(ISO strings, epoch ms, epoch seconds, relative text) into one contract —
an ISO-8601 UTC string, or '' when the value can't be a real timestamp.
"""
import datetime as dt

from src import sources


def _year(iso: str) -> int:
    return dt.datetime.fromisoformat(iso).year


def test_iso_passthrough_keeps_instant_normalized_to_utc():
    # Greenhouse/Ashby style offset timestamp -> same instant, expressed in UTC.
    out = sources._iso("2026-06-02T11:18:00-04:00")
    assert dt.datetime.fromisoformat(out) == dt.datetime(2026, 6, 2, 15, 18, tzinfo=dt.timezone.utc)


def test_iso_handles_z_suffix():
    out = sources._iso("2026-06-30T11:30:34.824Z")
    assert _year(out) == 2026
    assert dt.datetime.fromisoformat(out).tzinfo is not None


def test_iso_epoch_milliseconds_lever():
    # 1763656274612 ms -> 2025-11-20 UTC
    assert _year(sources._iso("1763656274612")) == 2025
    assert _year(sources._iso(1763656274612)) == 2025  # int form too


def test_iso_epoch_seconds_repo():
    # 1774073406 s -> 2026-03 UTC
    assert _year(sources._iso("1774073406")) == 2026
    assert _year(sources._iso(1774073406)) == 2026


def test_iso_naive_string_assumed_utc():
    out = sources._iso("2026-06-02T11:18:00")
    assert dt.datetime.fromisoformat(out).tzinfo is not None


def test_iso_unparseable_relative_text_becomes_empty():
    assert sources._iso("Jun 23") == ""
    assert sources._iso("Posted 30+ Days Ago") == ""


def test_iso_empty_inputs_become_empty():
    assert sources._iso("") == ""
    assert sources._iso(None) == ""
    assert sources._iso(0) == ""


def test_norm_normalizes_posted_at():
    # _norm must route posted_at through _iso so every fetcher gets it for free.
    row = sources._norm("123", "lever", "acme", "ML Intern", "NYC", "u", 1763656274612)
    assert _year(row["posted_at"]) == 2025
