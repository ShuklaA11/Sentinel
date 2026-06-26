"""Unit tests for dedupe + persistence. Paths are monkeypatched to tmp_path so
the real data/ files are never touched. Offline; no network.
"""
import csv

from src import store


def test_split_new_drops_already_seen():
    listings = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    new = store.split_new(listings, {"b"})
    assert [l["id"] for l in new] == ["a", "c"]


def test_seen_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "SEEN_PATH", str(tmp_path / "seen.json"))
    assert store.load_seen() == set()          # missing file -> empty set
    store.save_seen({"a", "b"})
    assert store.load_seen() == {"a", "b"}


def test_append_csv_writes_header_once_and_appends(tmp_path, monkeypatch):
    p = tmp_path / "listings.csv"
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "CSV_PATH", str(p))

    rows = [{"id": "a", "company": "X", "title": "ML Intern", "track": "ml"}]
    store.append_csv(rows, "2026-06-26T00:00:00")
    out = list(csv.DictReader(open(p)))
    assert len(out) == 1
    assert out[0]["company"] == "X"
    assert out[0]["status"] == "new"
    assert out[0]["id"] == "a"

    # Appending again adds a row without a duplicate header.
    store.append_csv(rows, "2026-06-26T01:00:00")
    assert len(list(csv.DictReader(open(p)))) == 2
