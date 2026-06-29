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


def test_pending_queue_enqueue_load_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "PENDING_PATH", str(tmp_path / "pending.csv"))

    assert store.load_pending() == []   # missing file -> empty

    store.enqueue_pending([
        {"id": "a", "company": "X", "title": "ML Intern", "track": "ml",
         "score": 92, "location": "SF", "url": "u1", "fit_reason": "good"},
        {"id": "b", "company": "Y", "title": "SWE Intern", "track": "swe",
         "score": "", "location": "", "url": "u2", "fit_reason": ""},
    ], "2026-06-29T00:00:00")

    pend = store.load_pending()
    assert len(pend) == 2
    assert pend[0]["score"] == 92    # coerced back to int for sorting
    assert pend[1]["score"] == ""    # unscored stays ""
    assert pend[0]["company"] == "X"

    store.clear_pending()
    assert store.load_pending() == []
