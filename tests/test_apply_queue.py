"""Unit tests for the apply queue ledger. QUEUE_PATH is monkeypatched to tmp_path
so the real data/ files are never touched. Offline; no network.
"""
from src import apply_queue as aq


def test_ats_from_derives_surface_from_source():
    assert aq.ats_from("greenhouse") == "greenhouse"
    assert aq.ats_from("lever:acme") == "lever"          # slug stripped
    assert aq.ats_from("repo:jobright-ai/2026-SWE") == "unknown"  # board-routed
    assert aq.ats_from("") == "unknown"


def test_enqueue_dedups_on_id_and_stamps_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(aq, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(aq, "QUEUE_PATH", str(tmp_path / "apply_queue.csv"))

    assert aq.load_queue() == []   # missing file -> empty

    added = aq.enqueue([
        {"id": "greenhouse:1", "company": "X", "title": "ML Intern", "track": "ml",
         "score": 92, "location": "SF", "url": "u1", "source": "greenhouse"},
        {"id": "lever:2", "company": "Y", "title": "SWE Intern", "track": "swe",
         "score": "", "location": "", "url": "u2", "source": "lever:y"},
    ], "2026-07-21T00:00:00")
    assert added == 2

    rows = aq.load_queue()
    assert len(rows) == 2
    assert rows[0]["status"] == "queued"
    assert rows[0]["attempts"] == 0           # coerced to int
    assert rows[0]["score"] == 92
    assert rows[0]["ats"] == "greenhouse"
    assert rows[1]["ats"] == "lever"
    assert rows[1]["score"] == ""             # unscored stays ""

    # Re-enqueue: same ids are skipped, a new id is added.
    added = aq.enqueue([
        {"id": "greenhouse:1", "company": "X", "title": "ML Intern", "source": "greenhouse"},
        {"id": "ashby:3", "company": "Z", "title": "AI Intern", "source": "ashby:z"},
    ], "2026-07-21T01:00:00")
    assert added == 1
    assert len(aq.load_queue()) == 3


def test_update_row_is_pure_and_targets_by_id(tmp_path, monkeypatch):
    monkeypatch.setattr(aq, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(aq, "QUEUE_PATH", str(tmp_path / "apply_queue.csv"))
    aq.enqueue([{"id": "greenhouse:1", "company": "X", "source": "greenhouse"}],
               "2026-07-21T00:00:00")

    rows = aq.load_queue()
    updated = aq.update_row(rows, "greenhouse:1",
                            status="prepared", prepared_at="2026-07-21T02:00:00",
                            screenshot="shots/gh1.png")

    # Original list untouched (immutability).
    assert rows[0]["status"] == "queued"
    assert updated[0]["status"] == "prepared"
    assert updated[0]["prepared_at"] == "2026-07-21T02:00:00"
    assert updated[0]["screenshot"] == "shots/gh1.png"

    # Unknown id passes through unchanged.
    same = aq.update_row(rows, "nope:9", status="failed")
    assert same[0]["status"] == "queued"

    # Round-trips through save/load.
    aq.save_queue(updated)
    assert aq.load_queue()[0]["status"] == "prepared"
