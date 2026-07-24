"""Unit tests for presence-based liveness. Pure classification + the CSV reconcile path.
Paths are monkeypatched to tmp_path so real data/ files are never touched. Offline.
"""
import csv

from src import liveness, store


# --- healthy_units: the outage guard -----------------------------------------

def test_healthy_units_excludes_zero_and_errored():
    stats = {"greenhouse:live": 3, "greenhouse:empty": 0, "lever:live": 1}
    assert liveness.healthy_units(stats) == {"greenhouse:live", "lever:live"}


def test_present_ids_read_from_raw():
    raw = [{"id": "greenhouse:1"}, {"id": "lever:2"}]
    assert liveness.present_ids(raw) == {"greenhouse:1", "lever:2"}


# --- classify: the core three-way decision -----------------------------------

def _row(id_, source, company, status="new"):
    return {"id": id_, "source": source, "company": company, "status": status}


def test_absent_in_healthy_unit_is_closed():
    rows = [_row("greenhouse:ramp:1", "greenhouse", "ramp")]
    closed, reopened = liveness.classify(rows, present=set(), healthy={"greenhouse:ramp"})
    assert closed == {"greenhouse:ramp:1"} and reopened == set()


def test_present_in_healthy_unit_stays_open():
    rows = [_row("greenhouse:ramp:1", "greenhouse", "ramp")]
    closed, reopened = liveness.classify(
        rows, present={"greenhouse:ramp:1"}, healthy={"greenhouse:ramp"})
    assert closed == set() and reopened == set()


def test_outage_guard_never_closes_when_unit_unhealthy():
    # Unit polled 0/errored -> not in `healthy` -> absent id must NOT be closed.
    rows = [_row("greenhouse:ramp:1", "greenhouse", "ramp")]
    closed, reopened = liveness.classify(rows, present=set(), healthy=set())
    assert closed == set() and reopened == set()


def test_reopen_self_heals_when_id_reappears():
    rows = [_row("greenhouse:ramp:1", "greenhouse", "ramp", status="closed")]
    closed, reopened = liveness.classify(
        rows, present={"greenhouse:ramp:1"}, healthy={"greenhouse:ramp"})
    assert closed == set() and reopened == {"greenhouse:ramp:1"}


def test_repo_source_is_excluded():
    # Repo aggregators have no trustworthy presence signal -> skipped even if absent.
    rows = [_row("repo:simplify:1", "repo:simplify", "SomeCo")]
    closed, reopened = liveness.classify(rows, present=set(), healthy={"repo:simplify:SomeCo"})
    assert closed == set() and reopened == set()


def test_already_closed_absent_row_is_not_reclosed():
    rows = [_row("greenhouse:ramp:1", "greenhouse", "ramp", status="closed")]
    closed, reopened = liveness.classify(rows, present=set(), healthy={"greenhouse:ramp"})
    assert closed == set() and reopened == set()


# --- reconcile_status: the CSV write path ------------------------------------

def _seed_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=store.CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in store.CSV_FIELDS})


def test_reconcile_flips_status_and_counts(tmp_path, monkeypatch):
    p = tmp_path / "listings.csv"
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "CSV_PATH", str(p))
    _seed_csv(p, [
        {"id": "a", "status": "new", "company": "X"},
        {"id": "b", "status": "closed", "company": "Y"},
        {"id": "c", "status": "new", "company": "Z"},
    ])
    changed = store.reconcile_status(closed_ids={"a"}, reopened_ids={"b"})
    assert changed == 2
    out = {r["id"]: r["status"] for r in csv.DictReader(open(p))}
    assert out == {"a": "closed", "b": "new", "c": "new"}


def test_reconcile_noop_when_nothing_to_change(tmp_path, monkeypatch):
    p = tmp_path / "listings.csv"
    monkeypatch.setattr(store, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(store, "CSV_PATH", str(p))
    _seed_csv(p, [{"id": "a", "status": "new", "company": "X"}])
    before = p.read_text()
    assert store.reconcile_status(set(), set()) == 0
    # id "a" already open -> closing an unrelated id changes nothing, no rewrite.
    assert store.reconcile_status(closed_ids={"zzz"}, reopened_ids=set()) == 0
    assert p.read_text() == before
