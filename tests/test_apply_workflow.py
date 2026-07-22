"""Unit test for the apply-session helper. LISTINGS is monkeypatched to a tmp CSV.
Offline; the browser/session parts are integration (run locally).
"""
import csv

from src import apply_workflow as w


def _write_listings(path, rows):
    fields = ["score", "company", "title", "url", "source"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def test_high_fit_manual_filters_and_sorts(tmp_path, monkeypatch):
    p = tmp_path / "listings.csv"
    _write_listings(p, [
        {"score": "92", "company": "Together", "title": "RL Intern", "url": "u1", "source": "repo:Simplify"},
        {"score": "88", "company": "X", "title": "ML Intern", "url": "u2", "source": "greenhouse"},   # supported -> excluded
        {"score": "70", "company": "Y", "title": "SWE Intern", "url": "u3", "source": "repo:JobRight"},  # below threshold
        {"score": "90", "company": "Amazon", "title": "Robotics", "url": "u4", "source": "workday"},   # unsupported ATS
        {"score": "", "company": "Z", "title": "unscored", "url": "u5", "source": "repo:X"},           # unscored
    ])
    monkeypatch.setattr(w, "LISTINGS", str(p))

    manual = w.high_fit_manual(threshold=85, cap=10)
    # only high-fit AND unsupported/repo, sorted by score desc; greenhouse excluded.
    assert [r["company"] for r in manual] == ["Together", "Amazon"]

    # cap is honored
    assert len(w.high_fit_manual(threshold=85, cap=1)) == 1
