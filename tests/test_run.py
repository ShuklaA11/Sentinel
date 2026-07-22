"""Unit test for the companies.yml + harvested.yml merge (union, deduped, order
preserved). _load is monkeypatched so no config files are read. Offline.
"""
from src import run


def test_merge_companies_unions_and_dedupes(monkeypatch):
    def fake_load(name):
        if name == "companies.yml":
            return {"greenhouse": ["airbnb", "stripe"], "repos": ["r1"]}
        if name == "harvested.yml":
            return {"greenhouse": ["stripe", "figma"], "lever": ["plaid"]}
        return {}

    monkeypatch.setattr(run, "_load", fake_load)
    merged = run._merge_companies()

    # greenhouse: curated + harvested, deduped, curated order first.
    assert merged["greenhouse"] == ["airbnb", "stripe", "figma"]
    # lever: only harvested.
    assert merged["lever"] == ["plaid"]
    # repos passes through untouched.
    assert merged["repos"] == ["r1"]


def test_apply_candidates_keeps_only_supported_ats():
    high = [
        {"id": "greenhouse:1", "source": "greenhouse"},
        {"id": "lever:2", "source": "lever:x"},           # supported
        {"id": "workday:4", "source": "workday"},         # not supported yet
        {"id": "repo:3", "source": "repo:SimplifyJobs"},  # board-routed, ats unknown
    ]
    ready = run.apply_candidates(high)
    assert [l["id"] for l in ready] == ["greenhouse:1", "lever:2"]
