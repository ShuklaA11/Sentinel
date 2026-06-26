"""Unit tests for ATS-slug mining from repo application URLs. Offline; no network
(sources.fetch_repo is monkeypatched).
"""
from src import harvest


def test_greenhouse_slug():
    m = harvest.PATTERNS["greenhouse"].search("https://boards.greenhouse.io/airbnb/jobs/123")
    assert m and m.group(1) == "airbnb"


def test_greenhouse_embed_url_slug():
    m = harvest.PATTERNS["greenhouse"].search(
        "https://boards.greenhouse.io/embed/job_app?for=stripe&token=1")
    assert m and m.group(1) == "stripe"


def test_lever_slug():
    m = harvest.PATTERNS["lever"].search("https://jobs.lever.co/figma/abc-def")
    assert m and m.group(1) == "figma"


def test_ashby_slug():
    m = harvest.PATTERNS["ashby"].search("https://jobs.ashbyhq.com/notion/xyz")
    assert m and m.group(1) == "notion"


def test_bamboohr_slug():
    m = harvest.PATTERNS["bamboohr"].search("https://acme.bamboohr.com/careers/1")
    assert m and m.group(1) == "acme"


def test_harvest_dedupes_blocks_and_ignores_unknown_hosts(monkeypatch):
    fake = {
        "repoA": [
            {"url": "https://boards.greenhouse.io/airbnb/jobs/1"},
            {"url": "https://boards.greenhouse.io/airbnb/jobs/2"},   # duplicate slug
            {"url": "https://jobs.lever.co/figma/x"},
            {"url": "https://example.com/not-an-ats"},               # no slug
        ]
    }
    monkeypatch.setattr(harvest.sources, "fetch_repo", lambda repo: fake[repo])
    out = harvest.harvest({"repos": ["repoA"]})
    assert out["greenhouse"] == ["airbnb"]   # deduped
    assert out["lever"] == ["figma"]
    assert out["ashby"] == []                # nothing matched


def test_harvest_excludes_known_dead_slugs(monkeypatch):
    fake = {"repoA": [
        {"url": "https://jobs.ashbyhq.com/flagright/x"},   # known-dead
        {"url": "https://jobs.ashbyhq.com/notion/y"},      # live
    ]}
    monkeypatch.setattr(harvest.sources, "fetch_repo", lambda repo: fake[repo])
    out = harvest.harvest({"repos": ["repoA"]})
    assert "flagright" not in out["ashby"]
    assert out["ashby"] == ["notion"]
