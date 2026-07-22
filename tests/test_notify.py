"""Unit tests for the email rendering helpers (source label, relative posted date,
and that the HTML renders with the metadata). Pure; no network. `now` is injected
so the date logic is deterministic.
"""
from datetime import datetime, timezone

from src import notify


def test_source_label_maps_known_providers_and_repos():
    assert notify._source_label("greenhouse") == "Greenhouse"
    assert notify._source_label("lever") == "Lever"
    assert notify._source_label("repo:SimplifyJobs/Summer2026-Internships") == "SimplifyJobs"
    assert notify._source_label("") == ""


def test_posted_label_is_relative_and_robust():
    now = datetime(2026, 6, 29, 12, 0, tzinfo=timezone.utc)
    assert notify._posted_label("2026-06-29T09:00:00+00:00", now) == "posted today"
    assert notify._posted_label("2026-06-28T09:00:00+00:00", now) == "posted 1d ago"
    assert notify._posted_label("2026-06-26T09:00:00+00:00", now) == "posted 3d ago"
    assert notify._posted_label("", now) == ""
    assert notify._posted_label("not-a-date", now) == ""   # unparseable -> empty, no crash


def test_digest_html_includes_source_and_posted_metadata():
    html = notify._digest_html([
        {"id": "1", "track": "ml", "score": 92, "company": "Anthropic", "title": "ML Intern",
         "location": "SF", "fit_reason": "strong fit", "source": "greenhouse",
         "posted_at": "2026-06-28T09:00:00+00:00", "url": "https://x/1"},
    ])
    assert "Anthropic" in html
    assert "Greenhouse" in html
    assert "posted" in html


def test_send_apply_ready_subject_and_delegates_to_send(monkeypatch):
    captured = {}

    def fake_send(subject, rows, high_fit=False):
        captured["subject"] = subject
        captured["rows"] = rows
        captured["high_fit"] = high_fit
        return True

    monkeypatch.setattr(notify, "_send", fake_send)
    rows = [{"track": "ml", "company": "Faire", "title": "Data Science Intern",
             "location": "SF", "score": 88, "source": "greenhouse", "url": "https://x/1"}]
    assert notify.send_apply_ready(rows) is True
    assert "1 application filled" in captured["subject"]
    assert captured["high_fit"] is True
    # pluralizes
    monkeypatch.setattr(notify, "_send", fake_send)
    notify.send_apply_ready(rows * 2)
    assert "2 applications filled" in captured["subject"]
