"""Unit tests for the per-ATS job-description fetcher. Offline; no network.

Every network path is exercised by monkeypatching ``requests.get`` to return a
canned response object, so these tests never touch the wire. The pure helpers
(`_html_to_text`, `_slug_and_id`) are tested directly.
"""
import requests

from src import jdfetch


class FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code: int = 200, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


# --- per-source network paths (monkeypatched) --------------------------------


def test_greenhouse_content_html_is_cleaned(monkeypatch):
    # Greenhouse returns the JD as an HTML-entity-encoded 'content' field.
    captured = {}

    def fake_get(url, **kw):
        captured["url"] = url
        return FakeResp(json_data={
            "content": ("&lt;p&gt;Build ML pipelines with "
                        "&lt;strong&gt;Python&lt;/strong&gt;.&lt;/p&gt;"
                        "&lt;script&gt;evil()&lt;/script&gt;"),
        })

    monkeypatch.setattr(requests, "get", fake_get)
    out = jdfetch.fetch_jd("greenhouse", "acme", "https://boards.greenhouse.io/acme/jobs/1234567")

    assert "Build ML pipelines with Python." in out
    assert "evil()" not in out
    assert "boards-api.greenhouse.io/v1/boards/acme/jobs/1234567" in captured["url"]
    assert "questions=false" in captured["url"]


def test_lever_human_fields_concatenated_and_cleaned(monkeypatch):
    payload = {
        "text": "Software Engineer Intern",
        "descriptionPlain": "Join our backend team.",
        "lists": [
            {"text": "Requirements", "content": "<li>Python</li><li>Go</li>"},
            {"text": "Nice to have", "content": "<li>Rust</li>"},
        ],
        "additionalPlain": "Equal opportunity employer.",
    }
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResp(json_data=payload))
    out = jdfetch.fetch_jd("lever", "acme", "https://jobs.lever.co/acme/abc-123-def")

    assert "Join our backend team." in out
    assert "Requirements" in out
    assert "Python" in out and "Go" in out and "Rust" in out
    assert "Equal opportunity employer." in out


def test_ashby_html_is_tag_stripped(monkeypatch):
    html = ("<html><head><style>.x{color:red}</style></head><body>"
            "<h1>ML Intern</h1><p>Do &amp; learn.</p>"
            "<script>track()</script></body></html>")
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResp(text=html))
    out = jdfetch.fetch_jd("ashby", "acme", "https://jobs.ashbyhq.com/acme/xyz")

    assert "ML Intern" in out
    assert "Do & learn." in out
    assert "track()" not in out
    assert "color:red" not in out


def test_unknown_source_falls_back_to_html_strip(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResp(text="<p>Hello world</p>"))
    out = jdfetch.fetch_jd("repo:some-list", "acme", "https://example.com/job/1")
    assert "Hello world" in out


def test_request_exception_yields_empty_never_raises(monkeypatch):
    def boom(url, **kw):
        raise requests.RequestException("network down")

    monkeypatch.setattr(requests, "get", boom)
    assert jdfetch.fetch_jd("greenhouse", "acme", "https://boards.greenhouse.io/acme/jobs/1") == ""
    assert jdfetch.fetch_jd("lever", "acme", "https://jobs.lever.co/acme/abc") == ""
    assert jdfetch.fetch_jd("ashby", "acme", "https://jobs.ashbyhq.com/acme/xyz") == ""


def test_non_200_yields_empty(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResp(status_code=404, text="nope"))
    assert jdfetch.fetch_jd("lever", "acme", "https://jobs.lever.co/acme/abc") == ""


def test_missing_job_id_yields_empty_without_request(monkeypatch):
    def fail(url, **kw):
        raise AssertionError("should not hit network when id is unparseable")

    monkeypatch.setattr(requests, "get", fail)
    assert jdfetch.fetch_jd("greenhouse", "acme", "https://boards.greenhouse.io/acme") == ""


# --- pure: _html_to_text -----------------------------------------------------


def test_html_to_text_drops_script_and_style():
    out = jdfetch._html_to_text(
        "<style>a{color:red}</style><script>bad()</script><p>Keep me</p>")
    assert "Keep me" in out
    assert "bad()" not in out
    assert "color:red" not in out


def test_html_to_text_unescapes_entities():
    assert jdfetch._html_to_text("Tom &amp; Jerry &lt;3") == "Tom & Jerry <3"


def test_html_to_text_strips_tags_and_collapses_whitespace():
    assert jdfetch._html_to_text("<div>  hello   world  </div>") == "hello world"


def test_html_to_text_empty_inputs():
    assert jdfetch._html_to_text("") == ""
    assert jdfetch._html_to_text(None) == ""


# --- pure: _slug_and_id ------------------------------------------------------


def test_slug_and_id_greenhouse_path():
    assert jdfetch._slug_and_id(
        "https://boards.greenhouse.io/acme/jobs/1234567", "greenhouse") == ("acme", "1234567")


def test_slug_and_id_greenhouse_query_params():
    slug, jid = jdfetch._slug_and_id(
        "https://boards.greenhouse.io/embed/job_app?for=acme&gh_jid=987654", "greenhouse")
    assert slug == "acme"
    assert jid == "987654"


def test_slug_and_id_lever():
    assert jdfetch._slug_and_id(
        "https://jobs.lever.co/acme/abc-123-def", "lever") == ("acme", "abc-123-def")


def test_slug_and_id_empty_url():
    assert jdfetch._slug_and_id("", "greenhouse") == ("", "")
    assert jdfetch._slug_and_id("", "lever") == ("", "")
