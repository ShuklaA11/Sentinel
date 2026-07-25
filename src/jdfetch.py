"""Fetch the full job-description body for a single listing, per ATS.

`fetch_jd` returns cleaned plaintext of the JD body, or '' on ANY failure
(logged, never raised) — mirroring src/sources.py's graceful style so a dead
slug or network blip degrades to empty text instead of crashing a caller.

Per-source strategy:
  greenhouse — parse the numeric job id from the url, hit the boards API and
               clean the HTML 'content' field.
  lever      — parse slug+id from the url, hit the postings API and concatenate
               the human description fields, then clean.
  ashby      — no reliable per-posting text API; fetch the public posting HTML
               and strip tags (best-effort).
  other      — unknown source: fetch the url HTML and strip tags (best-effort).
"""
from __future__ import annotations

import html
import logging
import re
import urllib.parse

import requests

log = logging.getLogger("jdfetch")

TIMEOUT = 15
HEADERS = {"User-Agent": "job-applier/0.1 (personal internship tracker)"}

# Closing tags that imply a line break when flattening HTML to text.
_BLOCK_CLOSE = re.compile(r"(?i)</(p|div|li|h[1-6]|tr|ul|ol|section|article|br)\s*>")
_BR = re.compile(r"(?i)<\s*br\s*/?>")
_SCRIPT_STYLE = re.compile(r"(?is)<(script|style)\b[^>]*>.*?</\1>")
_TAG = re.compile(r"<[^>]+>")
_GH_JOB_ID = re.compile(r"/jobs/(\d+)")


def _html_to_text(html_str: str) -> str:
    """Flatten an HTML fragment to tidy plaintext.

    Entities are unescaped first so entity-encoded markup (Greenhouse ships its
    'content' field that way) becomes real tags we can then strip. Script/style
    blocks are dropped whole, block-level closings become newlines, remaining
    tags are removed, and intra-line whitespace collapses to single spaces.
    """
    if not html_str:
        return ""
    text = html.unescape(html_str)
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _BR.sub("\n", text)
    text = _BLOCK_CLOSE.sub("\n", text)
    text = _TAG.sub(" ", text)
    out_lines = []
    for ln in text.split("\n"):
        ln = re.sub(r"\s+", " ", ln)  # collapse runs of whitespace
        ln = re.sub(r"\s+([.,;:!?)])", r"\1", ln)  # drop stray space before punctuation
        ln = ln.strip()
        if ln:
            out_lines.append(ln)
    return "\n".join(out_lines)


def _slug_and_id(url: str, source: str) -> tuple[str, str]:
    """Parse the ATS board slug and posting id from an ATS url.

    Returns ('', '') components that aren't parseable — callers treat a missing
    id as 'cannot fetch' and bail to ''.
    """
    if not url:
        return "", ""
    parsed = urllib.parse.urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    query = urllib.parse.parse_qs(parsed.query)

    if source == "greenhouse":
        job_id = ""
        m = _GH_JOB_ID.search(parsed.path)
        if m:
            job_id = m.group(1)
        elif "gh_jid" in query:
            job_id = query["gh_jid"][0]
        elif "token" in query:  # embed/job_app?for=slug&token=id
            job_id = query["token"][0]
        slug = ""
        if "for" in query:  # embedded board: ?for=slug
            slug = query["for"][0]
        elif "jobs" in parts:
            i = parts.index("jobs")
            if i > 0:
                slug = parts[i - 1]
        return slug, job_id

    if source == "lever":
        slug = parts[0] if parts else ""
        rest = [p for p in parts[1:] if p != "apply"]  # trailing /apply
        job_id = rest[0] if rest else ""
        return slug, job_id

    return "", ""


def _get_json(url: str):
    """GET returning parsed JSON, or None on any failure (logged)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        log.warning("jd request failed %s: %s", url, exc)
        return None
    if resp.status_code != 200:
        log.warning("jd HTTP %s %s", resp.status_code, url)
        return None
    try:
        return resp.json()
    except ValueError:
        log.warning("jd non-JSON response %s", url)
        return None


def _get_html(url: str) -> str:
    """GET returning raw response text, or '' on any failure (logged)."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        log.warning("jd request failed %s: %s", url, exc)
        return ""
    if resp.status_code != 200:
        log.warning("jd HTTP %s %s", resp.status_code, url)
        return ""
    return resp.text


def _fetch_greenhouse_jd(company: str, url: str) -> str:
    slug, job_id = _slug_and_id(url, "greenhouse")
    slug = slug or company  # company is the board slug for greenhouse listings
    if not slug or not job_id:
        log.warning("jd greenhouse: unparseable slug/id from %s", url)
        return ""
    api = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs/{job_id}?questions=false"
    data = _get_json(api)
    if not isinstance(data, dict):
        return ""
    return _html_to_text(data.get("content") or "")


def _fetch_lever_jd(company: str, url: str) -> str:
    slug, job_id = _slug_and_id(url, "lever")
    slug = slug or company
    if not slug or not job_id:
        log.warning("jd lever: unparseable slug/id from %s", url)
        return ""
    api = f"https://api.lever.co/v0/postings/{slug}/{job_id}?mode=json"
    data = _get_json(api)
    if not isinstance(data, dict):
        return ""
    parts: list[str] = []
    body = data.get("descriptionPlain") or data.get("description")
    if body:
        parts.append(body)
    for lst in data.get("lists") or []:
        if not isinstance(lst, dict):
            continue
        if lst.get("text"):
            parts.append(lst["text"])
        if lst.get("content"):
            parts.append(lst["content"])
    extra = data.get("additionalPlain") or data.get("additional")
    if extra:
        parts.append(extra)
    return _html_to_text("\n".join(parts))


def fetch_jd(source: str, company: str, url: str, listing_id: str = "") -> str:
    """Return cleaned plaintext of a listing's JD body, or '' on any failure.

    Never raises: every path is wrapped so a caller iterating over listings can
    treat '' as 'no description available' and move on.
    """
    try:
        s = (source or "").lower()
        if s == "greenhouse":
            return _fetch_greenhouse_jd(company, url)
        if s == "lever":
            return _fetch_lever_jd(company, url)
        # ashby has no reliable per-posting text API; unknown/other sources
        # likewise fall back to fetching the posting HTML and stripping tags.
        if not url:
            return ""
        return _html_to_text(_get_html(url))
    except Exception as exc:  # never let a JD fetch crash the caller
        log.warning("jd fetch failed source=%s url=%s: %s", source, url, exc)
        return ""
