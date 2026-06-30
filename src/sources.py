"""Fetch internship listings from ATS endpoints and community repos.

Every fetcher returns a list of normalized listing dicts:
    {id, source, company, title, location, url, posted_at}
A dead slug or network blip must never crash the run — fetchers log and return [].
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from concurrent.futures import ThreadPoolExecutor

import requests

# Epoch values at/above this are milliseconds, not seconds (1e12 s ≈ year 33658,
# so any real date in ms is >= 1e12 while a date in seconds is < 1e12).
_EPOCH_MS_CUTOFF = 1_000_000_000_000

log = logging.getLogger("sources")

TIMEOUT = 20
MAX_WORKERS = 16
HEADERS = {"User-Agent": "job-applier/0.1 (personal internship tracker)"}


def _get(url: str, method: str = "GET", json_body: dict | None = None):
    """Request returning parsed JSON, or None on any failure (logged)."""
    try:
        resp = requests.request(method, url, headers=HEADERS, json=json_body, timeout=TIMEOUT)
    except requests.RequestException as exc:
        log.warning("request failed %s: %s", url, exc)
        return None
    if resp.status_code != 200:
        log.warning("HTTP %s %s", resp.status_code, url)
        return None
    try:
        return resp.json()
    except ValueError:
        log.warning("non-JSON response %s", url)
        return None


def _iso(ts) -> str:
    """Normalize a source timestamp to an ISO-8601 UTC string, or '' if it
    isn't a real instant.

    Fetchers emit four shapes: ISO strings (greenhouse/ashby/smartrecruiters),
    epoch milliseconds (lever), epoch seconds (simplify repos), and relative
    text like 'Jun 23' (jobright readme). Storing one contract — ISO or '' —
    lets the digest render 'posted Nd ago' and makes detection-latency math
    possible downstream.
    """
    if ts is None or ts == "" or ts == 0:
        return ""
    s = str(ts).strip()
    if s.lstrip("-").isdigit():  # epoch seconds or milliseconds
        n = int(s)
        if abs(n) >= _EPOCH_MS_CUTOFF:
            n /= 1000
        try:
            return dt.datetime.fromtimestamp(n, tz=dt.timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return ""
    try:
        d = dt.datetime.fromisoformat(s)
    except ValueError:
        return ""  # relative/unparseable text — not a real timestamp
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc).isoformat()


def _norm(id_, source, company, title, location, url, posted_at):
    return {
        "id": f"{source}:{id_}",
        "source": source,
        "company": company,
        "title": (title or "").strip(),
        "location": (location or "").strip(),
        "url": url or "",
        "posted_at": _iso(posted_at),
    }


def fetch_greenhouse(slug: str) -> list[dict]:
    data = _get(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs")
    if not data or "jobs" not in data:
        return []
    out = []
    for j in data["jobs"]:
        loc = (j.get("location") or {}).get("name", "")
        out.append(_norm(j.get("id"), "greenhouse", slug, j.get("title"),
                         loc, j.get("absolute_url"), j.get("updated_at")))
    return out


def fetch_lever(slug: str) -> list[dict]:
    data = _get(f"https://api.lever.co/v0/postings/{slug}?mode=json")
    if not isinstance(data, list):
        return []
    out = []
    for j in data:
        cats = j.get("categories") or {}
        out.append(_norm(j.get("id"), "lever", slug, j.get("text"),
                         cats.get("location"), j.get("hostedUrl"), j.get("createdAt")))
    return out


def fetch_ashby(slug: str) -> list[dict]:
    data = _get(f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
    if not data or "jobs" not in data:
        return []
    out = []
    for j in data["jobs"]:
        url = j.get("jobUrl") or j.get("applyUrl") or ""
        out.append(_norm(j.get("id"), "ashby", slug, j.get("title"),
                         j.get("location"), url, j.get("publishedAt")))
    return out


def fetch_smartrecruiters(slug: str) -> list[dict]:
    data = _get(f"https://api.smartrecruiters.com/v1/companies/{slug}/postings?limit=100")
    if not data or "content" not in data:
        return []
    out = []
    for j in data["content"]:
        loc = j.get("location") or {}
        loc_str = ", ".join(p for p in (loc.get("city"), loc.get("region"), loc.get("country")) if p)
        url = f"https://jobs.smartrecruiters.com/{slug}/{j.get('id')}"
        out.append(_norm(j.get("id"), "smartrecruiters", slug, j.get("name"),
                         loc_str, url, j.get("releasedDate")))
    return out


def fetch_workable(slug: str) -> list[dict]:
    data = _get(f"https://apply.workable.com/api/v1/widget/accounts/{slug}?details=true")
    jobs = (data or {}).get("jobs") if isinstance(data, dict) else None
    if not jobs:
        return []
    out = []
    for j in jobs:
        loc = j.get("location") or {}
        if isinstance(loc, dict):
            loc_str = ", ".join(p for p in (loc.get("city"), loc.get("region"), loc.get("country")) if p)
        else:
            loc_str = str(loc)
        url = j.get("url") or j.get("application_url") or (
            f"https://apply.workable.com/{slug}/j/{j.get('shortcode')}" if j.get("shortcode") else "")
        out.append(_norm(j.get("id") or j.get("shortcode"), "workable", slug,
                         j.get("title"), loc_str, url, j.get("published_on") or j.get("created_at")))
    return out


def fetch_bamboohr(slug: str) -> list[dict]:
    data = _get(f"https://{slug}.bamboohr.com/careers/list")
    res = (data or {}).get("result") if isinstance(data, dict) else None
    if not res:
        return []
    out = []
    for j in res:
        loc = j.get("atsLocation") or {}
        loc_str = ", ".join(p for p in (loc.get("city"), loc.get("state"), loc.get("country")) if p)
        if not loc_str and j.get("isRemote"):
            loc_str = "Remote"
        url = f"https://{slug}.bamboohr.com/careers/{j.get('id')}"
        out.append(_norm(j.get("id"), "bamboohr", slug, j.get("jobOpeningName"), loc_str, url, ""))
    return out


def fetch_workday(cfg: dict) -> list[dict]:
    """Workday needs host + tenant + site (discover once from the careers URL)."""
    host, tenant, site = cfg.get("host"), cfg.get("tenant"), cfg.get("site")
    name = cfg.get("name", tenant)
    api = f"https://{host}/wday/cxs/{tenant}/{site}/jobs"
    out: list[dict] = []
    offset = 0
    while offset < 200:  # cap at 200 to stay polite
        data = _get(api, method="POST",
                    json_body={"appliedFacets": {}, "limit": 20, "offset": offset, "searchText": ""})
        postings = (data or {}).get("jobPostings") if isinstance(data, dict) else None
        if not postings:
            break
        for j in postings:
            path = j.get("externalPath", "")
            url = f"https://{host}/en-US/{site}{path}" if path else ""
            out.append(_norm(path or j.get("title"), "workday", name, j.get("title"),
                             j.get("locationsText"), url, j.get("postedOn")))
        if len(postings) < 20:
            break
        offset += 20
    return out


_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def fetch_repo_readme(repo: dict) -> list[dict]:
    """Parse a markdown-table README (JobRight format).

    Row: | **[Company](url)** | **[Title](apply_url)** | Location | Work Model | Date |
    A leading-cell '↳' means 'same company as the row above'.
    """
    try:
        resp = requests.get(repo["url"], headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        log.warning("request failed %s: %s", repo["url"], exc)
        return []
    if resp.status_code != 200:
        log.warning("HTTP %s %s", resp.status_code, repo["url"])
        return []

    out: list[dict] = []
    last_company = ""
    for line in resp.text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0].lower() in ("company", "-----", "") and "company" in line.lower():
            continue  # header
        # Company (col 0): first markdown link text, or ↳ -> reuse previous.
        if cells[0] in ("↳", "->") or not cells[0]:
            company = last_company
        else:
            m = _MD_LINK.search(cells[0])
            company = (m.group(1) if m else cells[0]).strip("* ")
        last_company = company or last_company
        # Title + apply url (col 1).
        m = _MD_LINK.search(cells[1])
        if not m:
            continue
        title, url = m.group(1).strip("* "), m.group(2)
        # Stable dedup id: drop the query string from the jobright url.
        stable = url.split("?", 1)[0]
        location = cells[2] if len(cells) > 2 else ""
        posted = cells[4] if len(cells) > 4 else ""
        out.append(_norm(stable, f"repo:{repo['name']}", company, title, location, url, posted))
    return out


def fetch_repo(repo: dict) -> list[dict]:
    """Community repos: structured listings.json (Simplify) or markdown README (JobRight)."""
    kind = repo.get("kind")
    if kind == "readme":
        return fetch_repo_readme(repo)
    if kind != "simplify_json":
        log.warning("unsupported repo kind %s", kind)
        return []
    data = _get(repo["url"])
    if not isinstance(data, list):
        return []
    out = []
    for j in data:
        # Simplify schema: company_name, title, locations[], url, date_posted, active, season
        if j.get("active") is False:
            continue
        loc = ", ".join(j.get("locations") or []) if isinstance(j.get("locations"), list) else j.get("locations", "")
        out.append(_norm(j.get("id") or j.get("url"), f"repo:{repo['name']}",
                         j.get("company_name"), j.get("title"), loc,
                         j.get("url"), j.get("date_posted") or j.get("date_updated")))
    return out


ATS_FETCHERS = {
    "greenhouse": fetch_greenhouse,
    "lever": fetch_lever,
    "ashby": fetch_ashby,
    "smartrecruiters": fetch_smartrecruiters,
    "workable": fetch_workable,
    "bamboohr": fetch_bamboohr,
}


def fetch_all(companies: dict) -> tuple[list[dict], dict]:
    """Run every configured source concurrently. Returns (listings, stats-per-source)."""
    # Build (label, thunk) tasks across all source types.
    tasks: list[tuple[str, callable]] = []
    for provider, fetcher in ATS_FETCHERS.items():
        for slug in companies.get(provider, []) or []:
            tasks.append((f"{provider}:{slug}", lambda f=fetcher, s=slug: f(s)))
    for cfg in companies.get("workday", []) or []:
        label = f"workday:{cfg.get('name', cfg.get('tenant'))}"
        tasks.append((label, lambda c=cfg: fetch_workday(c)))
    for repo in companies.get("repos", []) or []:
        tasks.append((repo["name"], lambda r=repo: fetch_repo(r)))

    listings: list[dict] = []
    stats: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = pool.map(lambda t: (t[0], t[1]()), tasks)
        for label, got in results:
            stats[label] = len(got)
            listings.extend(got)
    return listings, stats
