"""Apply queue: the outbox of listings selected for application.

This is the state ledger the (shadow / auto) apply driver reads and writes. It is
deliberately separate from listings.csv (the permanent record of everything ever
*detected*) — this file tracks only what we've decided to *apply* to and how far
each one got.

State machine per row:
    queued -> prepared -> submitted        happy path
    queued -> prepared -> needs_human      unknown screening q / CAPTCHA / login wall
    queued -> failed                       form unreachable or driver error
    queued -> skipped                      manual drop / below threshold

Enqueue is append-only and deduped on the listing id (never apply twice). Status
transitions rewrite the file — volume is tiny (tens of rows), so a full rewrite is
simpler and safer than in-place seeking. Per the repo's immutability rule, the
update helper returns a new list; callers save it explicitly.
"""
from __future__ import annotations

import csv
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
QUEUE_PATH = os.path.join(DATA_DIR, "apply_queue.csv")

# One row per listing we intend to apply to. Ordered for readable CSV output.
APPLY_FIELDS = [
    "id",            # listing id — dedup key (matches listings.csv `id`)
    "score",         # fit score carried over from ranking
    "track",         # ml / ai / swe / product / data
    "company",
    "title",
    "location",
    "url",           # where the driver starts (job page or apply form)
    "source",        # detection source (greenhouse / lever / repo:... )
    "ats",           # derived apply surface — drives which fill strategy runs
    "status",        # queued | prepared | submitted | needs_human | failed | skipped
    "attempts",      # times the driver has tried this row
    "queued_at",
    "prepared_at",
    "submitted_at",
    "screenshot",    # audit trail: path to filled-form / confirmation capture
    "note",          # why needs_human / failure reason / anything to eyeball
]

# ATS surfaces we can fill directly. Anything else (community-repo listings whose
# real form isn't known until the url is opened) is "unknown" until the driver
# resolves it live.
_ATS_NAMES = {"greenhouse", "lever", "ashby", "smartrecruiters", "workable", "bamboohr", "workday"}

_INT_FIELDS = ("score", "attempts")


def ats_from(source: str) -> str:
    """Best-guess apply surface from a detection source string.

    Direct pollers carry their ATS name ('greenhouse', 'lever:slug' -> 'lever').
    Repo-sourced listings ('repo:jobright-ai/...') route through a job board and
    only reveal their real ATS once opened, so they're 'unknown' here.
    """
    head = (source or "").split(":", 1)[0].strip().lower()
    return head if head in _ATS_NAMES else "unknown"


def _blank_row() -> dict:
    return {f: "" for f in APPLY_FIELDS}


def _coerce(row: dict) -> dict:
    """Return a copy with int-like fields coerced back to int (or '' if blank)."""
    out = dict(row)
    for f in _INT_FIELDS:
        s = (out.get(f) or "").strip() if isinstance(out.get(f), str) else out.get(f)
        if isinstance(s, str):
            out[f] = int(s) if s.lstrip("-").isdigit() else ""
    return out


def load_queue() -> list[dict]:
    """Read the queue. Missing file -> []. score/attempts coerced to int."""
    if not os.path.exists(QUEUE_PATH):
        return []
    with open(QUEUE_PATH, newline="") as f:
        return [_coerce(r) for r in csv.DictReader(f)]


def save_queue(rows: list[dict]) -> None:
    """Overwrite the queue with `rows` (header + all rows)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(QUEUE_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=APPLY_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in APPLY_FIELDS})


def queued_ids() -> set:
    """Ids already in the queue, for dedup before enqueueing."""
    return {r["id"] for r in load_queue() if r.get("id")}


def enqueue(new_listings: list[dict], now_iso: str) -> int:
    """Append listings not already queued. Returns the number added.

    Each listing is a normalized dict (as produced by sources/ranking): at least
    id, company, title, url, source; score/track/location optional.
    """
    have = queued_ids()
    fresh = [l for l in new_listings if l.get("id") and l["id"] not in have]
    if not fresh:
        return 0
    rows = load_queue()
    for l in fresh:
        row = _blank_row()
        row.update({
            "id": l.get("id", ""),
            "score": l.get("score", ""),
            "track": l.get("track", ""),
            "company": l.get("company", ""),
            "title": l.get("title", ""),
            "location": l.get("location", ""),
            "url": l.get("url", ""),
            "source": l.get("source", ""),
            "ats": ats_from(l.get("source", "")),
            "status": "queued",
            "attempts": 0,
            "queued_at": now_iso,
        })
        rows.append(row)
    save_queue(rows)
    return len(fresh)


def update_row(rows: list[dict], id_: str, **changes) -> list[dict]:
    """Return a NEW list with the row matching id_ updated by `changes`.

    Pure — does not mutate `rows` or its dicts (immutability rule). Unknown ids
    pass through unchanged.
    """
    out = []
    for r in rows:
        if r.get("id") == id_:
            out.append({**r, **changes})
        else:
            out.append(dict(r))
    return out
