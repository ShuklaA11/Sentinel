"""Dedupe against a seen-set and append new hits to a CSV tracker."""
from __future__ import annotations

import csv
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEEN_PATH = os.path.join(DATA_DIR, "seen.json")
CSV_PATH = os.path.join(DATA_DIR, "listings.csv")
PENDING_PATH = os.path.join(DATA_DIR, "pending.csv")  # outbox: sub-threshold listings awaiting the next batch digest

CSV_FIELDS = ["first_seen", "score", "track", "company", "title", "location", "fit_reason", "source", "url", "status", "id"]


def load_seen() -> set:
    if not os.path.exists(SEEN_PATH):
        return set()
    with open(SEEN_PATH) as f:
        return set(json.load(f))


def save_seen(seen: set) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SEEN_PATH, "w") as f:
        json.dump(sorted(seen), f, indent=0)


def split_new(listings: list[dict], seen: set) -> list[dict]:
    """Return listings whose id is not already seen."""
    return [l for l in listings if l["id"] not in seen]


def _write_rows(path: str, new_listings: list[dict], now_iso: str) -> None:
    """Append listings as CSV rows to `path` (writing the header if the file is new)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        for l in new_listings:
            w.writerow({
                "first_seen": now_iso,
                "score": l.get("score", ""),
                "track": l.get("track", ""),
                "company": l.get("company", ""),
                "title": l.get("title", ""),
                "location": l.get("location", ""),
                "fit_reason": l.get("fit_reason", ""),
                "source": l.get("source", ""),
                "url": l.get("url", ""),
                "status": "new",
                "id": l.get("id", ""),
            })


def append_csv(new_listings: list[dict], now_iso: str) -> None:
    """Append to the permanent tracker (record of everything ever detected)."""
    _write_rows(CSV_PATH, new_listings, now_iso)


def enqueue_pending(new_listings: list[dict], now_iso: str) -> None:
    """Add sub-threshold listings to the digest outbox, to be flushed on a schedule."""
    _write_rows(PENDING_PATH, new_listings, now_iso)


def load_pending() -> list[dict]:
    """Read the outbox. Coerces `score` back to int (or '') so the digest can sort on it."""
    if not os.path.exists(PENDING_PATH):
        return []
    out = []
    with open(PENDING_PATH, newline="") as f:
        for r in csv.DictReader(f):
            s = (r.get("score") or "").strip()
            r["score"] = int(s) if s.lstrip("-").isdigit() else ""
            out.append(r)
    return out


def clear_pending() -> None:
    """Empty the outbox (call only after a digest is confirmed sent)."""
    if os.path.exists(PENDING_PATH):
        os.remove(PENDING_PATH)
