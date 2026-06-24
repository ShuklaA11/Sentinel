"""Dedupe against a seen-set and append new hits to a CSV tracker."""
from __future__ import annotations

import csv
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEEN_PATH = os.path.join(DATA_DIR, "seen.json")
CSV_PATH = os.path.join(DATA_DIR, "listings.csv")

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


def append_csv(new_listings: list[dict], now_iso: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
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
