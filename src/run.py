"""Orchestrator: fetch -> filter -> dedupe -> persist. Run: python -m src.run"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os

import yaml

from . import sources, filter as filt, store, notify, rank

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


def _load(name: str) -> dict:
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_profile() -> dict:
    path = os.path.join(os.path.dirname(CONFIG_DIR), "profile", "profile.yml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge_companies() -> dict:
    """Curated companies.yml + auto-harvested slugs (union, deduped)."""
    companies = _load("companies.yml")
    harvested = _load("harvested.yml")
    for provider, slugs in harvested.items():
        merged = dict.fromkeys(companies.get(provider, []) or [])  # preserve order, dedupe
        merged.update(dict.fromkeys(slugs))
        companies[provider] = list(merged)
    return companies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="don't write seen-set / csv; just report")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    companies = _merge_companies()
    filters = _load("filters.yml")

    raw, stats = sources.fetch_all(companies)
    kept = filt.apply_filters(raw, filters)

    seen = store.load_seen()
    new = store.split_new(kept, seen)

    # Fit-score new listings (no-ops without API key / filled profile).
    profile = _load_profile()
    new = rank.score_listings(new, profile)

    # Report.
    dead = [k for k, v in stats.items() if v == 0]
    print(f"\n=== sources: {len(stats)} polled, {sum(stats.values())} raw listings ===")
    if dead:
        print(f"  dead/empty ({len(dead)}): {', '.join(dead)}")
    print(f"=== {len(kept)} match filters · {len(new)} are NEW ===\n")

    by_track: dict[str, int] = {}
    for l in new:
        by_track[l["track"]] = by_track.get(l["track"], 0) + 1
    for l in sorted(new, key=lambda x: (-(x.get("score") or 0), x["track"], x["company"])):
        sc = f"{l['score']:>3}" if l.get("score") != "" else "  ·"
        print(f"  {sc} [{l['track']:>7}] {l['company']:<18} {l['title'][:55]:<55} {l['location'][:22]}")
    if by_track:
        print("\n  new by track:", ", ".join(f"{k}={v}" for k, v in sorted(by_track.items())))

    if args.dry_run:
        print("\n(dry-run: nothing written)")
        return

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    store.append_csv(new, now)
    seen.update(l["id"] for l in kept)
    store.save_seen(seen)
    print(f"\nwrote {len(new)} new rows -> data/listings.csv · seen-set now {len(seen)}")

    threshold = filters.get("high_fit_threshold", 85)
    high, rest = rank.partition_by_fit(new, threshold)
    if high and notify.send_high_fit_alert(high):
        print(f"🔥 alerted {len(high)} high-fit (score >= {threshold})")
    if rest and notify.send_digest(rest):
        print(f"emailed digest of {len(rest)} new listings")


if __name__ == "__main__":
    main()
