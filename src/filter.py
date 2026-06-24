"""Keep only listings that look like internships in our tracks + season window."""
from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=None)
def _pattern(term: str) -> "re.Pattern":
    """Word-boundary match so 'intern' doesn't hit 'internal'/'international',
    and 'ai'/'ml'/'swe' don't hit substrings of larger words."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])")


def _has_any(text: str, terms) -> bool:
    return any(_pattern(t).search(text) for t in terms)


def _track_for(title_lc: str, tracks: dict) -> str | None:
    for track, kws in tracks.items():
        if _has_any(title_lc, kws):
            return track
    return None


def matches(listing: dict, filters: dict) -> str | None:
    """Return the track name if the listing passes all filters, else None."""
    title_lc = listing["title"].lower()
    if not title_lc:
        return None

    # Must be an internship.
    if not _has_any(title_lc, filters["intern_terms"]):
        return None

    # Exclude senior/non-student roles.
    if _has_any(title_lc, filters.get("exclude_terms", [])):
        return None

    # Must hit a track.
    track = _track_for(title_lc, filters["tracks"])
    if track is None:
        return None

    # Season/year gate: pass if title mentions an allowed year OR a season phrase,
    # or if the title carries no year at all (many intern posts omit it).
    years = filters.get("allowed_years", [])
    seasons = [s.lower() for s in filters.get("seasons", [])]
    has_any_year = any(str(y) in title_lc for y in ("2023", "2024", "2025", "2026", "2027", "2028"))
    year_ok = (not has_any_year) or any(str(y) in title_lc for y in years)
    season_ok = (not seasons) or _has_any(title_lc, seasons) or not has_any_year
    if not (year_ok or season_ok):
        return None

    # Location gate (empty list = accept all).
    locs = filters.get("locations") or []
    if locs and not _has_any(listing["location"].lower(), locs):
        return None

    return track


def apply_filters(listings: list[dict], filters: dict) -> list[dict]:
    kept = []
    for l in listings:
        track = matches(l, filters)
        if track:
            kept.append({**l, "track": track})
    return kept
