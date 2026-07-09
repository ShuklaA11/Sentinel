"""Fit-score new listings 0–100 against the user's profile via Claude (Haiku).

The model rates four dimensions per listing and flags hard deal-breakers; the composite
0–100 score is computed deterministically in Python (weighted mean), and a veto forces the
score to 0 so a disqualifying-but-shiny role can't sneak into the high-fit alert.

Graceful: if ANTHROPIC_API_KEY is unset, the `anthropic` package is missing, or the
profile is unfilled, scoring is skipped (listings pass through unscored) so the core
detect→alert pipeline never breaks. Scoring activates once the key + profile exist.
"""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger("rank")

MODEL = "claude-haiku-4-5"
BATCH = 25  # listings per API call

# Fit dimensions and their weights (must sum to 1.0). track/skill dominate because a
# 3-month internship is chosen on role-match and skill-overlap; growth is a tiebreaker.
WEIGHTS = {"track": 0.35, "skill": 0.30, "logistics": 0.20, "growth": 0.15}
DIMS = tuple(WEIGHTS)


def _profile_ready(profile: dict) -> bool:
    bg = (profile.get("background") or "")
    return "TODO" not in bg and bool(profile.get("preferences"))


def _prompt(profile: dict, batch: list[dict]) -> str:
    prefs = json.dumps(profile.get("preferences", {}), indent=0)
    items = "\n".join(f'{i}. {l["company"]} — {l["title"]} ({l["location"]})'
                      for i, l in enumerate(batch))
    return (
        f"Candidate background:\n{profile.get('background','').strip()}\n\n"
        f"Preferences (tracks, skills, locations, dealbreakers, nice-to-haves):\n{prefs}\n\n"
        "Score each internship for fit. Rate FOUR dimensions 0–100 and flag hard deal-breakers:\n"
        "- track: role matches the candidate's target tracks\n"
        "- skill: role overlaps the candidate's real skills\n"
        "- logistics: location / remote / season workability (soft preference)\n"
        "- growth: learning, mentorship, brand / resume value\n"
        'Set "veto": true ONLY for a hard deal-breaker — wrong season, a full-time (non-intern) '
        "role, a location the candidate cannot take, or work authorization they lack. A veto "
        "overrides the dimensions.\n\n"
        f"Listings:\n{items}\n\n"
        'Return ONLY a JSON array: [{"i":<index>,"track":<0-100>,"skill":<0-100>,'
        '"logistics":<0-100>,"growth":<0-100>,"veto":<true|false>,"reason":"<≤8 words>"}]'
    )


def _composite(row: dict) -> int:
    """Weighted-mean of the four dimension scores, clamped to 0–100."""
    score = round(sum(WEIGHTS[d] * float(row[d]) for d in DIMS))
    return max(0, min(100, score))


def _parse_batch(text: str) -> dict:
    """Parse the model's JSON array into {index: (score, reason)}.

    Pure (no network) so it's unit-testable. A veto forces score 0 and a "VETO:" reason;
    a malformed row is skipped, leaving that listing unscored downstream.
    """
    text = text[text.find("["): text.rfind("]") + 1]  # trim any prose around the array
    out: dict = {}
    for row in json.loads(text):
        try:
            i = int(row["i"])
            reason = str(row.get("reason", ""))[:60].strip()
            if row.get("veto"):
                out[i] = (0, f"VETO: {reason}" if reason else "VETO")
            else:
                out[i] = (_composite(row), reason)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed — skipping scoring")
        return None
    return anthropic.Anthropic(api_key=key)


def _score_batch(client, profile: dict, batch: list[dict]) -> dict:
    """Return {index: (score, reason)} for one batch."""
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=1536,
            messages=[{"role": "user", "content": _prompt(profile, batch)}],
        )
        return _parse_batch(resp.content[0].text.strip())
    except Exception as exc:  # noqa: BLE001 — never let scoring crash the run
        log.error("scoring batch failed: %s", exc)
        return {}


def score_listings(new: list[dict], profile: dict) -> list[dict]:
    """Add `score` (int or '') and `fit_reason` to each listing."""
    if not new:
        return new
    client = _client()
    if client is None or not _profile_ready(profile):
        if client and not _profile_ready(profile):
            log.warning("profile not filled — skipping scoring")
        return [{**l, "score": "", "fit_reason": ""} for l in new]

    scored = []
    for start in range(0, len(new), BATCH):
        batch = new[start:start + BATCH]
        results = _score_batch(client, profile, batch)
        for i, l in enumerate(batch):
            score, reason = results.get(i, ("", ""))
            scored.append({**l, "score": score, "fit_reason": reason})
    log.info("scored %d listings", len(scored))
    return scored


def partition_by_fit(listings: list[dict], threshold: int) -> tuple[list[dict], list[dict]]:
    """Split into (high_fit >= threshold, rest). Unscored listings ('' score) go to rest.

    The isinstance check matters: an unscored listing has score == "" (str), and
    "" >= 85 raises TypeError in Python 3 — so the guard both avoids the crash and
    correctly keeps unknown-fit listings out of the high-fit alert.
    """
    high = [l for l in listings if isinstance(l.get("score"), int) and l["score"] >= threshold]
    high_ids = {l["id"] for l in high}
    rest = [l for l in listings if l["id"] not in high_ids]
    return high, rest
