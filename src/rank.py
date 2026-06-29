"""Fit-score new listings 0–100 against the user's profile via Claude (Haiku).

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
        f"Score each internship 0–100 for fit with this candidate. Penalize dealbreakers "
        f"hard. Reward track/skill/location/nice-to-have matches.\n\nListings:\n{items}\n\n"
        f'Return ONLY a JSON array: [{{"i": <index>, "score": <0-100>, "reason": "<≤8 words>"}}]'
    )


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
            model=MODEL, max_tokens=1024,
            messages=[{"role": "user", "content": _prompt(profile, batch)}],
        )
        text = resp.content[0].text.strip()
        text = text[text.find("["): text.rfind("]") + 1]  # trim any prose
        return {r["i"]: (int(r["score"]), r.get("reason", "")) for r in json.loads(text)}
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
