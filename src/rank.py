"""Fit-score new listings 0–100 against the user's profile via an LLM.

The model rates four dimensions per listing and flags hard deal-breakers; the composite
0–100 score is computed deterministically in Python (weighted mean), and a veto forces the
score to 0 so a disqualifying-but-shiny role can't sneak into the high-fit alert.

Providers run in order — Anthropic (Claude Haiku) primary, OpenAI (GPT) fallback. If a
provider hits an *unavailable* condition (billing/auth/quota) it is dropped for the rest of
the run and the next provider takes over; a transient error just fails that batch. When
every provider is unavailable, `score_listings` returns a loud warning so a dead scorer is
never silent (esp. in CI). With no keys / an unfilled profile, scoring is skipped quietly
(listings pass through unscored) so the core detect→alert pipeline never breaks.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Callable, NamedTuple

log = logging.getLogger("rank")

ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = os.environ.get("SCORE_OPENAI_MODEL", "gpt-4o-mini")
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


# --- providers ---------------------------------------------------------------

class _Provider(NamedTuple):
    name: str
    client: object
    call: Callable[[object, str], str]  # (client, prompt) -> raw model text


def _anthropic_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed — Claude scoring disabled")
        return None
    return anthropic.Anthropic(api_key=key)


def _openai_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        import openai
    except ImportError:
        log.warning("openai package not installed — GPT fallback disabled")
        return None
    return openai.OpenAI(api_key=key)


def _call_anthropic(client, prompt: str) -> str:
    resp = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=1536,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _call_openai(client, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, max_tokens=1536,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _providers() -> list[_Provider]:
    """Available scorers in priority order: Anthropic primary, OpenAI fallback."""
    out: list[_Provider] = []
    a = _anthropic_client()
    if a is not None:
        out.append(_Provider("anthropic", a, _call_anthropic))
    o = _openai_client()
    if o is not None:
        out.append(_Provider("openai", o, _call_openai))
    return out


def _is_unavailable(exc: Exception) -> bool:
    """True when an error means the provider is *out of service* for this run (billing,
    auth, quota) — as opposed to a transient blip. Drives dropping vs. retrying a provider.
    """
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in (
        "credit balance is too low", "insufficient_quota", "billing",
        "exceeded your current quota", "invalid_api_key", "invalid api key",
        "authentication", "account is not active",
    ))


def _short(exc: Exception) -> str:
    return str(exc).splitlines()[0][:120] if str(exc) else exc.__class__.__name__


def _unscored(listings: list[dict]) -> list[dict]:
    return [{**l, "score": "", "fit_reason": ""} for l in listings]


def _score_batch(alive: list[_Provider], profile: dict, batch: list[dict]) -> tuple[dict, str]:
    """Try each still-alive provider until one scores the batch. Providers that hit an
    unavailable error are removed from `alive` (mutated in place). Returns (results, reason)
    where reason is the last unavailable-failure message (for the loud warning)."""
    prompt = _prompt(profile, batch)
    reason = ""
    for prov in list(alive):
        try:
            return _parse_batch(prov.call(prov.client, prompt)), reason
        except Exception as exc:  # noqa: BLE001 — never let scoring crash the run
            if _is_unavailable(exc):
                reason = f"{prov.name}: {_short(exc)}"
                alive.remove(prov)
                log.error("scorer %s unavailable — %s", prov.name, reason)
            else:
                log.error("scorer %s batch failed: %s", prov.name, exc)
    return {}, reason


def score_listings(new: list[dict], profile: dict) -> tuple[list[dict], str | None]:
    """Add `score` (int or '') and `fit_reason` to each listing.

    Returns (listings, warning). `warning` is None on the happy path or an expected skip
    (no keys / unfilled profile); it is a loud, human-readable string only when every
    configured provider became unavailable mid-run (so the caller can surface it).
    """
    if not new:
        return new, None
    providers = _providers()
    if not providers:
        return _unscored(new), None  # no keys configured — expected quiet skip
    if not _profile_ready(profile):
        log.warning("profile not filled — skipping scoring")
        return _unscored(new), None

    alive = list(providers)
    last_reason = ""
    scored: list[dict] = []
    for start in range(0, len(new), BATCH):
        batch = new[start:start + BATCH]
        results, reason = _score_batch(alive, profile, batch)
        if reason:
            last_reason = reason
        for i, l in enumerate(batch):
            score, fit = results.get(i, ("", ""))
            scored.append({**l, "score": score, "fit_reason": fit})
        if not alive:  # every provider is down — remaining batches can't be scored
            scored.extend(_unscored(new[len(scored):]))
            break

    n_scored = sum(1 for l in scored if isinstance(l.get("score"), int))
    log.info("scored %d/%d listings", n_scored, len(new))
    warning = None
    if not alive:
        warning = f"scoring unavailable — {last_reason}"
    return scored, warning


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
