"""Interview prep packs: per-company DSA + verbal/technical + behavioral questions.

DSA  : pulled live from a community LeetCode company-wise repo (sorted by frequency).
Verbal/behavioral : curated bank (config/interview_bank.yml), track-aware.
Personalized      : optional LLM questions about YOUR specific projects + tailored
                    behavioral, generated from profile.yml (graceful skip without key).

Run: python -m src.interview "Stripe" --track ml
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os

import requests
import yaml

log = logging.getLogger("interview")

DSA_BASE = "https://raw.githubusercontent.com/krishnadey30/LeetCode-Questions-CompanyWise/master"
WINDOWS = ("6months", "1year", "2year", "alltime")
ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL = os.environ.get("INTERVIEW_MODEL", "claude-haiku-4-5")


def _slugs(company: str) -> list[str]:
    c = company.strip().lower()
    return list(dict.fromkeys([c.replace(" ", "_"), c.replace(" ", ""), c.replace(" ", "-")]))


def dsa(company: str, top: int = 15) -> tuple[str | None, list[dict]]:
    """Return (window_used, top problems by frequency) or (None, []) if no data."""
    for slug in _slugs(company):
        for window in WINDOWS:
            try:
                resp = requests.get(f"{DSA_BASE}/{slug}_{window}.csv",
                                    headers={"User-Agent": "job-applier/0.1"}, timeout=20)
            except requests.RequestException:
                continue
            if resp.status_code != 200:
                continue
            rows = list(csv.DictReader(io.StringIO(resp.text)))
            if rows:
                return window, rows[:top]
    return None, []


def bank(track: str) -> dict:
    with open(os.path.join(ROOT, "config", "interview_bank.yml")) as f:
        data = yaml.safe_load(f)
    return {
        "behavioral": data.get("behavioral", []),
        "technical": (data.get("technical", {}) or {}).get(track, []),
    }


def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def personalized(profile: dict, company: str) -> dict | None:
    """LLM questions about the candidate's actual projects + tailored behavioral."""
    client = _client()
    background = (profile.get("background") or "")
    if client is None or "TODO" in background or not background:
        return None
    prompt = (
        f"Candidate background:\n{background}\n\n"
        f"They are interviewing at {company}. Generate questions an interviewer would "
        f"actually ask THIS candidate:\n"
        f"- project_qs: 5 deep technical questions about their specific projects "
        f"(probe design decisions, tradeoffs, failure modes).\n"
        f"- behavioral_qs: 5 behavioral questions tailored to their real experience.\n"
        f'Return ONLY JSON: {{"project_qs": [..], "behavioral_qs": [..]}}'
    )
    try:
        resp = client.messages.create(model=MODEL, max_tokens=1024,
                                      messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception as exc:  # noqa: BLE001
        log.error("personalized generation failed: %s", exc)
        return None


def prep(company: str, track: str, profile: dict) -> None:
    print(f"\n{'='*60}\n  INTERVIEW PREP — {company}  (track: {track})\n{'='*60}")

    window, problems = dsa(company)
    print("\n## DSA / Coding")
    if problems:
        print(f"   (top {len(problems)} by frequency, window: {window})")
        for p in problems:
            print(f"   [{p.get('Difficulty',''):<6}] {p.get('Title','')}  {p.get('Leetcode Question Link','').strip()}")
    else:
        print("   No company-specific data (likely a startup). Practice general sets:")
        print("   Blind 75 — https://neetcode.io/practice  ·  NeetCode 150 — https://neetcode.io")

    b = bank(track)
    print(f"\n## Verbal / Technical ({track})")
    for q in b["technical"]:
        print(f"   - {q}")

    print("\n## Behavioral")
    for q in b["behavioral"]:
        print(f"   - {q}")

    p = personalized(profile, company)
    if p:
        print("\n## Personalized — expect to defend YOUR projects")
        for q in p.get("project_qs", []):
            print(f"   - {q}")
        print("\n## Personalized — behavioral (your experience)")
        for q in p.get("behavioral_qs", []):
            print(f"   - {q}")
    else:
        print("\n   (set ANTHROPIC_API_KEY + fill profile.yml for personalized project/behavioral questions)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("company")
    ap.add_argument("--track", default="ml", choices=["ml", "swe", "product"])
    args = ap.parse_args()

    profile_path = os.path.join(ROOT, "profile", "profile.yml")
    profile = {}
    if os.path.exists(profile_path):
        with open(profile_path) as f:
            profile = yaml.safe_load(f) or {}
    prep(args.company, args.track, profile)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    main()
