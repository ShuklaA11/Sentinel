"""Draft free-text application answers in Arnav's voice, grounded only in his facts.

Two sources, kept strictly separate in the prompt:
  * profile.yml  -> the ONLY substance. No fact is stated that isn't here (truthfulness,
                    same discipline as tailor.py: reword real facts, never fabricate).
  * voice.md     -> style ONLY. Cadence and tone are imitated; content is never copied.

Em-dash guarantee (3 layers), because the source voice is em-dash-heavy and a prompt alone
won't hold:
  1. the prompt forbids em/en dashes,
  2. a validate-and-retry pass regenerates once if any slip through,
  3. strip_em_dashes() deterministically removes any that remain — the hard guarantee.

Graceful: no ANTHROPIC_API_KEY (or the anthropic package missing) -> draft_answer returns
None, and the driver leaves that field for a human instead of guessing.
"""
from __future__ import annotations

import logging
import os
import re

log = logging.getLogger("essays")

MODEL = os.environ.get("ESSAY_MODEL", "claude-sonnet-4-6")

# figure(2012), en(2013), em(2014), horizontal bar(2015) dashes — all eliminated.
_DASH_CLASS = "[‒–—―]"


def has_em_dash(text: str) -> bool:
    """True if the text contains any em/en dash or an ASCII double-hyphen em dash."""
    t = text or ""
    return bool(re.search(_DASH_CLASS, t)) or "--" in t


def strip_em_dashes(text: str) -> str:
    """Deterministically remove em/en dashes — the hard guarantee (layer 3).

    Number ranges collapse to a hyphen (2024-2025); every other dash becomes a comma,
    the least-bad general substitution. This is a backstop: the validate-and-retry pass
    is what keeps the prose grammatical, so this rarely has to fire.
    """
    if not text:
        return text
    t = text
    t = re.sub(r"(?<=\d)\s*" + _DASH_CLASS + r"\s*(?=\d)", "-", t)  # number ranges -> hyphen
    t = re.sub(r"\s*--\s*", ", ", t)                                # ASCII em dash -> comma
    t = re.sub(r"\s*" + _DASH_CLASS + r"\s*", ", ", t)              # any remaining dash -> comma
    # Clean up substitution artifacts.
    t = re.sub(r"^\s*,\s*", "", t)              # no leading comma
    t = re.sub(r",\s*,", ",", t)                # collapse doubled commas
    t = re.sub(r",(\s*[.!?;:])", r"\1", t)      # drop comma bumped against other punctuation
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)      # no space before punctuation
    t = re.sub(r"[ \t]{2,}", " ", t)            # collapse runs of spaces
    return t


def _facts_blob(profile: dict) -> str:
    """Serialize the profile into the factual context the model may draw from."""
    lines: list[str] = []
    edu = profile.get("education", {}) or {}
    if edu:
        lines.append(f"Education: {edu.get('degree','')} at {edu.get('school','')}, "
                     f"graduating {edu.get('graduation','')} ({edu.get('location','')}).")
    for e in profile.get("experience", []) or []:
        lines.append(f"{e.get('role','')} at {e.get('company','')} ({e.get('dates','')}): "
                     + "; ".join(e.get("bullets", [])))
    for p in profile.get("projects", []) or []:
        lines.append(f"Project {p.get('name','')}: " + "; ".join(p.get("bullets", [])))
    for a in profile.get("activities", []) or []:
        lines.append(f"{a.get('role','')} at {a.get('org','')}: " + "; ".join(a.get("bullets", [])))
    sk = profile.get("skills", {}) or {}
    if sk:
        flat = ", ".join(v for vals in sk.values()
                         for v in (vals if isinstance(vals, list) else [vals]))
        lines.append(f"Skills: {flat}")
    return "\n".join(lines)


def _prompt(question: str, profile: dict, voice: str, jd_text: str | None, word_limit: int | None) -> str:
    facts = _facts_blob(profile)
    wl = f"Keep it under {word_limit} words. " if word_limit else ""
    jd = f"\n\nROLE / JOB DESCRIPTION (context for relevance):\n{jd_text.strip()}\n" if jd_text else ""
    return (
        "You are drafting Arnav Shukla's answer to a free-text question on an internship "
        "application. Write it in first person as Arnav.\n\n"
        "TWO HARD RULES:\n"
        "1. SUBSTANCE comes ONLY from the FACTS below. Do not invent achievements, metrics, "
        "companies, coursework, or experiences. If the question asks about something the facts "
        "do not support, answer only with what they do support, honestly.\n"
        "2. STYLE imitates the VOICE reference, which is a style sample ONLY. Never copy its "
        "content or phrasing. Borrow cadence and tone, not material.\n\n"
        "WRITING CONSTRAINTS:\n"
        "- Absolutely NO em dashes or en dashes. Use periods and commas instead. Non-negotiable.\n"
        "- Professional register. Concrete and specific: name real tools, numbers, projects.\n"
        "- Avoid filler abstraction (meaningful, impactful, aspire, drive change) unless a "
        "concrete fact backs it. No memoir-style openers, no sentimental callbacks.\n"
        f"- {wl}Answer the question directly.\n\n"
        f"FACTS (the only substance you may use):\n{facts}\n\n"
        f"VOICE (style reference only, never copy its content):\n{voice.strip()}\n"
        f"{jd}\n"
        f"QUESTION:\n{question.strip()}\n\n"
        "Write only the answer text, nothing else."
    )


def _client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed — cannot draft answers")
        return None
    return anthropic.Anthropic(api_key=key)


def _generate(client, prompt: str) -> str | None:
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:  # noqa: BLE001 — never let drafting crash the run
        log.error("essay generation failed: %s", exc)
        return None


def draft_answer(question: str, profile: dict, voice: str, jd_text: str | None = None,
                 word_limit: int | None = None, client=None, max_retries: int = 1) -> str | None:
    """Draft one answer. Returns clean text (guaranteed no em/en dashes), or None if the
    LLM is unavailable — in which case the driver leaves the field for a human.
    """
    client = client or _client()
    if client is None:
        return None

    prompt = _prompt(question, profile, voice, jd_text, word_limit)
    text = _generate(client, prompt)
    tries = 0
    while text and has_em_dash(text) and tries < max_retries:
        text = _generate(
            client,
            prompt + "\n\nYour previous draft used em or en dashes. Rewrite it with NONE, "
            "using periods and commas instead.",
        )
        tries += 1

    if not text:
        return None
    return strip_em_dashes(text).strip()  # layer 3: the deterministic guarantee
