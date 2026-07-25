"""Phase 7b: per-role cover-letter generator.

Writes a short, human cover letter following the strategy guide's 5-paragraph
formula: (1) a specific hook for THIS company, (2) a technical-match paragraph
that surfaces the JD's own keywords the candidate GENUINELY has, (3) an
experience story built on a real quantified achievement from profile.yml,
(4) a why-this-role paragraph grounded in the JD, and (5) a short close with
location / availability.

Same guardrails as src/tailor.py:
  - Graceful skip: no ANTHROPIC_API_KEY => log a warning and return None. Never crash.
  - Truthfulness: keyword injection is limited to skills the candidate really holds
    (keywords.truthful_injection_candidates), and every numeric claim is gated by
    verify_facts.verify against the real profile/resume. A draft that fabricates a
    number is regenerated once; if it still fails, the offending sentence is dropped
    rather than shipped.

Run: python -m src.cover --company "Anthropic" --title "ML Intern" [--jd-file jd.txt] [--archetype startup]
"""
from __future__ import annotations

import argparse
import logging
import os
import re

import yaml

from . import keywords, verify_facts

log = logging.getLogger("cover")

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL = os.environ.get("COVER_MODEL", "claude-sonnet-4-6")
OUT_DIR = os.path.join(ROOT, "tailored")
PROFILE_YML = os.path.join(ROOT, "profile", "profile.yml")
VOICE_MD = os.path.join(ROOT, "profile", "voice.md")
RESUME_BANK = os.path.join(ROOT, "config", "resume_bank.yml")

MAX_WORDS = 400
MAX_JD_CHARS = 4000

# Headers a JD uses to introduce a company-values list.
_VALUES_HEADER = re.compile(
    r"(our\s+(core\s+)?values|what\s+we\s+value|values\s+we\s+(?:live|hold|share)"
    r"|our\s+(?:core\s+)?principles|things\s+we\s+value)\b",
    re.IGNORECASE,
)

# Leading bullet / numbering noise to peel off a value line.
_BULLET_PREFIX = re.compile(r"^[-*••–\d.)\s]+")


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _read_optional(path: str) -> str:
    """Read a file's text, returning '' (not raising) if it is unavailable."""
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


# --- company values -----------------------------------------------------------


def _is_value_phrase(phrase: str) -> bool:
    """A value item is a short, word-leading phrase (<= 5 words)."""
    words = phrase.split()
    return bool(words) and len(words) <= 5 and phrase[0].isalpha()


def _split_inline(text: str) -> list[str]:
    """Split an inline 'Craft, Candor, and Speed' list into its items."""
    parts = re.split(r"\s*(?:,|;|/|&|\band\b)\s*", text)
    return [p.strip(" .").strip() for p in parts if p.strip(" .").strip()]


def extract_values(jd_text: str) -> list[str]:
    """Pull the company's stated values from a JD when it lists them.

    Recognizes an 'Our values:' / 'What we value:' style header, then reads
    either an inline comma list on the same line or a short bullet list on the
    following lines. Returns a deduped list, or [] when no values block is
    present (the common case).
    """
    text = jd_text or ""
    m = _VALUES_HEADER.search(text)
    if not m:
        return []

    items: list[str] = []
    started = False
    for raw in text[m.end():].splitlines()[:12]:
        s = raw.strip().lstrip(":").strip()
        if not s:
            if started:
                break
            continue
        cleaned = _BULLET_PREFIX.sub("", s).strip(" .").strip()
        if not cleaned:
            continue
        # An inline comma list (with or without a leading bullet) at the top of
        # the block is the whole values list on one line.
        if not started and "," in cleaned:
            parts = [p for p in _split_inline(cleaned) if _is_value_phrase(p)]
            if len(parts) >= 2:
                return _dedupe(parts)
        if _is_value_phrase(cleaned):
            items.append(cleaned)
            started = True
        elif started:
            break
    return _dedupe(items)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# --- text hygiene -------------------------------------------------------------


def _sanitize(text: str) -> str:
    """Strip em/en dashes the guide bans, and tidy the punctuation left behind."""
    text = re.sub(r"\s*—\s*", ", ", text)   # em dash -> comma
    text = re.sub(r"\s*–\s*", "-", text)     # en dash -> hyphen (ranges)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def _split_sentences(paragraph: str) -> list[str]:
    """Split on sentence-final punctuation followed by whitespace. A period
    glued to a digit (e.g. '$1.2M') is not a boundary, so metrics stay intact."""
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p for p in parts if p.strip()]


def _cap_words(text: str, limit: int = MAX_WORDS) -> str:
    """Trim the letter to at most `limit` words, dropping whole trailing
    sentences (and empty paragraphs) so it never ends mid-thought."""
    if len(text.split()) <= limit:
        return text
    out_paras: list[str] = []
    used = 0
    for para in text.split("\n\n"):
        kept: list[str] = []
        for sent in _split_sentences(para):
            n = len(sent.split())
            if used + n > limit:
                break
            kept.append(sent)
            used += n
        if kept:
            out_paras.append(" ".join(kept))
        if used >= limit:
            break
    return "\n\n".join(out_paras)


def _drop_unverified_sentences(text: str, source_texts: list[str]) -> str:
    """Last-resort fact gate: remove any sentence whose numeric claim is not
    grounded in the sources, preserving paragraph structure. Guarantees the
    returned text passes verify_facts.verify."""
    ok, violations = verify_facts.verify(text, source_texts)
    if ok:
        return text
    bad = set(violations)
    out_paras: list[str] = []
    for para in text.split("\n\n"):
        kept = [s for s in _split_sentences(para)
                if not (verify_facts.extract_metrics(s) & bad)]
        if kept:
            out_paras.append(" ".join(kept))
    return "\n\n".join(out_paras)


# --- LLM client + generation --------------------------------------------------


def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _stringify(value: object) -> str:
    """Flatten a profile list item (str or dict) into one readable line."""
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def _profile_context(profile: dict) -> str:
    """Assemble the candidate's real material for the prompt: identity, skills,
    and quantified achievements the experience paragraph can draw on."""
    profile = profile or {}
    facts = profile.get("facts") or {}
    lines: list[str] = []

    name = profile.get("name") or facts.get("name")
    if name:
        lines.append(f"Name: {name}")
    location = facts.get("location") or profile.get("location")
    if location:
        lines.append(f"Location: {location}")
    availability = facts.get("availability") or profile.get("availability")
    if availability:
        lines.append(f"Availability: {availability}")

    skills = keywords.profile_skill_terms(profile)
    if skills:
        lines.append("Skills: " + ", ".join(skills[:40]))

    for key in ("achievements", "highlights", "experience", "projects"):
        vals = profile.get(key)
        if isinstance(vals, list) and vals:
            lines.append(f"{key.capitalize()}:")
            lines.extend(f"  - {_stringify(v)}" for v in vals[:6])
    return "\n".join(lines)


def _build_prompt(company: str, title: str, emphasis: str, jd: str,
                  inject: list[str], values: list[str], ctx: str, voice: str,
                  strict: bool = False) -> str:
    inject_block = (
        "Where TRUTHFUL and natural, weave in these employer keywords the candidate "
        f"genuinely has (use their exact wording): {', '.join(inject)}.\n" if inject else ""
    )
    values_block = (
        f"The company lists these values: {', '.join(values)}. Weave ONE brief, "
        "specific line of genuine alignment with them.\n" if values else ""
    )
    strict_block = (
        "\nEARLIER DRAFT FABRICATED A NUMBER. This time use ONLY numbers that appear in "
        "the candidate material above. If unsure, state the achievement without a number.\n"
        if strict else ""
    )
    return (
        f"Write a cover letter for the {title} role at {company}.\n\n"
        f"Candidate material (the ONLY facts you may use):\n{ctx}\n\n"
        + (f"Job description:\n{jd}\n\n" if jd else "")
        + (f"Voice / tone guide:\n{voice}\n\n" if voice else "")
        + f"Emphasis for this kind of company: {emphasis}\n\n"
        "Follow this 5-paragraph formula, one short paragraph each:\n"
        "1. HOOK: specific, genuine enthusiasm for THIS company's product / mission. "
        "No generic flattery.\n"
        "2. TECHNICAL MATCH: the candidate's relevant skills, using the job description's "
        "exact terminology.\n"
        "3. EXPERIENCE STORY: one real, quantified achievement from the candidate material. "
        "Never invent a number.\n"
        "4. WHY THIS ROLE: reference something concrete from the job description.\n"
        "5. CLOSE: short; include location and availability from the candidate material.\n\n"
        + inject_block
        + values_block
        + "HARD RULES:\n"
        "- Human and direct. Use contractions. Sound like a real person, not a template.\n"
        "- NEVER open with 'I am writing to express my interest' or similar boilerplate.\n"
        "- NO em dashes. NO bullet points. NO full CV restatement.\n"
        "- NEVER invent metrics, tools, employers, or facts. Only the material above is true.\n"
        f"- Under {MAX_WORDS} words total.\n"
        + strict_block
        + "\nReturn ONLY the letter body as plain prose paragraphs, nothing else."
    )


def _generate(client, prompt: str) -> str:
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except Exception as exc:  # noqa: BLE001
        log.error("cover generation failed (%s)", exc)
        return ""


# --- orchestration ------------------------------------------------------------


def tailor_cover(company: str, title: str, jd: str, archetype: str = "startup") -> str | None:
    """Generate a fact-gated cover letter, write it to tailored/, and return the text.

    Returns None (after logging a skip) when no ANTHROPIC_API_KEY is set.
    """
    client = _client()
    if client is None:
        log.warning("no ANTHROPIC_API_KEY — skipping cover letter for %s / %s", company, title)
        return None

    profile = _load_yaml(PROFILE_YML) if os.path.exists(PROFILE_YML) else {}
    bank = _load_yaml(RESUME_BANK) if os.path.exists(RESUME_BANK) else {}
    voice = _read_optional(VOICE_MD)

    jd = (jd or "")[:MAX_JD_CHARS]
    emphasis = (bank.get("company_archetype_emphasis", {}) or {}).get(
        archetype, "Balance impact, ownership, and rigor.")

    # Keyword lever: surface only JD keywords the candidate GENUINELY holds.
    jd_keywords = keywords.extract_jd_keywords(jd)
    profile_terms = keywords.profile_skill_terms(profile)
    inject = keywords.truthful_injection_candidates(jd_keywords, profile_terms)
    values = extract_values(jd)
    ctx = _profile_context(profile)
    if inject:
        log.info("weaving %d truthful keyword(s): %s", len(inject), ", ".join(inject))
    if values:
        log.info("aligning to %d company value(s): %s", len(values), ", ".join(values))

    sources = verify_facts.load_sources()

    # First draft, then the numeric fact gate: regenerate once, then drop the
    # unverifiable sentence rather than ship a fabricated number.
    prompt = _build_prompt(company, title, emphasis, jd, inject, values, ctx, voice)
    letter = _sanitize(_generate(client, prompt))
    ok, violations = verify_facts.verify(letter, sources)
    if not ok:
        log.warning("draft has ungrounded metric(s) %s — regenerating once", violations)
        strict = _build_prompt(company, title, emphasis, jd, inject, values, ctx, voice, strict=True)
        letter = _sanitize(_generate(client, strict))
        ok, violations = verify_facts.verify(letter, sources)
        if not ok:
            log.warning("still ungrounded %s — dropping unverifiable sentence(s)", violations)
            letter = _drop_unverified_sentences(letter, sources)

    letter = _cap_words(letter, MAX_WORDS)

    os.makedirs(OUT_DIR, exist_ok=True)
    slug = f"{company}_{title}".lower().replace(" ", "_").replace("/", "-")
    out_path = os.path.join(OUT_DIR, f"{slug}_cover.md")
    with open(out_path, "w") as f:
        f.write(letter)

    print(f"\n✓ cover letter ({len(letter.split())} words) -> {out_path}")
    return letter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--company", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--archetype", default="startup",
                    choices=["big_tech", "startup", "quant", "research_lab", "product"])
    ap.add_argument("--jd-file", default="")
    args = ap.parse_args()
    jd = ""
    if args.jd_file and os.path.exists(args.jd_file):
        with open(args.jd_file) as f:
            jd = f.read()
    result = tailor_cover(args.company, args.title, jd, archetype=args.archetype)
    if result is None:
        print("cover letter unavailable (no ANTHROPIC_API_KEY set)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    main()
