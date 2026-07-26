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
import datetime as dt
import html
import logging
import os
import re

import yaml

from . import keywords, llm, verify_facts

log = logging.getLogger("cover")

ROOT = os.path.dirname(os.path.dirname(__file__))
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


# --- PDF rendering ------------------------------------------------------------


def _cover_location(profile: dict) -> str:
    """Best available city/state for the PDF header; never invent an address."""
    direct = profile.get("location")
    if direct:
        return str(direct)
    education = profile.get("education")
    if isinstance(education, dict) and education.get("location"):
        return str(education["location"])
    preferences = profile.get("preferences")
    locations = preferences.get("locations") if isinstance(preferences, dict) else None
    preferred = locations.get("preferred") if isinstance(locations, dict) else None
    if isinstance(preferred, list) and preferred:
        return str(preferred[0])
    return ""


def _cover_contact(profile: dict) -> list[str]:
    """Contact fields for the header, in a stable ATS-readable order."""
    facts = profile.get("facts") if isinstance(profile.get("facts"), dict) else {}
    links = facts.get("links") if isinstance(facts.get("links"), dict) else {}
    fields = [
        profile.get("email"),
        profile.get("phone"),
        _cover_location(profile),
        profile.get("linkedin"),
        links.get("github"),
    ]
    out: list[str] = []
    for value in fields:
        if not value or not str(value).strip():
            continue
        text = str(value).strip()
        if text.startswith(("https://", "http://")):
            text = re.sub(r"^https?://(?:www\.)?", "", text).rstrip("/")
        out.append(text)
    return out


def _letter_body_paragraphs(letter: str) -> list[str]:
    """Return body-only paragraphs, peeling common model-added wrappers.

    The generator asks for body prose only, but providers occasionally add a
    greeting or sign-off. The PDF template owns those elements so they stay
    consistent across every application.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", letter or "") if p.strip()]
    if paragraphs and re.match(r"^(dear|to)\b", paragraphs[0], re.IGNORECASE):
        paragraphs.pop(0)
    if paragraphs and re.match(
        r"^(sincerely|best(?:\s+regards)?|kind\s+regards|respectfully)[,\s]",
        paragraphs[-1], re.IGNORECASE,
    ):
        paragraphs.pop()
    return paragraphs


def render_cover_pdf(letter: str, company: str, title: str, profile: dict,
                     output_path: str) -> str | None:
    """Render an ATS-readable, one-page business-letter PDF.

    The researched template uses a resume-style identity header, standard
    business-letter metadata, 10.5pt body text, generous white space, and no
    columns/icons/graphics. The completed file is reopened with pypdf; anything
    other than exactly one page is deleted rather than shipped.
    """
    try:
        from pypdf import PdfReader
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        log.error("cover PDF dependencies unavailable: %s", exc)
        return None

    paragraphs = _letter_body_paragraphs(letter)
    if not paragraphs:
        log.error("cover PDF skipped: no body paragraphs")
        return None

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    name = str(profile.get("name") or "Candidate").strip()
    contact_fields = _cover_contact(profile)
    contact_primary = " | ".join(contact_fields[:3])
    contact_links = " | ".join(contact_fields[3:])
    today = dt.date.today()
    date_text = f"{today.strftime('%B')} {today.day}, {today.year}"

    styles = getSampleStyleSheet()
    name_style = ParagraphStyle(
        "CoverName", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=18, leading=21, alignment=TA_LEFT, textColor=colors.HexColor("#111827"),
        spaceAfter=3,
    )
    contact_style = ParagraphStyle(
        "CoverContact", parent=styles["Normal"], fontName="Helvetica",
        fontSize=8.5, leading=11, alignment=TA_LEFT, textColor=colors.HexColor("#374151"),
        spaceAfter=7,
    )
    meta_style = ParagraphStyle(
        "CoverMeta", parent=styles["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=12, alignment=TA_LEFT, textColor=colors.HexColor("#111827"),
        spaceAfter=1,
    )
    role_style = ParagraphStyle(
        "CoverRole", parent=meta_style, fontName="Helvetica-Bold", spaceBefore=5, spaceAfter=9,
    )
    body_style = ParagraphStyle(
        "CoverBody", parent=styles["Normal"], fontName="Helvetica",
        fontSize=10.5, leading=14, alignment=TA_LEFT, textColor=colors.HexColor("#111827"),
        spaceAfter=9,
    )

    doc = SimpleDocTemplate(
        output_path, pagesize=LETTER,
        leftMargin=0.78 * inch, rightMargin=0.78 * inch,
        topMargin=0.62 * inch, bottomMargin=0.62 * inch,
        title=f"{name} - {title} Cover Letter", author=name,
        subject=f"Application for {title} at {company}",
    )
    esc = html.escape
    story = [Paragraph(esc(name), name_style)]
    if contact_primary:
        story.append(Paragraph(esc(contact_primary), ParagraphStyle(
            "CoverContactPrimary", parent=contact_style, spaceAfter=1,
        )))
    if contact_links:
        story.append(Paragraph(esc(contact_links), contact_style))
    story.extend([
        HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#9CA3AF"),
                   spaceBefore=1, spaceAfter=10),
        Paragraph(esc(date_text), meta_style),
        Spacer(1, 5),
        Paragraph("Hiring Team", meta_style),
        Paragraph(esc(company), meta_style),
        Paragraph(f"Re: {esc(title)}", role_style),
        Paragraph(f"Dear {esc(company)} Hiring Team,", body_style),
    ])
    story.extend(Paragraph(esc(p).replace("\n", "<br/>"), body_style) for p in paragraphs)
    story.extend([
        Spacer(1, 2),
        Paragraph("Sincerely,", body_style),
        Paragraph(esc(name), ParagraphStyle(
            "CoverSignoff", parent=body_style, fontName="Helvetica-Bold", spaceAfter=0,
        )),
    ])

    try:
        doc.build(story)
        reader = PdfReader(output_path)
        if len(reader.pages) != 1:
            log.error("cover PDF overflowed to %d pages — deleting %s",
                      len(reader.pages), output_path)
            os.remove(output_path)
            return None
        if not (reader.pages[0].extract_text() or "").strip():
            log.error("cover PDF has no extractable text — deleting %s", output_path)
            os.remove(output_path)
            return None
    except Exception as exc:  # noqa: BLE001 — PDF failure must not crash package generation
        log.error("cover PDF render failed: %s", exc)
        if os.path.exists(output_path):
            os.remove(output_path)
        return None
    return output_path


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


# --- LLM generation -----------------------------------------------------------


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


def _is_bachelors(degree: str | None) -> bool:
    """True when the degree string reads as an undergraduate (Bachelor's) degree,
    so we can assert the candidate is NOT a Master's / PhD candidate."""
    d = (degree or "").lower()
    return "bachelor" in d or bool(re.search(r"\bb\.?s\b|\bb\.?a\b", d))


def immutable_facts(profile: dict) -> str:
    """Render the candidate's non-negotiable identity as a labeled, reproduce-exactly
    block, read defensively from the live profile dict.

    Guards against the model inventing or upgrading a credential to match a JD (the
    live run once claimed a Master's in CS because the JD asked for 'MS or PhD'; the
    candidate is a BS Statistics & Data Science student). Reads education/experience/
    answer_bank locations with .get chains, renders only the facts actually present,
    and NEVER raises on a missing section — an empty profile yields ''.

    Facts read: education.degree / .school / .graduation (graduation falls back to the
    top-level grad_date), the current role+company from experience[0], and work
    authorization from answer_bank.eligibility (work_authorized_us and NOT
    requires_sponsorship => authorized, no sponsorship).
    """
    profile = profile or {}
    education = profile.get("education") if isinstance(profile.get("education"), dict) else {}
    lines: list[str] = []

    # Degree line: degree, school, graduation — only the parts present, exact wording.
    degree = education.get("degree")
    school = education.get("school")
    graduation = education.get("graduation") or profile.get("grad_date")
    degree_parts = [str(p).strip() for p in (degree, school, graduation) if p]
    if degree_parts:
        lines.append("Degree (NEVER change/upgrade): " + ", ".join(degree_parts))
        if _is_bachelors(degree):
            lines.append("Standing: undergraduate/BS student, NOT a Master's or PhD candidate")

    # Current role from the most recent experience entry.
    experience = profile.get("experience")
    if isinstance(experience, list) and experience and isinstance(experience[0], dict):
        role = experience[0].get("role")
        company = experience[0].get("company")
        if role and company:
            lines.append(f"Current role: {role} at {company}")
        elif role:
            lines.append(f"Current role: {role}")

    # Work authorization from answer_bank.eligibility.
    bank = profile.get("answer_bank") if isinstance(profile.get("answer_bank"), dict) else {}
    elig = bank.get("eligibility") if isinstance(bank.get("eligibility"), dict) else {}
    if elig.get("work_authorized_us") and not elig.get("requires_sponsorship"):
        lines.append("Work authorization: authorized to work in the US, no sponsorship required")

    if not lines:
        return ""
    header = "IMMUTABLE CANDIDATE FACTS (reproduce EXACTLY; never change, upgrade, or invent):"
    footer = (
        "- If the posting prefers a qualification the candidate lacks (e.g. a Master's/PhD), "
        "do NOT claim it, but also do NOT apologize for or draw attention to lacking it — "
        "lead with genuine strengths and simply omit what isn't there."
    )
    return header + "\n" + "\n".join(f"- {line}" for line in lines) + "\n" + footer


def _build_prompt(company: str, title: str, emphasis: str, jd: str,
                  inject: list[str], values: list[str], ctx: str, voice: str,
                  profile: dict | None = None, strict: bool = False) -> str:
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
    # HARD RULE, placed ABOVE the writing instructions: the model must reproduce the
    # candidate's real credentials verbatim and can never invent or upgrade one to match
    # the JD (no Master's/PhD to satisfy an 'MS or PhD' ask, no BS->MS, no intern->FTE).
    facts = immutable_facts(profile or {})
    facts_block = (
        facts + "\n"
        "These facts are NON-NEGOTIABLE: reproduce every credential EXACTLY as written "
        "above. NEVER claim a degree, major, school, job title, employer, or credential "
        "that is not listed here, EVEN IF THE JOB ASKS FOR ONE. Do NOT claim a Master's "
        "or PhD to match an 'MS or PhD' requirement; do NOT upgrade a Bachelor's to a "
        "Master's, an internship to a full-time role, or invent a field of study.\n\n"
        if facts else ""
    )
    return (
        f"Write a cover letter for the {title} role at {company}.\n\n"
        f"Candidate material (the ONLY facts you may use):\n{ctx}\n\n"
        + (f"Job description:\n{jd}\n\n" if jd else "")
        + (f"Voice / tone guide:\n{voice}\n\n" if voice else "")
        + f"Emphasis for this kind of company: {emphasis}\n\n"
        + facts_block
        + "Follow this 5-paragraph formula, one short paragraph each:\n"
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
        f"- Target 250-350 words; hard cap {MAX_WORDS} words.\n"
        + strict_block
        + "\nReturn ONLY the letter body as plain prose paragraphs, nothing else."
    )


def _generate(prompt: str) -> str:
    """Model text via the shared LLM helper (Claude primary, GPT fallback).

    Returns '' when no provider is configured or the call yields nothing —
    llm.complete never raises and returns None in that case, which we treat as
    empty so the empty-letter gate in tailor_cover fails closed."""
    return llm.complete(prompt, max_tokens=1200) or ""


# --- orchestration ------------------------------------------------------------


def tailor_cover(company: str, title: str, jd: str, archetype: str = "startup") -> str | None:
    """Generate a fact-gated cover letter, write it to tailored/, and return the text.

    Fails closed: when the LLM is unavailable (no ANTHROPIC_API_KEY / OPENAI_API_KEY
    provider configured, so llm.complete returns None) or the final letter is empty
    after the fact-gate scrub, this logs a skip, writes NO file, prints a failure
    line, and returns None. Only real non-empty content is written and returned.
    """
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
    prompt = _build_prompt(company, title, emphasis, jd, inject, values, ctx, voice, profile=profile)
    letter = _sanitize(_generate(prompt))
    ok, violations = verify_facts.verify(letter, sources)
    if not ok:
        log.warning("draft has ungrounded metric(s) %s — regenerating once", violations)
        strict = _build_prompt(company, title, emphasis, jd, inject, values, ctx, voice,
                               profile=profile, strict=True)
        letter = _sanitize(_generate(strict))
        ok, violations = verify_facts.verify(letter, sources)
        if not ok:
            log.warning("still ungrounded %s — dropping unverifiable sentence(s)", violations)
            letter = _drop_unverified_sentences(letter, sources)

    letter = _cap_words(letter, MAX_WORDS)

    # Fail closed: no LLM completion (llm.complete -> None) or an empty letter after
    # the fact-gate scrub means we have nothing to ship. Write NO file and never
    # print a ✓ — reporting success while shipping a 0-byte letter was the bug.
    if not letter.strip():
        log.warning(
            "no cover letter for %s / %s — LLM unavailable or produced no grounded "
            "content (set ANTHROPIC_API_KEY / OPENAI_API_KEY); wrote nothing",
            company, title,
        )
        print("\n✗ cover letter unavailable — no content generated (nothing written)")
        return None

    os.makedirs(OUT_DIR, exist_ok=True)
    slug = f"{company}_{title}".lower().replace(" ", "_").replace("/", "-")
    out_path = os.path.join(OUT_DIR, f"{slug}_cover.md")
    with open(out_path, "w") as f:
        f.write(letter)

    pdf_path = os.path.join(OUT_DIR, f"{slug}_cover.pdf")
    rendered_pdf = render_cover_pdf(letter, company, title, profile, pdf_path)
    if rendered_pdf:
        print(f"\n✓ cover letter ({len(letter.split())} words) -> {out_path} + {rendered_pdf}")
    else:
        log.warning("cover text generated, but PDF rendering failed — Markdown kept at %s", out_path)
        print(f"\n⚠ cover letter text generated, PDF unavailable -> {out_path}")
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
