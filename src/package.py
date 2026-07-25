"""On-demand application-package generator for ONE role.

Given a single listing (looked up by id in data/listings.csv, or specified
directly with flags), assemble a complete, ready-to-apply bundle under
applications/{slug}/:

    resume.pdf       — the per-role tailored resume (src.tailor), copied in
    cover_letter.md  — the per-role cover letter (src.cover), when the LLM produced one
    jd.txt           — the fetched job description (may be empty)
    application.md   — the master doc: header, snapshot answers, a message to the
                       hiring team, STAR behavioral answers, values alignment, and
                       JD keyword coverage

Same guardrails as the rest of the assist layer (src/tailor.py, src/cover.py):
every LLM-derived section degrades to a visible 'TODO (LLM unavailable)' when
llm.complete returns None, every snapshot gap renders 'TODO: <field>' (never
fabricated), and every numeric claim is gated by verify_facts.verify. Nothing
here raises to the caller except a clean SystemExit when an --id is not found.

Run: python -m src.package --id <listing_id>
     python -m src.package --company "Anthropic" --title "ML Intern" --url ... --source greenhouse
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil

import yaml

from . import cover, jdfetch, keywords, llm, store, tailor, verify_facts

log = logging.getLogger("package")

ROOT = os.path.dirname(os.path.dirname(__file__))
APPS_DIR = os.path.join(ROOT, "applications")
PROFILE_YML = os.path.join(ROOT, "profile", "profile.yml")
VOICE_MD = os.path.join(ROOT, "profile", "voice.md")
RESUME_TEX = os.path.join(ROOT, "profile", "resume.tex")

MAX_JD_CHARS = 4000
LLM_UNAVAILABLE = "TODO (LLM unavailable)"

# STAR behavioral competencies: (heading, prompt phrasing for the model).
STAR_PROMPTS: list[tuple[str, str]] = [
    ("A challenging project", "a technically challenging project you drove and its outcome"),
    ("Working on a team", "a time you worked effectively with others on a team"),
    ("Overcoming a setback", "a time you hit a setback or failure and how you recovered"),
]


# --- small helpers ------------------------------------------------------------


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_profile() -> dict:
    """Load profile.yml defensively — missing/unreadable yields {} (snapshot TODOs)."""
    if not os.path.exists(PROFILE_YML):
        log.warning("profile.yml not found at %s — snapshot answers will show TODOs", PROFILE_YML)
        return {}
    try:
        return _load_yaml(PROFILE_YML)
    except (OSError, yaml.YAMLError) as exc:
        log.warning("profile.yml unreadable (%s) — proceeding with an empty profile", exc)
        return {}


def _read_optional(path: str) -> str:
    """Read a file's text, returning '' (not raising) if it is unavailable."""
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _slug(company: str, title: str) -> str:
    """Package-dir slug — identical rule to tailor/cover so the three agree."""
    return f"{company}_{title}".lower().replace(" ", "_").replace("/", "-")


def _first(*values: object) -> object | None:
    """First value that is neither None nor an empty string/list."""
    for v in values:
        if v not in (None, "", []):
            return v
    return None


def _stringify(value: object) -> str:
    """Flatten a profile list item (str or dict) into one readable line."""
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def _sentences(paragraph: str) -> list[str]:
    """Split on sentence-final punctuation followed by whitespace; a period glued
    to a digit ('$1.2M') is not a boundary, so metrics stay intact."""
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p for p in parts if p.strip()]


def _grounded(text: str, sources: list[str]) -> str:
    """Drop any sentence carrying an ungrounded numeric claim so the returned
    text passes verify_facts.verify. Preserves paragraph structure. This is the
    fact gate the strategy guide requires behind every LLM-generated section."""
    ok, violations = verify_facts.verify(text, sources)
    if ok:
        return text
    bad = set(violations)
    log.warning("dropping sentence(s) with ungrounded metric(s): %s", violations)
    out_paras: list[str] = []
    for para in text.split("\n\n"):
        kept = [s for s in _sentences(para) if not (verify_facts.extract_metrics(s) & bad)]
        if kept:
            out_paras.append(" ".join(kept))
    return "\n\n".join(out_paras)


# --- listing resolution -------------------------------------------------------


def resolve_listing(listing_id: str, company: str, title: str,
                    url: str, source: str) -> dict:
    """Resolve the target listing into a normalized dict.

    With an id, look it up in data/listings.csv (store.load_listings, matching the
    'id' column) and use its fields as the base; explicit flags override any matched
    field. Without an id, build purely from the flags (CSV bypassed). A given id that
    is absent from the CSV, or a resolution with no company/title, raises a clean
    SystemExit with a helpful message — never a traceback-only crash.
    """
    matched: dict = {}
    if listing_id:
        for row in store.load_listings():
            if row.get("id") == listing_id:
                matched = row
                break
        else:
            msg = (f"no listing with id={listing_id!r} in data/listings.csv. "
                   f"Run `python -m src.run` to refresh the tracker, or pass "
                   f"--company/--title/--url/--source to build a package directly.")
            print(msg)
            raise SystemExit(1)

    resolved = {
        "id": listing_id or matched.get("id", ""),
        "company": _first(company, matched.get("company")) or "",
        "title": _first(title, matched.get("title")) or "",
        "url": _first(url, matched.get("url")) or "",
        "source": _first(source, matched.get("source")) or "",
        "score": matched.get("score", ""),
        "posted_at": matched.get("posted_at", ""),
        "location": matched.get("location", ""),
    }

    if not resolved["company"] or not resolved["title"]:
        print("cannot build a package without a company and title. "
              "Pass --id <listing_id>, or both --company and --title.")
        raise SystemExit(1)
    return resolved


# --- snapshot answers ---------------------------------------------------------


def _snapshot_answers(profile: dict) -> list[tuple[str, object | None]]:
    """The one-glance application-form answers, drawn from profile.yml facts +
    answer_bank. A genuinely-missing value is returned as None so the renderer
    surfaces a visible 'TODO: <field>' rather than fabricating anything."""
    profile = profile or {}
    facts = profile.get("facts") if isinstance(profile.get("facts"), dict) else {}
    links = facts.get("links") if isinstance(facts.get("links"), dict) else {}
    bank = profile.get("answer_bank") if isinstance(profile.get("answer_bank"), dict) else {}

    return [
        ("Full name", _first(profile.get("name"), facts.get("name"))),
        ("Email", _first(facts.get("email"), links.get("email"), bank.get("email"))),
        ("Phone", _first(facts.get("phone"), bank.get("phone"))),
        ("LinkedIn", _first(links.get("linkedin"), facts.get("linkedin"), bank.get("linkedin"))),
        ("GitHub", _first(links.get("github"))),
        ("Location", _first(facts.get("location"), profile.get("location"), bank.get("location"))),
        ("Work authorization", _first(bank.get("work_authorization"),
                                       facts.get("work_authorization"))),
        ("Availability / earliest start", _first(bank.get("availability"),
                                                  facts.get("availability"),
                                                  bank.get("earliest_start"))),
        ("Graduation date", _first(bank.get("graduation_date"), facts.get("graduation_date"),
                                    facts.get("graduation"))),
        ("Target salary range", _first(bank.get("target_salary"), bank.get("target_salary_range"),
                                        facts.get("target_salary"))),
    ]


# --- LLM-derived sections -----------------------------------------------------


def _candidate_context(profile: dict) -> str:
    """The candidate's real material for a prompt: identity, skills, quantified
    achievements, and any answer_bank notes. The ONLY facts a section may use."""
    profile = profile or {}
    lines: list[str] = []

    name = profile.get("name") or (profile.get("facts") or {}).get("name")
    if name:
        lines.append(f"Name: {name}")
    skills = keywords.profile_skill_terms(profile)
    if skills:
        lines.append("Skills: " + ", ".join(skills[:40]))
    for key in ("experience", "projects", "achievements", "highlights"):
        vals = profile.get(key)
        if isinstance(vals, list) and vals:
            lines.append(f"{key.capitalize()}:")
            lines.extend(f"  - {_stringify(v)}" for v in vals[:6])
    bank = profile.get("answer_bank")
    if isinstance(bank, dict) and bank:
        lines.append("Answer bank:")
        lines.extend(f"  - {k}: {_stringify(v)}" for k, v in list(bank.items())[:12])
    return "\n".join(lines)


def _message_body(company: str, title: str, jd: str, jd_keywords: list[str],
                  voice: str, ctx: str, sources: list[str]) -> str:
    """A ~200-300 word condensed message to the hiring team. Keyword-rich, in the
    candidate's voice, fact-gated. Returns the visible placeholder when the LLM
    is unavailable (llm.complete -> None)."""
    kw = ", ".join(jd_keywords[:25])
    prompt = (
        f"Write a concise message to the hiring team for the {title} role at {company}.\n\n"
        f"Candidate material (the ONLY facts you may use):\n{ctx}\n\n"
        + (f"Job description:\n{jd[:MAX_JD_CHARS]}\n\n" if jd else "")
        + (f"Voice / tone guide:\n{voice}\n\n" if voice else "")
        + (f"Where truthful and natural, surface these job keywords: {kw}.\n\n" if kw else "")
        + "HARD RULES:\n"
        "- 200-300 words, plain prose, no bullet points, no salutation or signature line.\n"
        "- Human and direct; use contractions; sound like a real person, not a template.\n"
        "- NEVER invent metrics, tools, employers, or facts. Only the material above is true.\n\n"
        "Return ONLY the message body."
    )
    text = llm.complete(prompt, max_tokens=800)
    if text is None:
        return LLM_UNAVAILABLE
    return _grounded(text.strip(), sources) or LLM_UNAVAILABLE


def _star_body(company: str, title: str, prompt_desc: str, jd_keywords: list[str],
               voice: str, ctx: str, sources: list[str]) -> str:
    """One STAR behavioral answer, fact-gated. Placeholder when the LLM is None."""
    kw = ", ".join(jd_keywords[:15])
    prompt = (
        f"Answer a behavioral interview prompt in STAR form for a candidate applying to "
        f"{title} at {company}. Prompt: describe {prompt_desc}.\n\n"
        f"Candidate material (the ONLY facts you may use):\n{ctx}\n\n"
        + (f"Where truthful, weave in these job keywords: {kw}.\n\n" if kw else "")
        + (f"Voice / tone guide:\n{voice}\n\n" if voice else "")
        + "HARD RULES:\n"
        "- Label the four parts: Situation, Task, Action, Result.\n"
        "- Draw ONLY on the candidate material. NEVER invent metrics, tools, or facts.\n\n"
        "Return ONLY the STAR answer."
    )
    text = llm.complete(prompt, max_tokens=700)
    if text is None:
        return LLM_UNAVAILABLE
    return _grounded(text.strip(), sources) or LLM_UNAVAILABLE


# --- base resume text (for keyword coverage) ----------------------------------


def _base_resume_text(profile: dict) -> str:
    """Raw text of the base resume for the JD keyword-coverage measure. Prefers the
    profile's resume_tex_path, then profile/resume.tex, then whatever verify_facts
    can load as ground truth."""
    facts = (profile or {}).get("facts") or {}
    path = facts.get("resume_tex_path") or RESUME_TEX
    txt = _read_optional(path)
    if txt:
        return txt
    return "\n".join(verify_facts.load_sources())


# --- master-doc rendering -----------------------------------------------------


def _fmt_kv(label: str, value: object) -> str:
    return f"- **{label}:** {value}"


def _header_section(listing: dict, letter: str | None) -> str:
    score = listing["score"] if listing["score"] not in ("", None) else "n/a"
    lines = [
        f"# Application — {listing['company']} · {listing['title']}",
        "",
        "## HEADER",
        "",
        _fmt_kv("Company", listing["company"]),
        _fmt_kv("Title", listing["title"]),
        _fmt_kv("Apply URL", listing["url"] or "TODO: url"),
        _fmt_kv("Source", listing["source"] or "n/a"),
        _fmt_kv("Score", score),
        _fmt_kv("Posted at", listing["posted_at"] or "n/a"),
        _fmt_kv("Cover letter", "cover_letter.md" if letter else "unavailable (LLM unavailable)"),
    ]
    return "\n".join(lines)


def _snapshot_section(profile: dict) -> str:
    lines = ["## SNAPSHOT ANSWERS", ""]
    for label, value in _snapshot_answers(profile):
        lines.append(_fmt_kv(label, value if value not in (None, "") else f"TODO: {label}"))
    return "\n".join(lines)


def _message_section(listing: dict, jd: str, jd_keywords: list[str], voice: str,
                     ctx: str, sources: list[str]) -> str:
    body = _message_body(listing["company"], listing["title"], jd, jd_keywords,
                         voice, ctx, sources)
    return f"## MESSAGE TO HIRING TEAM\n\n{body}"


def _star_section(listing: dict, jd_keywords: list[str], voice: str,
                  ctx: str, sources: list[str]) -> str:
    parts = ["## STAR BEHAVIORAL ANSWERS"]
    for heading, desc in STAR_PROMPTS:
        body = _star_body(listing["company"], listing["title"], desc, jd_keywords,
                          voice, ctx, sources)
        parts.append(f"### {heading}\n\n{body}")
    return "\n\n".join(parts)


def _values_section(values: list[str]) -> str:
    lines = ["## VALUES ALIGNMENT", "",
             "The role names these values — connect each to a concrete, truthful example:", ""]
    lines.extend(f"- {v}" for v in values)
    return "\n".join(lines)


def _coverage_section(profile: dict, jd_keywords: list[str]) -> str:
    frac, present, _missing = keywords.coverage(_base_resume_text(profile), jd_keywords)
    total = len(jd_keywords)
    pct = round(frac * 100)
    body = f"{pct}% ({len(present)}/{total})"
    if _missing:
        body += " — missing: " + ", ".join(_missing[:30])
    return f"## JD KEYWORD COVERAGE\n\n{body}"


def _render_application_md(listing: dict, jd: str, letter: str | None) -> str:
    """Assemble the master application.md from all sections."""
    profile = _load_profile()
    voice = _read_optional(VOICE_MD)
    ctx = _candidate_context(profile)
    sources = verify_facts.load_sources()
    jd_keywords = keywords.extract_jd_keywords(jd)

    sections = [
        _header_section(listing, letter),
        _snapshot_section(profile),
        _message_section(listing, jd, jd_keywords, voice, ctx, sources),
        _star_section(listing, jd_keywords, voice, ctx, sources),
    ]
    values = cover.extract_values(jd)
    if values:  # omit the section entirely when the JD lists no values
        sections.append(_values_section(values))
    sections.append(_coverage_section(profile, jd_keywords))
    return "\n\n".join(sections) + "\n"


# --- orchestration ------------------------------------------------------------


def build_package(listing_id: str = "", company: str = "", title: str = "",
                  url: str = "", source: str = "", archetype: str = "startup",
                  role_title: bool = True) -> str:
    """Build the full application package for ONE role and return its directory.

    Degrades gracefully at every step: an empty JD, a missing resume PDF, or an
    unavailable LLM each produce a still-usable package rather than a crash.
    """
    listing = resolve_listing(listing_id, company, title, url, source)
    company, title = listing["company"], listing["title"]

    jd = jdfetch.fetch_jd(listing["source"], company, listing["url"], listing["id"])
    if not jd:
        log.warning("empty JD for %s / %s — proceeding without a description", company, title)

    pkg_dir = os.path.join(APPS_DIR, _slug(company, title))
    os.makedirs(pkg_dir, exist_ok=True)

    with open(os.path.join(pkg_dir, "jd.txt"), "w") as f:
        f.write(jd)

    # Resume: tailor returns the PDF to ship; copy it in when present.
    resume_src = tailor.tailor(company, title, archetype, jd, role_title=role_title)
    if resume_src and os.path.exists(resume_src):
        shutil.copyfile(resume_src, os.path.join(pkg_dir, "resume.pdf"))
    else:
        log.warning("no resume PDF to copy (tailor returned %r) — skipping resume.pdf", resume_src)

    # Cover: may be None when the LLM is unavailable — write a file only on success.
    letter = cover.tailor_cover(company, title, jd, archetype)
    if letter:
        with open(os.path.join(pkg_dir, "cover_letter.md"), "w") as f:
            f.write(letter)
    else:
        log.warning("no cover letter (LLM unavailable) — noted in application.md")

    with open(os.path.join(pkg_dir, "application.md"), "w") as f:
        f.write(_render_application_md(listing, jd, letter))

    print(f"\n✓ application package -> {pkg_dir}")
    return pkg_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a ready-to-apply package for one role.")
    ap.add_argument("--id", dest="listing_id", default="",
                    help="listing id to look up in data/listings.csv")
    ap.add_argument("--company", default="", help="override or bypass the CSV company")
    ap.add_argument("--title", default="", help="override or bypass the CSV title")
    ap.add_argument("--url", default="", help="override or bypass the CSV apply url")
    ap.add_argument("--source", default="", help="override or bypass the CSV ATS source")
    ap.add_argument("--archetype", default="startup",
                    choices=["big_tech", "startup", "quant", "research_lab", "product"])
    ap.add_argument("--role-title", action=argparse.BooleanOptionalAction, default=True,
                    help="pass through to tailor's target-title headline (default on)")
    args = ap.parse_args()

    if not args.listing_id and not (args.company and args.title):
        ap.error("provide --id, or both --company and --title")

    build_package(
        listing_id=args.listing_id, company=args.company, title=args.title,
        url=args.url, source=args.source, archetype=args.archetype,
        role_title=args.role_title,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    main()
