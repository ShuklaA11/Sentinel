"""Phase 7: per-role resume tailoring -> PDF.

Rewrites ONLY the text inside \\resumeItem{} in the document body, preserving the template
byte-for-byte, then compiles with tectonic. Falls back to the base resume if anything fails.
Truthfulness: reword / re-emphasize real bullets only — never fabricate.

Run: python -m src.tailor --company "Anthropic" --title "ML Intern" [--archetype startup] [--jd-file jd.txt]
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess

import yaml

from . import keywords, llm, resumeselect, roletitles, verify_facts

log = logging.getLogger("tailor")

ROOT = os.path.dirname(os.path.dirname(__file__))
MARKER = "\\resumeItem{"
MAX_CHARS = 115
OUT_DIR = os.path.join(ROOT, "tailored")

# Target-track vocabulary for the plausibility gate on the headline. The applied-for
# title must contain one of these (word-boundary) for a target-title line to be added.
# Augmented at runtime with the candidate's own profile preferences.tracks.
DEFAULT_TRACK_TERMS = (
    "ml", "machine learning", "data", "ai", "swe", "software", "product",
    "computer vision", "vision", "cv", "perception", "deep learning",
    "ml engineer", "research",
)


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _commented(tex: str, pos: int) -> bool:
    """True if pos sits after an unescaped % on its line (i.e. it's commented out)."""
    seg = tex[tex.rfind("\n", 0, pos) + 1: pos]
    return any(c == "%" and (i == 0 or seg[i - 1] != "\\") for i, c in enumerate(seg))


def _extract_items(tex: str) -> list[tuple[int, int, str]]:
    """Find \\resumeItem{...} spans in the document body (balanced braces, not commented)."""
    body_start = tex.find("\\begin{document}")
    items, i = [], max(body_start, 0)
    while True:
        start = tex.find(MARKER, i)
        if start == -1:
            break
        j, depth = start + len(MARKER), 1
        while j < len(tex) and depth > 0:
            depth += (tex[j] == "{") - (tex[j] == "}")
            j += 1
        if not _commented(tex, start):
            items.append((start, j, tex[start + len(MARKER): j - 1]))
        i = j
    return items


def _safe_bullet(new: str, old: str) -> str:
    """Accept the rewrite only if it's LaTeX-safe and within budget, else keep original."""
    if not new or len(new) > MAX_CHARS:
        return old
    depth = 0
    for c in new:
        depth += (c == "{") - (c == "}")
        if depth < 0:
            return old
    if depth != 0:
        return old
    if any(c in "%&#$_" and (i == 0 or new[i - 1] != "\\") for i, c in enumerate(new)):
        return old
    return new


def _apply_rewrites(text: str, bullets: list[str]) -> list[str]:
    """Parse the model's `INDEX|||BULLET` lines, keeping originals for any unsafe/missing row."""
    out = dict(enumerate(bullets))  # default to originals; override with safe rewrites
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*\|\|\|\s*(.*)", line)
        if m and int(m.group(1)) < len(bullets):
            idx = int(m.group(1))
            out[idx] = _safe_bullet(m.group(2).strip(), bullets[idx])
    return [out[i] for i in range(len(bullets))]


def _count_pages(data: bytes) -> int:
    """PDF page count from raw bytes. Uses pypdf (reads tectonic's compressed object
    streams correctly); falls back to a plaintext byte-regex only if pypdf can't parse.
    Returns >=1.

    The regex alone silently undercounted: tectonic stores the page tree inside a
    compressed object stream, so '/Count N' and '/Type /Page' aren't visible as plain
    bytes — the fallback returned 1 and 2-page resumes shipped as '✓ compiled'.
    """
    try:
        import io

        from pypdf import PdfReader

        pages = len(PdfReader(io.BytesIO(data)).pages)
        if pages:
            return pages
    except Exception:  # noqa: BLE001 — fall back to the byte-regex on any parse failure
        pass
    counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", data)]
    if counts:
        return max(counts)
    return len(re.findall(rb"/Type\s*/Page[^s]", data)) or 1


def _render(tex: str, items: list[tuple[int, int, str]], bullets: list[str]) -> str:
    """Rebuild the full .tex, swapping each \\resumeItem{} body for its (possibly rewritten)
    bullet. Patches from last span to first so earlier indices stay valid."""
    out = tex
    for (start, end, _old), nb in sorted(zip(items, bullets), key=lambda x: -x[0][0]):
        out = out[:start] + MARKER + nb + "}" + out[end:]
    return out


def _xetex_compat(tex: str) -> str:
    """Comment out pdfTeX-only lines tectonic's XeTeX engine can't run. Visually a no-op:
    XeTeX is Unicode-native, so the PDF stays ATS-readable without glyphtounicode."""
    tex = tex.replace("\\input{glyphtounicode}", "%\\input{glyphtounicode}")
    tex = tex.replace("\\pdfgentounicode=1", "%\\pdfgentounicode=1")
    return tex


# --- target-title headline (plausibility-gated) ------------------------------
# A single decorative line under the name in the \begin{center} header. It never
# touches a real \resumeSubheading experience title — those stay byte-identical.

def _track_terms(profile: dict) -> list[str]:
    """Default target-track vocabulary plus the candidate's profile preferences.tracks."""
    prefs = (profile or {}).get("preferences") or {}
    tracks = prefs.get("tracks") if isinstance(prefs, dict) else None
    out = list(DEFAULT_TRACK_TERMS)
    out.extend(str(t).strip().lower() for t in (tracks or []) if str(t).strip())
    return list(dict.fromkeys(out))


def _escape_latex(title: str) -> str:
    """Escape the LaTeX specials %,&,#,_,$ that would otherwise break a headline.
    Braces are intentionally left alone — balance is validated by _safe_bullet."""
    out = []
    for c in title:
        out.append("\\" + c if c in "%&#_$" else c)
    return "".join(out)


def _safe_headline(title: str) -> str | None:
    """Escape the title and re-use _safe_bullet's LaTeX/length checks. Returns the
    escaped, injection-safe string, or None if it cannot be made safe (e.g. unbalanced
    braces or over length)."""
    escaped = _escape_latex((title or "").strip())
    if not escaped:
        return None
    return escaped if _safe_bullet(escaped, "\x00") == escaped else None


def _plan_headline(tex: str, title: str, profile: dict) -> str | None:
    """Decide whether a target-title headline may be injected, returning the safe,
    escaped headline text or None. Logs the reason on every skip. Gates on: (1) the
    applied-for title plausibly matching a target track, (2) LaTeX safety after
    escaping, (3) the header name line being present to anchor the insert."""
    terms = _track_terms(profile)
    hay = (title or "").lower()
    if not any(keywords._pattern(t).search(hay) for t in terms):
        log.info("target-title headline skipped: %r matches no target track %s", title, terms)
        return None
    safe = _safe_headline(title)
    if safe is None:
        log.info("target-title headline skipped: %r is not LaTeX-safe after escaping", title)
        return None
    if _name_line(tex) is None:
        log.info("target-title headline skipped: no header name line to anchor under")
        return None
    return safe


def _name_line(tex: str) -> "re.Match | None":
    r"""Match the full header name line — \textbf{\Huge ... \\ \vspace{..} — so the
    insert lands on the NEXT line and the real name line stays byte-identical."""
    return re.search(r"\\textbf\{\\Huge\b.*?\\\\(?:\s*\\vspace\{[^}]*\})?", tex)


def _inject_headline(tex: str, headline: str) -> str:
    """Insert a single target-title line immediately after the header name line.
    No-op (returns tex unchanged) if the name line is absent."""
    m = _name_line(tex)
    if m is None:
        return tex
    line = f"\n    \\textbf{{\\large {headline}}} \\\\ \\vspace{{1pt}}"
    return tex[:m.end()] + line + tex[m.end():]


def _rewrite(bullets: list[str], company: str, title: str, emphasis: str, jd: str,
             bank: dict, inject: list[str] | None = None) -> list[str]:
    numbered = "\n".join(f"{i}. {b}" for i, b in enumerate(bullets))
    principles = "\n".join(f"- {p}" for p in bank.get("principles", []))
    # Keyword-coverage lever: nudge (never force) the model to surface employer keywords
    # the candidate genuinely holds. Added AFTER the HARD RULES so the truthfulness guard
    # reads first and is not weakened; empty candidate list adds no block at all.
    inject_block = (
        "\nWhere TRUTHFUL and natural, incorporate these employer keywords the candidate "
        f"genuinely has: {', '.join(inject)}.\n" if inject else ""
    )
    prompt = (
        f"Tailor a resume for: {title} at {company}.\n"
        f"Emphasis for this kind of company: {emphasis}\n"
        + (f"Job description:\n{jd}\n" if jd else "")
        + f"\nStyle rules:\n{principles}\n\n"
        "HARD RULES:\n"
        "- TRUTHFUL: only reword / re-emphasize. NEVER invent metrics, tools, or facts.\n"
        "- Copy tool names, library names, VERSION NUMBERS, proper nouns, and metrics "
        "CHARACTER-FOR-CHARACTER. Do not 'correct' them (e.g. if it says YOLOv26, keep "
        "YOLOv26 exactly — never change it to YOLOv8). Keep every number exactly as given.\n"
        "- LaTeX-safe: keep escapes like \\% and \\& intact; valid LaTeX text only; no new macros.\n"
        "- <= 115 characters per bullet, one line. Keep the SAME count and order.\n"
        + inject_block
        + f"\nBullets:\n{numbered}\n\n"
        f"Return each rewritten bullet on its own line as:  INDEX|||BULLET\n"
        f"Output exactly {len(bullets)} lines, nothing else. "
        "(Delimiter format, not JSON — LaTeX backslashes are fine.)"
    )
    text = llm.complete(prompt, max_tokens=2048)
    if text is None:
        log.warning("no LLM completion — leaving bullets unchanged")
        return bullets
    return _apply_rewrites(text, bullets)


def _tighten(bullets: list[str], company: str, title: str, issues: list[str]) -> list[str]:
    """Second-pass shortening when the rendered resume overflows one page. Same truthfulness
    guard as _rewrite; only cuts words, never facts."""
    numbered = "\n".join(f"{i}. {b}" for i, b in enumerate(bullets))
    prompt = (
        f"A tailored one-page resume for {title} at {company} does not fit: {'; '.join(issues)}.\n"
        "Shorten these bullets so the resume fits ONE page. Cut the weakest words and redundancy.\n"
        "HARD RULES:\n"
        "- Keep every metric, tool name, library name, VERSION NUMBER, and proper noun "
        "CHARACTER-FOR-CHARACTER. Never invent, alter, or drop a fact.\n"
        "- LaTeX-safe: keep escapes like \\% and \\& intact; no new macros.\n"
        "- Aim <= ~95 characters per bullet, one line. Keep the SAME count and order.\n\n"
        f"Bullets:\n{numbered}\n\n"
        f"Return each shortened bullet as  INDEX|||BULLET , exactly {len(bullets)} lines, nothing else."
    )
    text = llm.complete(prompt, max_tokens=2048)
    if text is None:
        log.warning("no LLM completion — leaving bullets unchanged")
        return bullets
    return _apply_rewrites(text, bullets)


def _compile_inspect(tex_path: str) -> tuple[str | None, list[str]]:
    """Compile with tectonic (preferred) or xelatex (fallback) and report soft layout issues.

    Returns (pdf_path or None, issues). A resume that overflows still yields a PDF we keep
    as best-effort; issues drive the tightening retry loop in tailor()."""
    if shutil.which("tectonic") is not None:
        try:
            proc = subprocess.run(["tectonic", tex_path, "--outdir", OUT_DIR],
                                  capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            log.error("tectonic timed out")
            return None, []
        if proc.returncode != 0:
            log.error("tectonic compile failed:\n%s", proc.stderr[-800:])
            return None, []
        combined = proc.stdout + proc.stderr
    elif shutil.which("xelatex") is not None:
        log.info("tectonic not found — falling back to xelatex")
        try:
            proc = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", f"-output-directory={OUT_DIR}", tex_path],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            log.error("xelatex timed out")
            return None, []
        if proc.returncode != 0:
            log.error("xelatex compile failed:\n%s", proc.stdout[-800:])
            return None, []
        combined = proc.stdout + proc.stderr
    else:
        log.error("no LaTeX compiler found (install tectonic or xelatex)")
        return None, []
    pdf = os.path.splitext(tex_path)[0] + ".pdf"
    if not os.path.exists(pdf):
        return None, []
    with open(pdf, "rb") as f:
        pages = _count_pages(f.read())
    issues = []
    if pages > 1:
        issues.append(f"{pages} pages (resume must be 1)")
    if "overfull \\hbox" in combined.lower():
        issues.append("overfull hbox (line spills the margin)")
    return pdf, issues


def tailor(company: str, title: str, archetype: str, jd: str, role_title: bool = False,
           tailor_titles: bool = False, select_bullets: bool = False) -> str | None:
    """Tailor the base resume for one role and return the path to the PDF to ship.

    Returns the tailored PDF path when a tailored PDF was shipped (success OR residual
    layout issue), or the base resume_path when it fell back (missing base .tex, fact-gate
    failure, or compile unavailable/failed)."""
    profile = _load_yaml(os.path.join(ROOT, "profile", "profile.yml"))
    bank = _load_yaml(os.path.join(ROOT, "config", "resume_bank.yml"))
    tex_path = (profile.get("facts") or {}).get("resume_tex_path", "")
    # If the stored absolute path doesn't resolve (e.g. running in a sandbox/CI with a
    # different mount point), fall back to the same filename relative to ROOT.
    if tex_path and not os.path.exists(tex_path):
        rel = os.path.join(ROOT, os.path.relpath(tex_path, os.path.commonpath([ROOT, tex_path]))
                           if os.path.isabs(tex_path) else tex_path)
        # simpler: just take the last two path components (e.g. profile/resume.tex)
        parts = tex_path.replace("\\", "/").split("/")
        for n in range(2, min(5, len(parts)) + 1):
            candidate = os.path.join(ROOT, *parts[-n:])
            if os.path.exists(candidate):
                tex_path = candidate
                log.info("resolved tex_path via ROOT fallback: %s", tex_path)
                break
    if not tex_path or not os.path.exists(tex_path):
        base_pdf = (profile.get("facts") or {}).get("resume_path", "")
        print(f"base .tex not found: {tex_path!r} — set facts.resume_tex_path")
        return base_pdf

    with open(tex_path) as f:
        tex = f.read()

    # Role-title tailoring: swap experience titles for their best JD match from the
    # user-APPROVED alternate set (seniority locked, canonical wins ties). Applied to the
    # base tex first so bullet spans are computed against the retitled document.
    if tailor_titles:
        tex, title_changes = roletitles.apply(tex, jd, profile)
        if title_changes:
            print("\nTITLE CHANGES (approved alternates — review before sending):")
            for c in title_changes:
                print(f"  {c['company']}: {c['old']}  ->  {c['new']}")
        else:
            print("\n(no role-title changes: no approved alternate out-matched the canonical)")

    # Bullet-pool selection: pick the most JD-relevant subset of the pooled \resumeItem
    # bullets (active AND commented) under a one-page budget — byte-preserving, offline.
    # Runs AFTER retitling so both compose, and BEFORE _extract_items so downstream only
    # sees the newly-activated bullets (commented pool bullets are skipped by _extract_items).
    if select_bullets:
        tex, bullet_changes = resumeselect.select(tex, jd)
        if bullet_changes:
            print("\nBULLETS SELECTED (fit to one page, by JD relevance):")
            for c in bullet_changes:
                sign = "+activated" if c["action"] == "activated" else "-deactivated"
                print(f"  {c['company']}: {sign}: {c['text'][:70]}")
        else:
            print("\n(no bullet changes)")

    items = _extract_items(tex)
    bullets = [c for _, _, c in items]
    emphasis = (bank.get("company_archetype_emphasis", {}) or {}).get(
        archetype, "Balance impact, ownership, and rigor.")

    # Keyword-coverage lever: measure the CURRENT resume against the JD, then pass the
    # missing keywords the candidate GENUINELY has into the rewrite as a truthful nudge.
    jd_keywords = keywords.extract_jd_keywords(jd)
    profile_terms = keywords.profile_skill_terms(profile)
    before_frac, before_present, missing = keywords.coverage(tex, jd_keywords)
    inject = keywords.truthful_injection_candidates(missing, profile_terms)
    if inject:
        log.info("injecting %d truthful employer keyword(s): %s", len(inject), ", ".join(inject))

    new = _rewrite(bullets, company, title, emphasis, jd, bank, inject=inject)

    # Target-title headline: decide once (plausibility + LaTeX safety + anchor present),
    # then apply the same string insert on each compile attempt.
    headline = _plan_headline(tex, title, profile) if role_title else None
    if not role_title:
        log.info("--no-role-title: skipping target-title headline")

    os.makedirs(OUT_DIR, exist_ok=True)
    slug = f"{company}_{title}".lower().replace(" ", "_").replace("/", "-")
    tailored_tex = os.path.join(OUT_DIR, f"{slug}.tex")

    # Compile → inspect → tighten loop: if the resume overflows one page, shorten the
    # bullets and recompile (max 2 retries), then keep the best-effort PDF.
    pdf: str | None = None
    issues: list[str] = []
    rendered = ""
    for attempt in range(3):
        rendered = _render(tex, items, new)
        if headline:
            rendered = _inject_headline(rendered, headline)
        with open(tailored_tex, "w") as f:
            f.write(_xetex_compat(rendered))
        pdf, issues = _compile_inspect(tailored_tex)
        if pdf is None or not issues:
            break
        if attempt < 2:
            log.warning("layout issue (%s) — tightening", "; ".join(issues))
            new = _tighten(new, company, title, issues)

    changed = sum(nb != old for (_, _, old), nb in zip(items, new))
    print(f"\n{changed}/{len(bullets)} bullets reworded (template untouched). Diff:")
    for (_, _, old), nb in zip(items, new):
        if nb != old:
            print(f"  - {old}\n  + {nb}")

    # Coverage report: recompute on the rewritten resume and show the before→after delta.
    if jd_keywords:
        after_frac, after_present, _ = keywords.coverage(rendered, jd_keywords)
        gained = len(after_present) - len(before_present)
        print(f"\nJD keyword coverage: {round(before_frac * 100)}% -> "
              f"{round(after_frac * 100)}% ({gained:+d} keywords)")

    # Fact gate: the safety rail behind aggressive keyword injection. If the rewritten
    # resume introduces any ungrounded numeric claim, refuse to ship the tailored PDF.
    ok, violations = verify_facts.verify(rendered, verify_facts.load_sources())
    base_pdf = (profile.get("facts") or {}).get("resume_path", "")

    if not ok:
        log.error("FACT GATE FAILED — ungrounded metric(s): %s", violations)
        print(f"\n✗ fact gate failed (ungrounded metrics: {', '.join(violations)}) — "
              f"shipping base PDF, not the tailored one: {base_pdf}")
        return base_pdf
    elif pdf and not issues:
        print(f"\n✓ compiled -> {pdf}")
        return pdf
    elif pdf:
        print(f"\n⚠ compiled with residual layout issue ({'; '.join(issues)}) -> {pdf}")
        return pdf
    else:
        print(f"\n✗ compile unavailable/failed — fall back to base PDF: {base_pdf}")
        return base_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--company", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--archetype", default="startup",
                    choices=["big_tech", "startup", "quant", "research_lab", "product"])
    ap.add_argument("--jd-file", default="")
    ap.add_argument("--role-title", action=argparse.BooleanOptionalAction, default=False,
                    help="opt in to a plausibility-gated target-title line under the name "
                         "(default off — a title tagline is a weak ATS signal for a student "
                         "resume; ATS read titles from the experience section)")
    ap.add_argument("--tailor-titles", action=argparse.BooleanOptionalAction, default=False,
                    help="opt in to swapping experience titles for their best JD match from "
                         "your approved role_titles set in profile.yml (default off; seniority "
                         "locked, canonical wins ties)")
    ap.add_argument("--select-bullets", action=argparse.BooleanOptionalAction, default=False,
                    help="opt in to per-JD selection of which pooled resume bullets to show "
                         "(default off; picks the most JD-relevant true bullets to fit one page)")
    args = ap.parse_args()
    jd = ""
    if args.jd_file and os.path.exists(args.jd_file):
        with open(args.jd_file) as f:
            jd = f.read()[:4000]
    tailor(args.company, args.title, args.archetype, jd, role_title=args.role_title,
           tailor_titles=args.tailor_titles, select_bullets=args.select_bullets)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    main()
