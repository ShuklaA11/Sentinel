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

from . import llm

log = logging.getLogger("tailor")

ROOT = os.path.dirname(os.path.dirname(__file__))
MARKER = "\\resumeItem{"
MAX_CHARS = 115
OUT_DIR = os.path.join(ROOT, "tailored")


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
    """Best-effort PDF page count from raw bytes (no poppler dep). Prefer the page-tree
    /Count; fall back to counting /Type /Page objects. Returns >=1."""
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


def _rewrite(bullets: list[str], company: str, title: str, emphasis: str, jd: str, bank: dict) -> list[str]:
    numbered = "\n".join(f"{i}. {b}" for i, b in enumerate(bullets))
    principles = "\n".join(f"- {p}" for p in bank.get("principles", []))
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
        "- <= 115 characters per bullet, one line. Keep the SAME count and order.\n\n"
        f"Bullets:\n{numbered}\n\n"
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
        return bullets
    return _apply_rewrites(text, bullets)


def _compile_inspect(tex_path: str) -> tuple[str | None, list[str]]:
    """Compile with tectonic and report soft layout issues (>1 page, overfull hbox).

    Returns (pdf_path or None, issues). A resume that overflows still yields a PDF we keep
    as best-effort; issues drive the tightening retry loop in tailor()."""
    if shutil.which("tectonic") is None:
        log.error("tectonic not installed (brew install tectonic) — cannot compile")
        return None, []
    try:
        proc = subprocess.run(["tectonic", tex_path, "--outdir", OUT_DIR],
                              capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        log.error("tectonic timed out")
        return None, []
    if proc.returncode != 0:
        log.error("tectonic compile failed:\n%s", proc.stderr[-800:])
        return None, []
    pdf = os.path.splitext(tex_path)[0] + ".pdf"
    if not os.path.exists(pdf):
        return None, []
    with open(pdf, "rb") as f:
        pages = _count_pages(f.read())
    issues = []
    if pages > 1:
        issues.append(f"{pages} pages (resume must be 1)")
    if "overfull \\hbox" in (proc.stdout + proc.stderr).lower():
        issues.append("overfull hbox (line spills the margin)")
    return pdf, issues


def tailor(company: str, title: str, archetype: str, jd: str) -> str | None:
    profile = _load_yaml(os.path.join(ROOT, "profile", "profile.yml"))
    bank = _load_yaml(os.path.join(ROOT, "config", "resume_bank.yml"))
    tex_path = (profile.get("facts") or {}).get("resume_tex_path", "")
    if not tex_path or not os.path.exists(tex_path):
        base_pdf = (profile.get("facts") or {}).get("resume_path", "")
        print(f"base .tex not found: {tex_path!r} — set facts.resume_tex_path")
        return base_pdf

    with open(tex_path) as f:
        tex = f.read()
    items = _extract_items(tex)
    bullets = [c for _, _, c in items]
    emphasis = (bank.get("company_archetype_emphasis", {}) or {}).get(
        archetype, "Balance impact, ownership, and rigor.")

    new = _rewrite(bullets, company, title, emphasis, jd, bank)

    os.makedirs(OUT_DIR, exist_ok=True)
    slug = f"{company}_{title}".lower().replace(" ", "_").replace("/", "-")
    tailored_tex = os.path.join(OUT_DIR, f"{slug}.tex")

    # Compile → inspect → tighten loop: if the resume overflows one page, shorten the
    # bullets and recompile (max 2 retries), then keep the best-effort PDF.
    pdf: str | None = None
    issues: list[str] = []
    for attempt in range(3):
        with open(tailored_tex, "w") as f:
            f.write(_xetex_compat(_render(tex, items, new)))
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

    if pdf and not issues:
        print(f"\n✓ compiled -> {pdf}")
        return pdf
    elif pdf:
        print(f"\n⚠ compiled with residual layout issue ({'; '.join(issues)}) -> {pdf}")
        return pdf
    else:
        base_pdf = (profile.get("facts") or {}).get("resume_path", "")
        print(f"\n✗ compile unavailable/failed — fall back to base PDF: {base_pdf}")
        return base_pdf


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
            jd = f.read()[:4000]
    tailor(args.company, args.title, args.archetype, jd)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    main()
