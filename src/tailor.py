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

log = logging.getLogger("tailor")

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL = os.environ.get("TAILOR_MODEL", "claude-sonnet-4-6")
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


def _xetex_compat(tex: str) -> str:
    """Comment out pdfTeX-only lines tectonic's XeTeX engine can't run. Visually a no-op:
    XeTeX is Unicode-native, so the PDF stays ATS-readable without glyphtounicode."""
    tex = tex.replace("\\input{glyphtounicode}", "%\\input{glyphtounicode}")
    tex = tex.replace("\\pdfgentounicode=1", "%\\pdfgentounicode=1")
    return tex


def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _rewrite(bullets: list[str], company: str, title: str, emphasis: str, jd: str, bank: dict) -> list[str]:
    client = _client()
    if client is None:
        log.warning("no ANTHROPIC_API_KEY — leaving bullets unchanged")
        return bullets
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
    try:
        resp = client.messages.create(model=MODEL, max_tokens=2048,
                                      messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text
        out = dict(enumerate(bullets))  # default to originals; override with safe rewrites
        for line in text.splitlines():
            m = re.match(r"\s*(\d+)\s*\|\|\|\s*(.*)", line)
            if m and int(m.group(1)) < len(bullets):
                idx = int(m.group(1))
                out[idx] = _safe_bullet(m.group(2).strip(), bullets[idx])
        return [out[i] for i in range(len(bullets))]
    except Exception as exc:  # noqa: BLE001
        log.error("rewrite failed (%s) — keeping originals", exc)
        return bullets


def _compile(tex_path: str) -> str | None:
    if shutil.which("tectonic") is None:
        log.error("tectonic not installed (brew install tectonic) — cannot compile")
        return None
    try:
        subprocess.run(["tectonic", tex_path, "--outdir", OUT_DIR],
                       check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        log.error("tectonic compile failed:\n%s", exc.stderr[-800:])
        return None
    pdf = os.path.splitext(tex_path)[0] + ".pdf"
    return pdf if os.path.exists(pdf) else None


def tailor(company: str, title: str, archetype: str, jd: str) -> None:
    profile = _load_yaml(os.path.join(ROOT, "profile", "profile.yml"))
    bank = _load_yaml(os.path.join(ROOT, "config", "resume_bank.yml"))
    tex_path = (profile.get("facts") or {}).get("resume_tex_path", "")
    if not tex_path or not os.path.exists(tex_path):
        print(f"base .tex not found: {tex_path!r} — set facts.resume_tex_path"); return

    with open(tex_path) as f:
        tex = f.read()
    items = _extract_items(tex)
    bullets = [c for _, _, c in items]
    emphasis = (bank.get("company_archetype_emphasis", {}) or {}).get(
        archetype, "Balance impact, ownership, and rigor.")

    new = _rewrite(bullets, company, title, emphasis, jd, bank)

    # Rebuild from last span to first so indices stay valid.
    out = tex
    changed = 0
    for (start, end, old), nb in sorted(zip(items, new), key=lambda x: -x[0][0]):
        out = out[:start] + MARKER + nb + "}" + out[end:]
        changed += nb != old

    os.makedirs(OUT_DIR, exist_ok=True)
    slug = f"{company}_{title}".lower().replace(" ", "_").replace("/", "-")
    tailored_tex = os.path.join(OUT_DIR, f"{slug}.tex")
    with open(tailored_tex, "w") as f:
        f.write(_xetex_compat(out))

    print(f"\n{changed}/{len(bullets)} bullets reworded (template untouched). Diff:")
    for (_, _, old), nb in zip(items, new):
        if nb != old:
            print(f"  - {old}\n  + {nb}")

    pdf = _compile(tailored_tex)
    if pdf:
        print(f"\n✓ compiled -> {pdf}")
    else:
        base_pdf = (profile.get("facts") or {}).get("resume_path", "")
        print(f"\n✗ compile unavailable/failed — fall back to base PDF: {base_pdf}")


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
