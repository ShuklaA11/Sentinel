"""Per-JD resume role-title tailoring from a user-APPROVED alternate-titles set.

This is the safe form of the strategy guide's "Role Title Trick": the model never
invents a title. The candidate curates a small list of truthful alternates per role in
`profile.yml` under `role_titles:` (keyed by the company string exactly as it appears in
`resume.tex`; first entry = canonical). For a given JD this module deterministically
picks the approved title whose domain best matches the JD keywords — no LLM, so there is
no fabrication surface and it runs without API credits.

Guarantees:
  - Never emits a title that isn't in the approved list (truthfulness = human pre-approval).
  - Canonical wins ties, so a title only changes when an alternate STRICTLY out-matches it.
  - Drift guard: if the resume's current title isn't the approved canonical, it is left
    untouched (a config mismatch must not trigger a silent rewrite).
  - Seniority preservation is a property of the approved data (Intern/Researcher/Engineer
    kept across every alternate), not enforced here.
"""
from __future__ import annotations

import logging
import re

from . import keywords

log = logging.getLogger("roletitles")

_SUBHEADING = "\\resumeSubheading"

# Generic role/level words that shouldn't drive JD matching — only the domain does.
_GENERIC = frozenset({
    "intern", "researcher", "research", "engineer", "developer", "analyst",
    "scientist", "assistant", "undergraduate", "graduate", "applied", "co-op",
    "coop", "fellow", "lead", "senior", "junior", "staff", "of", "and", "the",
})


def _brace_group(tex: str, i: int) -> tuple[str, int, int] | None:
    """Read a balanced {...} group starting at the '{' at or after index i (skipping
    whitespace). Returns (inner_text, inner_start, inner_end) or None if not a group."""
    while i < len(tex) and tex[i].isspace():
        i += 1
    if i >= len(tex) or tex[i] != "{":
        return None
    depth, j = 1, i + 1
    while j < len(tex) and depth > 0:
        depth += (tex[j] == "{") - (tex[j] == "}")
        j += 1
    return tex[i + 1: j - 1], i + 1, j - 1


def parse_subheadings(tex: str) -> list[tuple[str, str, int, int]]:
    """Locate each \\resumeSubheading and return (company, title, title_start, title_end).

    The macro takes four brace groups: {company}{dates}{title}{location}. We return the
    company (group 1) and the *span* of the title text (group 3) so a caller can swap the
    title in place while byte-preserving the rest of the document.
    """
    out: list[tuple[str, str, int, int]] = []
    i = tex.find(_SUBHEADING)
    while i != -1:
        pos = i + len(_SUBHEADING)
        groups = []
        for _ in range(4):
            g = _brace_group(tex, pos)
            if g is None:
                break
            groups.append(g)
            pos = g[2] + 1  # continue just after this group's closing brace
        if len(groups) == 4:
            company = groups[0][0].strip()
            title_text, t_start, t_end = groups[2]
            out.append((company, title_text.strip(), t_start, t_end))
        i = tex.find(_SUBHEADING, pos)
    return out


def approved_titles(profile: dict) -> dict[str, list[str]]:
    """The per-company approved title lists from profile.yml (empty when absent)."""
    raw = (profile or {}).get("role_titles") or {}
    return {str(k): [str(t) for t in (v or [])] for k, v in raw.items() if v}


def _domain_score(title: str, jd_kw: set[str]) -> int:
    """How many of the title's DOMAIN tokens appear in the JD keyword set.

    Generic role/level words are ignored so 'Researcher' vs 'Intern' never sways the
    choice — only the specialization ('computer', 'vision', ...) does.
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", title.lower()) if t and t not in _GENERIC]
    return sum(1 for t in tokens if t in jd_kw)


def select_title(candidates: list[str], jd_kw: set[str]) -> str:
    """Pick the approved title best matching the JD. candidates[0] is canonical.

    An alternate is chosen only if it STRICTLY out-scores the canonical; ties resolve to
    the earliest candidate, so the canonical is never displaced by an equal match.
    """
    if not candidates:
        return ""
    best, best_score = candidates[0], _domain_score(candidates[0], jd_kw)
    for cand in candidates[1:]:
        if _domain_score(cand, jd_kw) > best_score:
            best, best_score = cand, _domain_score(cand, jd_kw)
    return best


def plan(tex: str, profile: dict, jd: str) -> list[dict]:
    """Compute the title changes for this JD as a list of
    {company, old, new, start, end}, in document order. Only rows where the title
    actually changes are returned; unmatched companies and drifted titles are skipped.
    """
    approved = approved_titles(profile)
    jd_kw = set(keywords.extract_jd_keywords(jd))
    changes: list[dict] = []
    for company, title, start, end in parse_subheadings(tex):
        candidates = approved.get(company)
        if not candidates:
            continue
        if title != candidates[0]:
            log.warning(
                "role-title: %r resume title %r != approved canonical %r — left untouched",
                company, title, candidates[0])
            continue
        chosen = select_title(candidates, jd_kw)
        if chosen and chosen != title:
            changes.append({"company": company, "old": title, "new": chosen,
                            "start": start, "end": end})
    return changes


def apply(tex: str, jd: str, profile: dict) -> tuple[str, list[dict]]:
    """Return (new_tex, changes): the resume with title spans swapped for their best JD
    match, byte-preserving everything else. Patches from last span to first so earlier
    offsets stay valid."""
    changes = plan(tex, profile, jd)
    out = tex
    for c in sorted(changes, key=lambda x: -x["start"]):
        out = out[:c["start"]] + c["new"] + out[c["end"]:]
    return out, changes
