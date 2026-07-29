"""Per-JD bullet-pool selector — pick the active \\resumeItem subset per experience.

A resume carries more real bullets than fit on one page. This module treats every
\\resumeItem (active AND commented) as a POOL, scores each against the JD, and chooses
which to activate per experience under a one-page budget — uncommenting the winners and
commenting the losers while byte-preserving everything else.

Pure and offline: no LLM, no network, no file I/O — it operates on a tex string and is
fully unit-testable. Graceful by design: never crashes and no-ops when there is no pool.

Guarantees:
  - Every experience keeps >= 1 active bullet (all-zero-score -> its first/canonical one).
  - No commented pool bullets + budget >= active count -> exact no-op (new_tex == tex).
  - Toggling only ever adds/removes a leading '% ' on the item's line; uncommented bodies
    stay byte-identical.
"""
from __future__ import annotations

import logging
import re

from . import keywords, roletitles

log = logging.getLogger("resumeselect")

MARKER = "\\resumeItem{"
LIST_START = "\\resumeItemListStart"
LIST_END = "\\resumeItemListEnd"
PROJECT_HEADING = "\\resumeProjectHeading"
_TEXTBF = re.compile(r"^\\textbf\{(.*)\}$")


def _commented(tex: str, pos: int) -> bool:
    """True if pos sits after an unescaped % on its line (i.e. it's commented out).

    Same detection idea as tailor._commented: scan the line prefix for a '%' that is not
    itself escaped by a preceding backslash.
    """
    seg = tex[tex.rfind("\n", 0, pos) + 1: pos]
    return any(c == "%" and (i == 0 or seg[i - 1] != "\\") for i, c in enumerate(seg))


def _in_commented_list(tex: str, item_start: int) -> bool:
    """True when the \\resumeItem sits inside a COMMENTED \\resumeItemListStart block.

    Such items are template scaffolding (e.g. the Sourabh-Bajaj 'Apache Beam' example) —
    activating them would inject fake content, so they must never enter the pool. The
    enclosing list is the nearest \\resumeItemListStart that isn't already closed by a
    \\resumeItemListEnd before this item; if that start's line is commented, so is the block.
    """
    ls = tex.rfind(LIST_START, 0, item_start)
    le = tex.rfind(LIST_END, 0, item_start)
    if ls == -1 or ls < le:  # not clearly inside an open list -> don't exclude
        return False
    return _commented(tex, ls)


def parse_items(tex: str) -> list[dict]:
    """Every \\resumeItem occurrence — INCLUDING commented ones (unlike tailor._extract_items).

    Each dict is {text, start, end, commented, line_start, selectable} where:
      - text        : the {...} body (balanced braces)
      - start/end   : bracket the whole '\\resumeItem{...}' (or '% \\resumeItem{...}') span,
                      from the first non-space char on the line to just past the closing brace,
                      so the item can be rewritten wholesale
      - commented   : True when the line's first non-space char is an unescaped '%'
      - line_start  : index just after the preceding newline (start of the physical line)
      - selectable  : False when the item lives in a commented list block (template
                      scaffolding) — such items are never scored, grouped, or toggled
    """
    items: list[dict] = []
    i = 0
    while True:
        marker = tex.find(MARKER, i)
        if marker == -1:
            break
        j, depth = marker + len(MARKER), 1
        while j < len(tex) and depth > 0:
            depth += (tex[j] == "{") - (tex[j] == "}")
            j += 1
        text = tex[marker + len(MARKER): j - 1]
        line_start = tex.rfind("\n", 0, marker) + 1
        seg = tex[line_start:marker]
        start = line_start + (len(seg) - len(seg.lstrip()))  # skip indentation
        items.append({
            "text": text,
            "start": start,
            "end": j,
            "commented": _commented(tex, marker),
            "line_start": line_start,
            "selectable": not _in_commented_list(tex, marker),
        })
        i = j
    return items


def _boundaries(tex: str) -> list[tuple[int, str]]:
    """Sorted (position, label) block boundaries: every ACTIVE \\resumeSubheading (label =
    company) AND \\resumeProjectHeading (label = project name). Projects use a different
    macro, so without this they'd wrongly fold into the preceding experience.
    """
    bounds = [(tstart, company) for company, _t, tstart, _te in roletitles.parse_subheadings(tex)]
    i = tex.find(PROJECT_HEADING)
    while i != -1:
        if not _commented(tex, i):
            g = roletitles._brace_group(tex, i + len(PROJECT_HEADING))
            if g:
                name = g[0].strip()
                m = _TEXTBF.match(name)
                bounds.append((i, m.group(1).strip() if m else name))
        i = tex.find(PROJECT_HEADING, i + 1)
    bounds.sort()
    return bounds


def group_by_experience(tex: str) -> list[tuple[str, list[int]]]:
    """Group each SELECTABLE \\resumeItem index under the heading it falls beneath.

    Boundaries come from both \\resumeSubheading and \\resumeProjectHeading (see
    _boundaries), so projects form their own groups instead of folding into the last
    experience. Items before the first heading are KEPT under the synthetic key '_'.
    Non-selectable items (commented template scaffolding) are excluded entirely.
    """
    bounds = _boundaries(tex)
    items = parse_items(tex)

    def block_for(pos: int) -> int:
        block = -1
        for k, (bpos, _label) in enumerate(bounds):
            if bpos < pos:
                block = k
            else:
                break  # boundaries are in ascending position order
        return block

    order: list[int] = []
    members: dict[int, list[int]] = {}
    for idx, it in enumerate(items):
        if not it["selectable"]:
            continue  # template scaffolding in a commented block — never a real bullet
        block = block_for(it["start"])
        if block not in members:
            members[block] = []
            order.append(block)
        members[block].append(idx)

    return [("_" if b == -1 else bounds[b][1], members[b]) for b in order]


def score_item(text: str, jd_keywords: set) -> int:
    """Count of JD keywords with a word-boundary match in the bullet (case-insensitive).

    Word-boundary like src/filter.py so 'intern' doesn't hit 'international'; phrase-aware
    so multi-word JD keywords ('computer vision') count too.
    """
    if not text or not jd_keywords:
        return 0
    hay = text.lower()
    return sum(1 for kw in jd_keywords if keywords._pattern(kw).search(hay))


MIN_PER_ROLE = 2  # every experience shows at least this many bullets (pool permitting)
MAX_PER_ROLE = 4  # no experience shows more than this — keeps one role from dominating


def _allocate(groups: list[tuple[str, list[int]]], max_score: list[int], budget: int) -> list[int]:
    """Active-slot count per experience, in [MIN_PER_ROLE, MAX_PER_ROLE] (clamped to pool
    size). Each starts at its floor (min wins even over budget so no role looks sparse);
    the remaining budget goes to the highest max-score experiences first, up to their cap."""
    k = len(groups)
    pool = [len(idxs) for _c, idxs in groups]
    lo = [min(MIN_PER_ROLE, p) for p in pool]
    hi = [min(MAX_PER_ROLE, p) for p in pool]
    alloc = lo[:]
    remaining = budget - sum(lo)  # may be negative — the floors still hold (page fit is
    # the tighten loop's job), so we simply skip the discretionary top-up below.
    for g in sorted(range(k), key=lambda g: -max_score[g]):
        if remaining <= 0:
            break
        take = min(hi[g] - alloc[g], remaining)
        if take > 0:
            alloc[g] += take
            remaining -= take
    return alloc


def select(tex: str, jd: str, budget: int | None = None) -> tuple[str, list[dict]]:
    """Choose the active \\resumeItem subset per experience for this JD.

    Ranks each experience's pool by JD score (stable — ties keep document order so the
    canonical/first bullet wins), allocates a total active-bullet budget (default = the
    number of currently-active items, preserving one-page length), then returns
    (new_tex, changes): the tex with chosen items uncommented and the rest commented, and
    changes listing {company, text, action} only for items whose commented-state flipped.
    """
    items = parse_items(tex)
    if not items:
        return tex, []

    groups = group_by_experience(tex)
    jd_kw = set(keywords.extract_jd_keywords(jd))
    scores = [score_item(it["text"], jd_kw) for it in items]

    if budget is None:
        budget = sum(1 for it in items if not it["commented"])

    max_score = [max((scores[idx] for idx in idxs), default=0) for _c, idxs in groups]
    alloc = _allocate(groups, max_score, budget)

    should_active = [False] * len(items)
    company_of: dict[int, str] = {}
    for g, (company, idxs) in enumerate(groups):
        for idx in idxs:
            company_of[idx] = company
        # stable sort by score desc -> ties keep document order (first/canonical wins).
        ranked = sorted(idxs, key=lambda idx: -scores[idx])
        for idx in ranked[:alloc[g]]:
            should_active[idx] = True

    # Record flips in document order; apply them last-to-first so offsets stay valid.
    flips = [idx for idx, it in enumerate(items)
             if (not should_active[idx]) != it["commented"]]
    changes = [{
        "company": company_of[idx],
        "text": items[idx]["text"],
        "action": "activated" if should_active[idx] else "deactivated",
    } for idx in flips]

    new_tex = tex
    for idx in sorted(flips, reverse=True):
        it = items[idx]
        active_span = MARKER + it["text"] + "}"
        replacement = active_span if should_active[idx] else "% " + active_span
        new_tex = new_tex[:it["start"]] + replacement + new_tex[it["end"]:]

    return new_tex, changes
