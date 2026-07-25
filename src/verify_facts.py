"""Numeric fact-check gate — the safety rail behind aggressive keyword injection.

When we lean on an LLM to surface a candidate's real skills more forcefully, the
failure mode we most fear is *numeric* fabrication: an invented percentage, a
made-up dollar figure, a multiplier or headcount that never appears in the
ground truth. This module catches exactly that, mechanically and offline.

`verify` extracts every metric-shaped claim from generated text and requires each
one to be grounded in the source material (profile.yml + resume.tex) or in an
explicit allow-list. Anything ungrounded is a violation.

LIMITATION (by design): this gate catches NUMERIC fabrication ONLY. Qualitative
fabrication — e.g. 'expert in Kubernetes', 'led the team', 'deep RL background' —
carries no number to check and therefore stays prompt-trust, the same boundary
the career-ops port draws. Do not treat a green result here as a truthfulness
guarantee for prose claims; it only certifies that the numbers are backed.

Pure and offline: no network, no LLM. Fully unit-testable.
"""
from __future__ import annotations

import logging
import os
import re

import yaml

log = logging.getLogger("verify_facts")

ROOT = os.path.dirname(os.path.dirname(__file__))
PROFILE_YML = os.path.join(ROOT, "profile", "profile.yml")
RESUME_TEX = os.path.join(ROOT, "profile", "resume.tex")

# Numeric core: digits with optional thousands commas and an optional decimal.
_NUM = r"\d[\d,]*(?:\.\d+)?"

# A digit that opens a free-standing number must not be glued to a preceding
# word char or decimal point, so 'A100' and the '250' inside '1.250' are not
# mistaken for their own metrics.
_LEAD = r"(?<![\w.])"

# One combined scanner. Alternatives are tried left-to-right at each position,
# so ordering is the precedence: percent, currency, multiplier, then N-unit count.
_METRIC_RE = re.compile(
    rf"""
      {_LEAD}(?P<pct>{_NUM})\s*%                                  # 94%   3,050%
    | \$\s*(?P<cur>{_NUM})(?P<cur_suf>[kmb])?                     # $1.2M  $50,000
    | {_LEAD}(?P<mult>{_NUM})x\b                                  # 3x   10X
    | {_LEAD}(?P<cnt>{_NUM})(?P<cnt_suf>[kmb])?\+?\s+(?P<unit>[a-z]+)  # 250+ hours
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _num(core: str) -> str:
    """Canonical numeric core: strip thousands commas ('2,000' -> '2000')."""
    return core.replace(",", "")


def extract_metrics(text: str) -> set[str]:
    """Extract metric-shaped claims and normalize each to a canonical form.

    Normalization strips commas from the numeric core, lowercases units and
    magnitude suffixes, and drops a trailing '+', so '2,000', '2000' and
    '2,000+' all collapse to the same token. Returns a set of canonical strings
    such as '94%', '$1.2m', '10x', '250 hours', '100k samples'.
    """
    out: set[str] = set()
    for m in _METRIC_RE.finditer(text or ""):
        if m.group("pct") is not None:
            out.add(f"{_num(m.group('pct'))}%")
        elif m.group("cur") is not None:
            suf = (m.group("cur_suf") or "").lower()
            out.add(f"${_num(m.group('cur'))}{suf}")
        elif m.group("mult") is not None:
            out.add(f"{_num(m.group('mult'))}x")
        elif m.group("cnt") is not None:
            suf = (m.group("cnt_suf") or "").lower()
            out.add(f"{_num(m.group('cnt'))}{suf} {m.group('unit').lower()}")
    return out


def _canon_allow(allow: set[str] | None) -> set[str]:
    """Canonicalize allow-list entries the same way sources are canonicalized,
    so a caller may pass raw human forms ('$1.2M') or canonical ones ('$1.2m').
    An entry that yields no metric (e.g. a bare token) is kept verbatim."""
    canon: set[str] = set()
    for entry in allow or set():
        found = extract_metrics(entry)
        canon |= found if found else {entry}
    return canon


def verify(
    generated_text: str,
    source_texts: list[str],
    allow: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Check that every metric in `generated_text` is grounded.

    A metric is grounded if it appears in the union of metrics extracted from
    `source_texts`, or in `allow`. `violations` is the sorted list of generated
    metrics with no such backing; `ok` is True exactly when there are none.
    Empty or whitespace-only generated text is vacuously OK.
    """
    if not (generated_text or "").strip():
        return (True, [])

    generated = extract_metrics(generated_text)
    grounded: set[str] = _canon_allow(allow)
    for source in source_texts:
        grounded |= extract_metrics(source)

    violations = sorted(m for m in generated if m not in grounded)
    return (violations == [], violations)


def _read_text(path: str) -> str | None:
    """Read a file's text, returning None (and logging) if it is unavailable."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        log.warning("fact source missing: %s", path)
    except OSError as e:
        log.warning("fact source unreadable: %s (%s)", path, e)
    return None


def load_sources() -> list[str]:
    """Gather ground-truth text from profile.yml + resume.tex.

    profile.yml is parsed and re-serialized to a flat string (falling back to
    str() if dumping fails); resume.tex is taken raw. Missing or unreadable
    files are skipped — the returned list holds whatever was readable and may
    be empty.
    """
    sources: list[str] = []

    raw_yml = _read_text(PROFILE_YML)
    if raw_yml is not None:
        try:
            data = yaml.safe_load(raw_yml)
            sources.append(yaml.safe_dump(data) if data is not None else raw_yml)
        except yaml.YAMLError as e:
            log.warning("profile.yml parse failed, using raw text (%s)", e)
            sources.append(raw_yml)

    raw_tex = _read_text(RESUME_TEX)
    if raw_tex is not None:
        sources.append(raw_tex)

    return sources
