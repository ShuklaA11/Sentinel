"""JD keyword-coverage engine — the strategy guide's core lever.

Mechanically compares a job description against the candidate's CV/profile:
which requested skills are already present, which are missing, and which of the
missing ones the candidate *genuinely* has and could truthfully surface.

Pure and offline: no network, no file I/O, no LLM. Fully unit-testable.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

log = logging.getLogger("keywords")

# Common English words + generic job-posting filler that carry no skill signal.
STOPWORDS: frozenset[str] = frozenset({
    # articles / conjunctions / prepositions
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "as", "at", "by", "from", "into", "over", "under", "up",
    "down", "out", "off", "about", "across", "within", "than", "so", "too",
    "very", "just", "also", "while", "both",
    # pronouns / determiners
    "this", "that", "these", "those", "it", "its", "we", "you", "your", "yours",
    "our", "ours", "us", "they", "them", "their", "he", "she", "his", "her",
    "him", "i", "me", "my", "mine", "who", "what", "why", "how", "where",
    "when", "which", "all", "any", "each", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "no", "nor", "not",
    # auxiliary / common verbs
    "is", "are", "be", "been", "being", "was", "were", "will", "would", "can",
    "could", "should", "may", "might", "must", "do", "does", "did", "done",
    "has", "have", "had", "having",
    # contraction fragments left by tokenizing
    "s", "t", "re", "ve", "ll", "d", "m", "o",
    # generic job-posting filler (required set + neighbours)
    "work", "team", "role", "strong", "ability", "experience", "skills",
    "looking", "join", "candidate", "candidates", "ideal", "plus", "preferred",
    "required", "requirements", "responsibilities", "responsibility", "help",
    "using", "use", "well", "including", "include", "etc", "new", "great",
    "good", "excellent", "related", "relevant", "years", "year", "company",
})

# Curated multi-word tech/skill phrases captured as single keywords when their
# two words appear adjacent in the text.
_BIGRAMS: frozenset[str] = frozenset({
    "machine learning",
    "computer vision",
    "deep learning",
    "data pipeline",
    "reinforcement learning",
    "time series",
    # a few common neighbours in the same vein
    "natural language",
    "data science",
    "neural networks",
    "large language",
})


@lru_cache(maxsize=None)
def _pattern(term: str) -> "re.Pattern":
    """Word-boundary match (same idea as src.filter._pattern) so 'intern' does
    not hit 'international' and 'java' does not hit 'javascript'. Works for
    multi-word phrases too — the boundary applies to the phrase edges."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on any run of non-alphanumeric characters."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def extract_jd_keywords(jd_text: str) -> list[str]:
    """Extract skill keywords from a job description.

    Keeps meaningful unigrams (stopwords dropped) plus curated multi-word
    bigrams captured when both words are adjacent. Deduped, first-seen order.
    """
    tokens = _tokenize(jd_text or "")
    out: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        if term and term not in seen:
            seen.add(term)
            out.append(term)

    n = len(tokens)
    for i, tok in enumerate(tokens):
        if i + 1 < n:
            bigram = f"{tok} {tokens[i + 1]}"
            if bigram in _BIGRAMS:
                _add(bigram)
        if tok not in STOPWORDS:
            _add(tok)

    log.debug("extracted %d jd keywords", len(out))
    return out


def coverage(cv_text: str, jd_keywords: list[str]) -> tuple[float, list[str], list[str]]:
    """Score how many JD keywords appear in the CV via word-boundary matching.

    Returns (fraction_present_0_to_1, present_keywords, missing_keywords).
    Fraction is 0.0 for empty jd_keywords (no division by zero).
    """
    if not jd_keywords:
        return 0.0, [], []

    text = (cv_text or "").lower()
    present: list[str] = []
    missing: list[str] = []
    for keyword in jd_keywords:
        if _pattern(keyword).search(text):
            present.append(keyword)
        else:
            missing.append(keyword)

    fraction = len(present) / len(jd_keywords)
    return fraction, present, missing


def truthful_injection_candidates(
    missing_jd_keywords: list[str], profile_skills: list[str]
) -> list[str]:
    """Of the missing JD keywords, return only those the candidate genuinely has.

    A keyword is genuinely held when it appears as a word-boundary match inside
    the candidate's profile skills (case-insensitive, phrase-aware): profile
    'PyTorch' surfaces 'pytorch'; 'machine learning frameworks' surfaces
    'machine learning'. A keyword with no profile backing is never returned.
    """
    if not missing_jd_keywords or not profile_skills:
        return []

    haystack = " ".join(str(s).lower() for s in profile_skills if s)
    out: list[str] = []
    seen: set[str] = set()
    for keyword in missing_jd_keywords:
        if keyword in seen:
            continue
        if _pattern(keyword).search(haystack):
            seen.add(keyword)
            out.append(keyword)
    return out


def profile_skill_terms(profile: dict) -> list[str]:
    """Flatten the candidate's skills into a deduped, lowercased term list.

    Reads profile.yml's real structure defensively: skills.languages,
    skills.frameworks, skills.tools, preferences.skills, and domain_tags.
    Never raises on a missing or empty key.
    """
    profile = profile or {}

    skills = profile.get("skills") or {}
    if not isinstance(skills, dict):
        skills = {}
    prefs = profile.get("preferences") or {}
    if not isinstance(prefs, dict):
        prefs = {}

    raw: list = []
    for key in ("languages", "frameworks", "tools"):
        raw.extend(skills.get(key) or [])
    raw.extend(prefs.get("skills") or [])
    raw.extend(profile.get("domain_tags") or [])

    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        term = str(item).strip().lower()
        if term and term not in seen:
            seen.add(term)
            out.append(term)
    return out
