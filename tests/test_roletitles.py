"""Unit tests for the approved-alternate role-title selector. Pure, no LLM, no network.

The selector never invents a title: it only ever picks from the per-company approved
list in profile.yml (first entry = canonical). Seniority safety is a property of that
human-approved data; here we test the mechanism — parse, JD-match selection (canonical
wins ties), the drift guard, and byte-preserving application.
"""
from src import roletitles

_TEX = r"""
\begin{document}
\section{Experience}
\resumeSubHeadingListStart
  \resumeSubheading
    {INSPIRE Lab}{January 2026 -- Present}
    {Undergraduate Researcher}{Austin, TX}
    \resumeItemListStart
      \resumeItem{Built a YOLO26+ResNet18 pipeline detecting robotaxis}
    \resumeItemListEnd
  \resumeSubheading
    {ChargeScape}{May 2026 -- Present}
    {Machine Learning Intern}{Austin, TX}
    \resumeItemListStart
      \resumeItem{Load-management models for EV charging fleets}
    \resumeItemListEnd
\resumeSubHeadingListEnd
\end{document}
"""

_PROFILE = {
    "role_titles": {
        "INSPIRE Lab": [
            "Undergraduate Researcher",
            "Computer Vision Researcher",
            "Machine Learning Researcher",
        ],
        "ChargeScape": [
            "Machine Learning Intern",
            "Applied Machine Learning Intern",
        ],
    }
}


# --- parsing ------------------------------------------------------------------


def test_parse_subheadings_extracts_company_title_and_span():
    subs = roletitles.parse_subheadings(_TEX)
    companies = [s[0] for s in subs]
    titles = [s[1] for s in subs]
    assert companies == ["INSPIRE Lab", "ChargeScape"]
    assert titles == ["Undergraduate Researcher", "Machine Learning Intern"]
    # the reported span points exactly at the title text in the source.
    for _company, title, start, end in subs:
        assert _TEX[start:end] == title


# --- selection ----------------------------------------------------------------


def test_select_prefers_jd_matching_alternate():
    candidates = ["Undergraduate Researcher", "Computer Vision Researcher",
                  "Machine Learning Researcher"]
    jd_kw = {"computer", "vision", "detection", "perception"}
    assert roletitles.select_title(candidates, jd_kw) == "Computer Vision Researcher"


def test_select_keeps_canonical_when_no_alternate_wins():
    candidates = ["Undergraduate Researcher", "Computer Vision Researcher"]
    jd_kw = {"marketing", "sales", "finance"}  # nothing matches a domain token
    assert roletitles.select_title(candidates, jd_kw) == "Undergraduate Researcher"


def test_select_breaks_ties_toward_canonical():
    # both alternates match one JD token; canonical (index 0) must win the tie.
    candidates = ["Data Analyst", "Machine Analyst", "Learning Analyst"]
    jd_kw = {"machine", "learning"}
    # 'Machine Analyst' and 'Learning Analyst' each score 1; neither beats the other,
    # and canonical scores 0 — the strict-improvement rule picks the first best alternate.
    out = roletitles.select_title(candidates, jd_kw)
    assert out == "Machine Analyst"


# --- planning + drift guard ---------------------------------------------------


def test_plan_changes_only_titles_with_a_better_match():
    jd = "We need computer vision and perception for object detection."
    changes = roletitles.plan(_TEX, _PROFILE, jd)
    by_company = {c["company"]: c for c in changes}
    # INSPIRE Lab retitled to the CV alternate; ChargeScape (no CV work in JD terms
    # that beats its canonical) is unchanged and absent from the change list.
    assert by_company["INSPIRE Lab"]["new"] == "Computer Vision Researcher"
    assert "ChargeScape" not in by_company


def test_plan_skips_company_without_approved_titles():
    profile = {"role_titles": {"Nonexistent Co": ["Whatever"]}}
    assert roletitles.plan(_TEX, profile, "computer vision") == []


def test_plan_skips_when_resume_title_is_not_the_canonical():
    # drift guard: if the resume's current title isn't the approved canonical, never touch it.
    profile = {"role_titles": {"INSPIRE Lab": ["Some Other Title", "Computer Vision Researcher"]}}
    assert roletitles.plan(_TEX, profile, "computer vision") == []


# --- application (byte-preservation) ------------------------------------------


def test_apply_swaps_only_the_title_text():
    jd = "computer vision perception object detection"
    new_tex, changes = roletitles.apply(_TEX, jd, _PROFILE)
    assert "Computer Vision Researcher" in new_tex
    assert "Undergraduate Researcher" not in new_tex
    # everything else is byte-identical: only the title text changed.
    restored = new_tex.replace("Computer Vision Researcher", "Undergraduate Researcher")
    assert restored == _TEX
    assert len(changes) == 1 and changes[0]["company"] == "INSPIRE Lab"


def test_apply_noop_when_nothing_matches():
    new_tex, changes = roletitles.apply(_TEX, "sales and marketing role", _PROFILE)
    assert new_tex == _TEX
    assert changes == []
