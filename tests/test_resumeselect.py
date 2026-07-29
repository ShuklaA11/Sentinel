"""Unit tests for the per-JD bullet-pool selector. Pure, no LLM, no network.

The selector groups every \\resumeItem (active AND commented) into per-experience pools,
scores each bullet by JD relevance, then picks the active subset per experience under a
one-page budget — uncommenting the chosen bullets and commenting the rest while byte-
preserving everything else. Here we test the mechanism: parse (incl. commented),
grouping, scoring, selection, the empty-experience guard, the no-op guarantee, and the
byte-preserving toggle.
"""
from src import resumeselect

# Two experiences, each 1 active + 2 commented pool bullets with distinct domains:
# an ML/'gradient boosting' bullet (active), a 'computer vision' bullet, a 'geospatial'
# bullet (both commented) per experience.
_TEX = r"""
\begin{document}
\section{Experience}
\resumeSubHeadingListStart
  \resumeSubheading
    {INSPIRE Lab}{January 2026 -- Present}
    {Machine Learning Researcher}{Austin, TX}
    \resumeItemListStart
      \resumeItem{Built gradient boosting models for tabular risk prediction}
      % \resumeItem{Trained computer vision detectors for object detection}
      % \resumeItem{Processed geospatial raster tiles for terrain mapping}
    \resumeItemListEnd
  \resumeSubheading
    {ChargeScape}{May 2026 -- Present}
    {Machine Learning Intern}{Austin, TX}
    \resumeItemListStart
      \resumeItem{Designed load-management forecasting for EV charging fleets}
      % \resumeItem{Implemented computer vision segmentation for camera feeds}
      % \resumeItem{Optimized SQL pipelines for geospatial telemetry}
    \resumeItemListEnd
\resumeSubHeadingListEnd
\end{document}
"""


# --- parsing (includes commented pool bullets) --------------------------------


def test_parse_items_includes_commented_pool_bullets():
    items = resumeselect.parse_items(_TEX)
    assert len(items) == 6  # unlike tailor._extract_items, commented ones are kept
    assert [it["commented"] for it in items] == [False, True, True, False, True, True]
    assert items[0]["text"] == "Built gradient boosting models for tabular risk prediction"
    assert items[1]["text"] == "Trained computer vision detectors for object detection"


def test_parse_items_span_brackets_whole_item():
    items = resumeselect.parse_items(_TEX)
    active = items[0]
    assert _TEX[active["start"]:active["end"]] == "\\resumeItem{" + active["text"] + "}"
    commented = items[1]
    assert _TEX[commented["start"]:commented["end"]] == "% \\resumeItem{" + commented["text"] + "}"


# --- grouping into per-experience pools ---------------------------------------


def test_group_by_experience_groups_under_subheadings():
    groups = resumeselect.group_by_experience(_TEX)
    assert [company for company, _ in groups] == ["INSPIRE Lab", "ChargeScape"]
    assert [len(idxs) for _, idxs in groups] == [3, 3]


def test_group_keeps_pre_subheading_items_under_synthetic_key():
    tex = (
        "\\resumeItem{A standalone project bullet}\n"
        "\\resumeSubheading{Acme}{2025}{Engineer}{City}\n"
        "\\resumeItem{A real job bullet}\n"
    )
    groups = resumeselect.group_by_experience(tex)
    assert groups[0][0] == "_"        # pre-subheading item kept, not dropped
    assert len(groups[0][1]) == 1
    assert groups[1][0] == "Acme"


# --- scoring (word-boundary, like src/filter.py) ------------------------------


def test_score_item_counts_jd_keyword_matches():
    kw = {"computer vision", "detection", "gradient"}
    assert resumeselect.score_item(
        "Trained computer vision detectors for object detection", kw) == 2
    assert resumeselect.score_item("Built gradient boosting models", kw) == 1


def test_score_item_word_boundary_no_substring():
    # 'intern' must not match 'international' — word-boundary, not substring.
    assert resumeselect.score_item("worked on international logistics", {"intern"}) == 0


# --- selection under budget ---------------------------------------------------


def test_select_activates_jd_relevant_commented_bullet():
    jd = "We need computer vision and object detection for perception."
    new_tex, changes = resumeselect.select(_TEX, jd)

    # MIN_PER_ROLE=2 -> each experience shows 2 bullets (its active ML one + the top-scoring
    # commented pool bullet, the computer-vision one).
    active = [it for it in resumeselect.parse_items(new_tex) if not it["commented"]]
    assert len(active) == 4
    # both computer-vision pool bullets got activated (they out-score geospatial).
    cv_active = [it for it in active if "computer vision" in it["text"]]
    assert len(cv_active) == 2
    activated = [c for c in changes if c["action"] == "activated"]
    assert all("computer vision" in c["text"] for c in activated)
    assert {c["company"] for c in activated} == {"INSPIRE Lab", "ChargeScape"}


def test_select_respects_explicit_budget():
    # budget 4 == the two per-experience minimums (MIN_PER_ROLE=2 each) -> 4 active.
    new_tex, _changes = resumeselect.select(_TEX, "computer vision geospatial", budget=4)
    active = sum(not it["commented"] for it in resumeselect.parse_items(new_tex))
    assert active == 4


# --- guarantees ---------------------------------------------------------------


def test_select_noop_when_no_commented_pool():
    tex = (
        "\\resumeSubheading{Acme}{2025}{Engineer}{City}\n"
        "\\resumeItem{Built gradient boosting models}\n"
        "\\resumeItem{Shipped computer vision detectors}\n"
    )
    new_tex, changes = resumeselect.select(tex, "computer vision")
    assert new_tex == tex   # budget >= active count and nothing to uncomment
    assert changes == []


def test_select_keeps_min_bullets_when_all_score_zero():
    # unrelated JD -> every bullet scores 0; each experience still shows MIN_PER_ROLE (2),
    # taking its first two in document order (canonical + next).
    new_tex, changes = resumeselect.select(_TEX, "unrelated marketing finance role")
    groups = resumeselect.group_by_experience(new_tex)
    items = resumeselect.parse_items(new_tex)
    for _company, idxs in groups:
        assert sum(not items[i]["commented"] for i in idxs) == 2
    # nothing deactivated (the two originally-active stay; one more per role activates).
    assert all(c["action"] == "activated" for c in changes)


def test_max_per_role_cap():
    # one experience with 6 pool bullets that all score on the JD -> capped at 4.
    tex = ("\\resumeSubheading{Acme}{2025}{ML Intern}{City}\n\\resumeItemListStart\n"
           + "".join(f"  \\resumeItem{{model number {n} for prediction}}\n" for n in range(6))
           + "\\resumeItemListEnd\n")
    new_tex, _ = resumeselect.select(tex, "model prediction", budget=6)
    active = sum(not it["commented"] for it in resumeselect.parse_items(new_tex))
    assert active == resumeselect.MAX_PER_ROLE  # 4, not 6


def test_select_toggle_only_touches_comment_prefix():
    new_tex, _changes = resumeselect.select(_TEX, "computer vision object detection")
    assert new_tex != _TEX
    # bodies are byte-identical; only the leading '% ' prefixes moved.
    orig = resumeselect.parse_items(_TEX)
    new = resumeselect.parse_items(new_tex)
    assert [it["text"] for it in orig] == [it["text"] for it in new]


def test_select_empty_document_is_noop():
    new_tex, changes = resumeselect.select("no resume items here", "computer vision")
    assert new_tex == "no resume items here"
    assert changes == []


# --- regression: real-resume structural bugs the live run exposed ---------------

_SCAFFOLD_TEX = (
    "\\resumeSubheading\n  {Acme}{2024}\n  {ML Intern}{NYC}\n"
    "  \\resumeItemListStart\n"
    "    \\resumeItem{Real active bullet on gradient boosting forecasting}\n"
    "    % \\resumeItem{Real commented pool bullet on computer vision detection}\n"
    "  \\resumeItemListEnd\n"
    "% -----------Multiple Positions Heading (template example)-----------\n"
    "%    \\resumeSubSubheading\n"
    "%     {Software Engineer I}{Oct 2014 - Sep 2016}\n"
    "%     \\resumeItemListStart\n"
    "%        \\resumeItem{Apache Beam scaffolding that is NOT a real bullet}\n"
    "%     \\resumeItemListEnd\n"
)


def test_commented_template_block_is_never_pool():
    # a JD that matches the scaffolding text must NEVER activate it, and the line stays put.
    new_tex, changes = resumeselect.select(_SCAFFOLD_TEX, "apache beam streaming pipelines", budget=2)
    assert all("Apache Beam scaffolding" not in c["text"] for c in changes)
    # whitespace-robust: the scaffolding item is still present AND still commented.
    after = {it["text"]: it["commented"] for it in resumeselect.parse_items(new_tex)}
    assert after["Apache Beam scaffolding that is NOT a real bullet"] is True
    # the genuine commented pool bullet (inside the ACTIVE list) is still selectable.
    cv_tex, cv_changes = resumeselect.select(_SCAFFOLD_TEX, "computer vision detection", budget=2)
    assert any("computer vision" in c["text"] and c["action"] == "activated" for c in cv_changes)


_PROJECT_TEX = (
    "\\resumeSubheading\n  {Acme}{2024}\n  {ML Intern}{NYC}\n"
    "  \\resumeItemListStart\n    \\resumeItem{acme work on demand forecasting}\n  \\resumeItemListEnd\n"
    "\\resumeProjectHeading\n  {\\textbf{SciRAG}}{2026}\n"
    "  \\resumeItemListStart\n    \\resumeItem{scirag retrieval augmented generation over papers}\n  \\resumeItemListEnd\n"
)


def test_project_headings_form_their_own_group():
    groups = dict(resumeselect.group_by_experience(_PROJECT_TEX))
    assert "Acme" in groups
    assert "SciRAG" in groups           # project is its own group...
    # ...and the project bullet is NOT lumped under the preceding experience.
    acme_idx = groups["Acme"][0]
    scirag_idx = groups["SciRAG"][0]
    assert acme_idx != scirag_idx
