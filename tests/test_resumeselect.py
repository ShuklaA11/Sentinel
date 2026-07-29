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

    # default budget = 2 currently-active bullets -> each experience keeps exactly 1.
    active = [it for it in resumeselect.parse_items(new_tex) if not it["commented"]]
    assert len(active) == 2
    # the computer-vision bullets out-score the (now commented) ML bullets and are active.
    assert all("computer vision" in it["text"] for it in active)

    actions = sorted(c["action"] for c in changes)
    assert actions == ["activated", "activated", "deactivated", "deactivated"]
    assert {c["company"] for c in changes} == {"INSPIRE Lab", "ChargeScape"}
    assert all("computer vision" in c["text"]
               for c in changes if c["action"] == "activated")


def test_select_respects_explicit_budget():
    # budget 4 = the two per-experience minimums + 2 extra slots.
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


def test_select_keeps_first_bullet_when_all_score_zero():
    # unrelated JD -> every bullet scores 0; each experience keeps its canonical (first)
    # active bullet, nothing flips.
    new_tex, changes = resumeselect.select(_TEX, "unrelated marketing finance role")
    active = [it for it in resumeselect.parse_items(new_tex) if not it["commented"]]
    assert len(active) == 2
    assert new_tex == _TEX
    assert changes == []


def test_select_toggle_only_touches_comment_prefix():
    new_tex, _changes = resumeselect.select(_TEX, "computer vision object detection")
    assert new_tex != _TEX
    # bodies are byte-identical; only the leading '% ' prefixes moved.
    orig = resumeselect.parse_items(_TEX)
    new = resumeselect.parse_items(new_tex)
    assert [it["text"] for it in orig] == [it["text"] for it in new]
    # active-bullet count is preserved (default budget == input active count).
    assert sum(not it["commented"] for it in new) == sum(not it["commented"] for it in orig)


def test_select_empty_document_is_noop():
    new_tex, changes = resumeselect.select("no resume items here", "computer vision")
    assert new_tex == "no resume items here"
    assert changes == []
