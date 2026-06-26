"""Unit tests for the keyword/track/season filter — the highest-logic module.

Self-contained FILTERS fixture (does not read config/) so we test the matching
logic, not the live config. Offline; no network.
"""
from src import filter as filt

FILTERS = {
    "intern_terms": ["intern", "internship"],
    "exclude_terms": ["senior", "staff", "principal", "phd"],
    "tracks": {
        "ml": ["machine learning", "ml", "ai"],
        "swe": ["software engineer", "swe"],
        "product": ["product manager"],
    },
    "allowed_years": ["2026", "2027"],
    "seasons": ["fall 2026", "summer 2027"],
    "locations": [],
}


def L(title, location=""):
    return {"title": title, "location": location, "id": "x"}


def test_ml_intern_gets_ml_track():
    assert filt.matches(L("Machine Learning Intern"), FILTERS) == "ml"


def test_swe_intern_gets_swe_track():
    assert filt.matches(L("Software Engineer Intern"), FILTERS) == "swe"


def test_non_intern_role_rejected():
    assert filt.matches(L("Software Engineer"), FILTERS) is None


def test_senior_role_excluded_even_if_intern():
    assert filt.matches(L("Senior Software Engineer Intern"), FILTERS) is None


def test_word_boundary_international_is_not_intern():
    # 'International' must NOT satisfy the 'intern' term (the bug that was fixed).
    assert filt.matches(L("International Business Analyst"), FILTERS) is None


def test_word_boundary_ai_not_matched_inside_maintenance():
    # 'ai' inside 'Maintenance' must not assign the ml track.
    assert filt.matches(L("Maintenance Intern"), FILTERS) is None


def test_disallowed_year_in_title_rejected():
    assert filt.matches(L("Software Engineer Intern 2024"), FILTERS) is None


def test_allowed_year_passes():
    assert filt.matches(L("Software Engineer Intern Fall 2026"), FILTERS) == "swe"


def test_title_without_year_passes():
    assert filt.matches(L("Software Engineer Intern"), FILTERS) == "swe"


def test_location_gate_filters_and_accepts():
    f = {**FILTERS, "locations": ["austin", "remote"]}
    assert filt.matches(L("Software Engineer Intern", "New York, NY"), f) is None
    assert filt.matches(L("Software Engineer Intern", "Austin, TX"), f) == "swe"


def test_apply_filters_adds_track_and_drops_non_matches():
    kept = filt.apply_filters([L("ML Intern"), L("Accountant")], FILTERS)
    assert len(kept) == 1
    assert kept[0]["track"] == "ml"
