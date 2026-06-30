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


# --- Live config regression: roles the widened keywords must recover ---------
# These titles (Tesla's naming + RL/CV/etc.) were silently dropped before the
# keyword widening. Loads the real config so a keyword removal trips the test.
import os

import pytest
import yaml

_CFG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "filters.yml")
LIVE_FILTERS = yaml.safe_load(open(_CFG))


@pytest.mark.parametrize("title,track", [
    ("Reinforcement Learning Engineer Intern", "ml"),
    ("Computer Vision Engineer Intern", "ml"),
    ("Perception Intern", "ml"),
    ("Digital Ship NLP Intern", "ml"),
    ("Fullstack C++ Engineer Intern", "swe"),
    ("Data Engineer Intern - Multiple Teams", "swe"),
    ("Software Integration Engineer Intern", "swe"),
    ("Mobile Applications Engineering Intern", "swe"),
])
def test_widened_keywords_recover_dropped_roles(title, track):
    assert filt.matches(L(title), LIVE_FILTERS) == track


@pytest.mark.parametrize("title", [
    "ML PhD Intern - LLMs & Generative AI",
    "Research Scientist Intern (PhD)",
    "Software Engineering Masters Intern",
    "Data Science Intern, Master's",
])
def test_grad_required_roles_excluded(title):
    assert filt.matches(L(title), LIVE_FILTERS) is None


@pytest.mark.parametrize("title,track", [
    # "BS/MS" accepts undergrads — must NOT be excluded by the degree gate.
    ("Software Engineer Intern - Security-Data - BS/MS", "swe"),
    ("Machine Learning Engineer Intern - Ads, BS/MS", "ml"),
])
def test_bs_ms_roles_survive_degree_gate(title, track):
    assert filt.matches(L(title), LIVE_FILTERS) == track


@pytest.mark.parametrize("title,expected", [
    ("Data Scientist Intern", "data"),
    ("Applied Data Science Intern", "data"),
    ("Machine Learning Data Scientist Intern", "ml"),   # ml matched first
    ("Data Analyst Intern", None),                       # analyst intentionally excluded
    ("Data Analytics Intern", None),
])
def test_data_science_track(title, expected):
    assert filt.matches(L(title), LIVE_FILTERS) == expected
