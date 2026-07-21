"""Unit tests for the auto-apply field mapper. Profiles are built inline so the
tests never depend on the real (gitignored) profile.yml. Offline; no network.
"""
from src import apply_fill as af

PROFILE = {
    "name": "Arnav Shukla",
    "email": "a@utexas.edu",
    "phone": "469-000-0000",
    "linkedin": "https://linkedin.com/in/arnav",
    "grad_date": "May 2028",
    "facts": {"resume_path": "/p/resume.pdf", "links": {"github": "https://github.com/x"}},
    "education": {"school": "UT Austin", "degree": "BS Statistics", "location": "Austin, TX"},
    "answer_bank": {
        "eligibility": {"work_authorized_us": True, "requires_sponsorship": False,
                        "age_18_or_older": True, "gpa": "decline"},
        "logistics": {"willing_to_relocate": True,
                      "salary_expectation": "Match the posted range in the JD",
                      "how_heard_about_us": "Job board (or LinkedIn)"},
        "background": {"felony_conviction": False},
        "eeo": {"gender": "prefer_not_to_say"},
    },
}


def _plan(label, ftype="text", required=False):
    return af.plan_field({"label": label, "type": ftype, "required": required}, PROFILE)


# --- Screening: the safety core ---------------------------------------------

def test_work_auth_maps_to_yes():
    p = _plan("Are you legally authorized to work in the US?")
    assert p["action"] == "fill" and p["value"] == "Yes"
    assert p["source"] == "eligibility.work_authorized_us"


def test_sponsorship_maps_to_no():
    p = _plan("Will you now or in the future require visa sponsorship?")
    assert p["action"] == "fill" and p["value"] == "No"


def test_null_answer_is_hard_stop_never_guessed():
    prof = {"answer_bank": {"eligibility": {"requires_sponsorship": None}}}
    p = af.plan_field({"label": "Do you require sponsorship?", "type": "radio"}, prof)
    assert p["action"] == "needs_human"
    assert "do not guess" in p["reason"]


def test_gpa_decline_skips_when_optional_but_stops_when_required():
    assert _plan("GPA", required=False)["action"] == "skip"
    assert _plan("GPA", required=True)["action"] == "needs_human"


def test_salary_is_a_directive_fill():
    p = _plan("Expected salary")
    assert p["action"] == "fill" and p.get("directive") is True
    assert p["source"] == "logistics.salary_expectation"


def test_unmapped_field_is_needs_human():
    p = _plan("What is your favorite programming language and why?")
    assert p["action"] == "needs_human"


def test_free_text_textarea_is_deferred():
    p = _plan("Cover letter", ftype="textarea")
    assert p["action"] == "needs_human" and "essay" in p["reason"]


# --- Identity / contact facts -----------------------------------------------

def test_identity_fields_fill_from_profile():
    assert _plan("Email")["value"] == "a@utexas.edu"
    assert _plan("First name")["value"] == "Arnav"
    assert _plan("Last name")["value"] == "Shukla"
    assert _plan("LinkedIn URL")["value"] == "https://linkedin.com/in/arnav"
    assert _plan("Resume / CV")["source"] == "resume_path"
    assert _plan("University")["value"] == "UT Austin"


def test_matched_identity_field_with_no_value_stops():
    prof = {"name": "", "facts": {}, "education": {}}
    p = af.plan_field({"label": "Email address"}, prof)
    assert p["action"] == "needs_human"


# --- Aggregation ------------------------------------------------------------

def test_plan_fields_buckets_by_action():
    fields = [
        {"label": "Email"},
        {"label": "GPA"},                                  # decline -> skip
        {"label": "Why this company?", "type": "textarea"},  # essay -> needs_human
        {"label": "Blorp identifier"},                     # unmapped -> needs_human
    ]
    plan = af.plan_fields(fields, PROFILE)
    assert [f["label"] for f in plan["fills"]] == ["Email"]
    assert [f["label"] for f in plan["skips"]] == ["GPA"]
    assert len(plan["needs_human"]) == 2
