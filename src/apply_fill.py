"""Field mapper for the auto-apply driver — the deterministic, testable core.

Given a form's enumerated fields and the profile, produce a fill plan:
  - fills:       fields we can answer (identity facts + answer_bank lookups)
  - drafts:      free-text questions for the essay generator (essays.draft_answer)
  - skips:       optional fields intentionally left blank (e.g. gpa 'decline')
  - needs_human: fields we must NOT guess (unmapped, no answer on file)

Safety model: screening/eligibility answers are LOOKED UP, never generated. A
label that doesn't confidently match a known question, or whose answer_bank value
is null, becomes needs_human. That is what keeps a wrong work-auth / sponsorship
answer — a permanent auto-reject — from ever being autofilled. The Claude-in-Chrome
agent runs the browser, but it defers every eligibility answer to this table.
"""
from __future__ import annotations

import re

# Ordered (label patterns -> answer_bank path). First match wins, so put the more
# specific phrases before generic ones. Matching is case-insensitive substring.
_SCREENING: list[tuple[tuple[str, ...], tuple[str, str]]] = [
    (("legally authorized", "authorized to work", "work authorization", "authorized to work in the u"),
     ("eligibility", "work_authorized_us")),
    (("sponsorship", "sponsor you", "require visa", "visa sponsor"),
     ("eligibility", "requires_sponsorship")),
    (("18 years", "at least 18", "over 18", "older than 18", "least 18 years"),
     ("eligibility", "age_18_or_older")),
    (("currently enrolled", "are you a student", "current student", "enrolled student"),
     ("eligibility", "currently_enrolled_student")),
    (("gpa", "grade point"),
     ("eligibility", "gpa")),
    (("earliest start", "start date", "when can you start", "availability", "available to start", "date available"),
     ("logistics", "earliest_start_date")),
    (("relocate", "relocation"),
     ("logistics", "willing_to_relocate")),
    (("salary", "compensation expectation", "expected pay", "desired compensation", "pay expectation", "expected salary"),
     ("logistics", "salary_expectation")),
    (("how did you hear", "how were you referred", "referral source", "hear about", "how you heard"),
     ("logistics", "how_heard_about_us")),
    (("previously employed", "worked here before", "former employee", "previously worked"),
     ("logistics", "previously_employed_here")),
    (("felony", "convicted", "criminal record", "criminal history"),
     ("background", "felony_conviction")),
    (("non-compete", "noncompete", "non compete"),
     ("background", "non_compete_agreement")),
    (("gender", "what is your sex"), ("eeo", "gender")),
    (("race", "ethnicity"), ("eeo", "race_ethnicity")),
    (("veteran",), ("eeo", "veteran_status")),
    (("disability",), ("eeo", "disability_status")),
]

# These answer_bank values are instructions the LLM driver applies per-form (it
# reads the JD / dropdown options and picks), not literal strings to type. They
# are low-risk fields — never auto-reject filters.
_DIRECTIVE_KEYS = {("logistics", "salary_expectation"), ("logistics", "how_heard_about_us")}

# Free-text prompts we defer to essay drafting (step 3), never autofill blindly.
_ESSAY_HINTS = ("why ", "why do", "why are", "describe", "tell us", "cover letter",
                "what interests", "in your own words", "elaborate", "explain why")

# Questions that presuppose enrollment in a graduate program. For a Bachelor's
# student these are answered "No" (yes/no) or left to the human (program details) —
# never a blind "Yes". Degree-aware so it self-adjusts if degree_level changes.
_GRAD_QUALIFIERS = ("master", "phd", "ph.d", "mba", "doctoral", "doctorate",
                    "graduate program", "grad program")


def _is_grad_student(profile: dict) -> bool:
    dl = str(((profile.get("answer_bank") or {}).get("eligibility") or {})
             .get("degree_level", "")).lower()
    return any(q in dl for q in ("master", "phd", "ph.d", "mba", "doctor"))


def _words(s: str) -> set[str]:
    return set(re.split(r"\W+", s.lower()))


def _identity_value(label: str, profile: dict) -> tuple[str | None, str | None]:
    """Resolve an identity/contact field from the profile. Returns (value, source);
    (None, None) if the label doesn't look like an identity field; ('', source) if
    it matched a known field but the profile has no value for it.
    """
    l = label.lower()
    w = _words(label)
    facts = profile.get("facts", {}) or {}
    edu = profile.get("education", {}) or {}
    links = facts.get("links", {}) or {}
    name = profile.get("name", "") or ""

    if "resume" in w or "cv" in w or "upload" in l:
        return facts.get("resume_path", ""), "resume_path"
    if "linkedin" in l:
        return profile.get("linkedin", ""), "linkedin"
    if "github" in l:
        return links.get("github", ""), "github"
    if "email" in l:
        return profile.get("email", ""), "email"
    if "phone" in l or "mobile" in w:
        return profile.get("phone", ""), "phone"
    if "first name" in l or "given name" in l:
        return name.split(" ")[0], "name"
    if "last name" in l or "surname" in l or "family name" in l:
        return " ".join(name.split(" ")[1:]), "name"
    if "full name" in l or "legal name" in l or l.strip() in ("name", "your name"):
        return name, "name"
    if {"school", "university", "institution", "college"} & w:
        return edu.get("school", ""), "school"
    if "degree" in w or "major" in w:
        return edu.get("degree", ""), "degree"
    if "graduation" in l or "grad date" in l or "expected graduation" in l:
        return (profile.get("grad_date") or edu.get("graduation", "")), "graduation"
    if "location" in l or "city" in w or l.strip() == "address":
        return edu.get("location", ""), "location"
    return None, None


def _render(value, required: bool, path: tuple[str, str]) -> tuple[str, str]:
    """Turn an answer_bank value into a (action, payload) decision."""
    if value is None:
        return "needs_human", "no answer on file — do not guess"
    if value == "decline":
        if required:
            return "needs_human", "field required but answer is 'decline'"
        return "skip", "intentionally left blank (decline)"
    if path in _DIRECTIVE_KEYS:
        return "directive", str(value)
    if isinstance(value, bool):
        return "fill", "Yes" if value else "No"
    return "fill", str(value)


_DECLINE_HINTS = ("decline", "prefer not", "wish to", "not to say", "not wish")


def _resolve_option(value, options) -> str | None:
    """Map an intended value to a dropdown/radio's actual option text.

    Order: decline-intent -> exact (case-insensitive) -> substring either way (only
    for values longer than 3 chars, so "No" can't match "...not..."). Returns None
    when nothing matches confidently, so the caller can fall back to needs_human
    rather than fill a value the form doesn't offer. No options -> passthrough (a
    free-text field imposes no constraint).
    """
    if not options:
        return str(value)
    opts = [str(o) for o in options]
    v = str(value).strip().lower()

    if v in ("prefer_not_to_say", "decline", "prefer not to say"):
        for o in opts:
            if any(h in o.lower() for h in _DECLINE_HINTS):
                return o
        return None
    for o in opts:                      # exact
        if o.strip().lower() == v:
            return o
    if len(v) > 3:                      # whole-word match (so "Male" != "Female")
        for o in opts:
            ol = o.strip().lower()
            if (re.search(r"\b" + re.escape(v) + r"\b", ol)
                    or re.search(r"\b" + re.escape(ol) + r"\b", v)):
                return o
    return None


def _fill(label: str, value, source: str, field: dict) -> dict:
    """Build a fill decision, resolving to a real option when the field is a select."""
    resolved = _resolve_option(value, field.get("options"))
    if resolved is None:
        return {"label": label, "action": "needs_human",
                "reason": f"'{value}' has no matching option for {source}"}
    return {"label": label, "action": "fill", "value": resolved, "source": source}


def plan_field(field: dict, profile: dict) -> dict:
    """Decide what to do with one enumerated form field.

    field: {"label": str, "type": str, "required": bool, "options"?: [...]}
    Returns one of:
      {"label", "action": "fill", "value", "source", "directive"?: True}
      {"label", "action": "skip", "reason"}
      {"label", "action": "needs_human", "reason"}
    """
    label = field.get("label", "")
    ftype = (field.get("type") or "text").lower()
    required = bool(field.get("required"))
    l = label.lower()

    # Free-text essays: flag for the essay generator (essays.draft_answer). The
    # classifier only marks the field; the driver does the actual drafting, so this
    # module stays pure and offline-testable.
    if ftype == "textarea" or any(h in l for h in _ESSAY_HINTS):
        return {"label": label, "action": "draft", "question": label}

    # Any file input on an application form is the resume — forms rarely have more
    # than one. Handles Greenhouse's "Attach" button label, which says nothing about
    # "resume". (If a form ever has a second file field, it would also get the resume;
    # revisit only if that shows up.)
    if ftype == "file":
        rp = (profile.get("facts") or {}).get("resume_path") or ""
        if rp:
            return {"label": label, "action": "fill", "value": rp, "source": "resume_path"}
        return {"label": label, "action": "needs_human", "reason": "file field but no resume_path on file"}

    # Screening / eligibility / EEO: answer_bank lookup (the safety core).
    ab = profile.get("answer_bank", {}) or {}
    for patterns, (section, key) in _SCREENING:
        if any(p in l for p in patterns):
            value = (ab.get(section, {}) or {}).get(key)
            # "Enrolled in a Masters/PhD program?" is No for a Bachelor's student,
            # even though the generic "currently enrolled" answer is Yes.
            if (key == "currently_enrolled_student" and not _is_grad_student(profile)
                    and any(q in l for q in _GRAD_QUALIFIERS)):
                value = False
            action, payload = _render(value, required, (section, key))
            src = f"{section}.{key}"
            if action == "fill":
                return _fill(label, payload, src, field)
            if action == "directive":
                return {"label": label, "action": "fill", "value": payload, "source": src, "directive": True}
            if action == "skip":
                return {"label": label, "action": "skip", "reason": payload}
            return {"label": label, "action": "needs_human", "reason": payload}

    # Identity / contact facts.
    value, src = _identity_value(label, profile)
    if value:
        # A "which Masters/PhD program + grad year" field presupposes grad enrollment;
        # don't fill it with the undergrad grad date.
        if (src == "graduation" and not _is_grad_student(profile)
                and any(q in l for q in _GRAD_QUALIFIERS)):
            return {"label": label, "action": "needs_human", "reason": "grad-program-specific field"}
        return _fill(label, value, src, field)
    if value == "":  # matched a known field but the profile has no data for it
        return {"label": label, "action": "needs_human", "reason": f"matched {src} but no value in profile"}

    # No confident match anywhere.
    return {"label": label, "action": "needs_human", "reason": "unmapped field — do not guess"}


def plan_fields(fields: list[dict], profile: dict) -> dict:
    """Aggregate per-field decisions into {fills, drafts, skips, needs_human}."""
    fills, drafts, skips, needs = [], [], [], []
    buckets = {"fill": fills, "draft": drafts, "skip": skips, "needs_human": needs}
    for f in fields:
        p = plan_field(f, profile)
        buckets[p["action"]].append(p)
    return {"fills": fills, "drafts": drafts, "skips": skips, "needs_human": needs}
