"""Unit tests for the cover-letter generator's pure / mockable parts.

Offline only: llm.complete is monkeypatched so nothing hits the network, and
profile / fact-source I/O is monkeypatched so nothing depends on the live
profile/ directory. The LLM prose path itself is integration-only; here we cover
value extraction, the fact gate, the word cap, keyword injection, the graceful
no-API-key skip, and the fail-closed path when the LLM is unavailable.
"""
from __future__ import annotations

import os

from pypdf import PdfReader

from src import cover


# ---------------------------------------------------------------------------
# extract_values — parses a company-values list, [] when absent
# ---------------------------------------------------------------------------


def test_extract_values_bullet_list():
    jd = (
        "About the team. We build payments infra.\n\n"
        "Our values:\n"
        "- Ownership\n"
        "- Customer obsession\n"
        "- Move fast\n\n"
        "Responsibilities: ship things.\n"
    )
    values = cover.extract_values(jd)
    assert "Ownership" in values
    assert "Customer obsession" in values
    assert "Move fast" in values


def test_extract_values_inline_comma_list():
    jd = "What we value: Craft, Candor, and Speed.\nResponsibilities: ...\n"
    values = cover.extract_values(jd)
    assert "Craft" in values
    assert "Candor" in values
    assert "Speed" in values


def test_extract_values_absent_returns_empty():
    jd = (
        "We are hiring a Machine Learning Intern to work on retrieval systems.\n"
        "You will build pipelines and evaluate models. Python and PyTorch required.\n"
    )
    assert cover.extract_values(jd) == []


def test_extract_values_empty_string():
    assert cover.extract_values("") == []


# ---------------------------------------------------------------------------
# fact gate — drops sentences carrying an ungrounded (fabricated) number
# ---------------------------------------------------------------------------


def test_fact_gate_drops_fabricated_number():
    # 'boosted signups 5000%' is a fabricated metric absent from the sources;
    # the grounded sentence stays, the fabricated one is dropped.
    draft = (
        "I love what you're building. "
        "I boosted signups 5000% in a single week. "
        "I shipped a retrieval service teams actually use."
    )
    sources = ["candidate shipped a retrieval service; no such metric anywhere"]
    cleaned = cover._drop_unverified_sentences(draft, sources)
    assert "5000%" not in cleaned
    assert "retrieval service" in cleaned
    # the gate must certify the cleaned text as grounded.
    from src import verify_facts
    ok, violations = verify_facts.verify(cleaned, sources)
    assert ok is True
    assert violations == []


def test_fact_gate_keeps_grounded_number():
    draft = "I cut latency 3x on the checkout path."
    sources = ["candidate reduced checkout latency 3x"]
    assert cover._drop_unverified_sentences(draft, sources) == draft


# ---------------------------------------------------------------------------
# word cap — never ships more than MAX_WORDS
# ---------------------------------------------------------------------------


def test_cap_words_trims_to_limit():
    long_para = " ".join(f"word{i}." for i in range(500))
    capped = cover._cap_words(long_para, limit=400)
    assert len(capped.split()) <= 400


def test_cap_words_noop_when_within_limit():
    text = "Short letter. Two sentences only."
    assert cover._cap_words(text, limit=400) == text


# ---------------------------------------------------------------------------
# em-dash sanitizer — the guide bans em dashes in the letter
# ---------------------------------------------------------------------------


def test_sanitize_strips_em_dashes():
    out = cover._sanitize("I build fast—and I ship—every week.")
    assert "—" not in out


# ---------------------------------------------------------------------------
# tailor_cover — graceful skip with no API key (never raises)
# ---------------------------------------------------------------------------


def test_tailor_cover_no_api_key_returns_none(monkeypatch, caplog):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # both providers gone -> no fallback
    import logging
    with caplog.at_level(logging.WARNING, logger="cover"):
        result = cover.tailor_cover("Acme", "ML Intern", "Python and PyTorch role.")
    assert result is None
    assert any("ANTHROPIC_API_KEY" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# tailor_cover — assembled letter is <= 400 words and carries a genuine keyword
# ---------------------------------------------------------------------------


def _fake_complete(captured: dict, reply: str):
    """A stand-in for llm.complete: captures the prompt, returns a canned body."""

    def complete(prompt, *, max_tokens=1200):
        captured["prompt"] = prompt
        return reply

    return complete


_CANNED_LETTER = (
    "Your work on retrieval-first search is exactly the problem I want to work on.\n\n"
    "I've spent the last year building with PyTorch, so the stack here feels like home.\n\n"
    "On my last project I shipped a retrieval service my lab still uses every day.\n\n"
    "The ML Intern role lines up with the evaluation work you describe.\n\n"
    "I'm based in Boston and free to start in June. Thanks for reading."
)


def _profile() -> dict:
    return {
        "name": "Test Candidate",
        "facts": {"location": "Boston, MA", "availability": "June 2026"},
        "skills": {"languages": ["Python"], "frameworks": ["PyTorch"], "tools": ["Docker"]},
        "preferences": {"tracks": ["ml"]},
    }


def test_tailor_cover_assembles_bounded_letter_with_keyword(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured: dict = {}
    monkeypatch.setattr(cover.llm, "complete", _fake_complete(captured, _CANNED_LETTER))
    monkeypatch.setattr(cover, "_load_yaml", lambda p: _profile() if p.endswith("profile.yml") else {})
    monkeypatch.setattr(cover, "_read_optional", lambda p: "")
    monkeypatch.setattr(cover.verify_facts, "load_sources", lambda: [])
    monkeypatch.setattr(cover, "OUT_DIR", str(tmp_path / "out"))

    jd = "We need a Machine Learning Intern strong in Python and PyTorch for retrieval work."
    letter = cover.tailor_cover("Acme", "ML Intern", jd, archetype="startup")

    assert letter is not None
    # bounded: never more than 400 words.
    assert len(letter.split()) <= 400
    # a genuinely-held JD keyword the candidate has surfaced in the letter.
    from src import keywords
    inject = keywords.truthful_injection_candidates(
        keywords.extract_jd_keywords(jd), keywords.profile_skill_terms(_profile())
    )
    assert "pytorch" in inject
    assert any(kw in letter.lower() for kw in inject)
    # the injected keywords were passed into the prompt.
    assert "pytorch" in captured["prompt"].lower()
    # the letter was written to the slugified path.
    out_file = tmp_path / "out" / "acme_ml_intern_cover.md"
    assert out_file.exists()
    assert out_file.read_text() == letter
    pdf_file = tmp_path / "out" / "acme_ml_intern_cover.pdf"
    assert pdf_file.exists()
    reader = PdfReader(str(pdf_file))
    assert len(reader.pages) == 1
    pdf_text = reader.pages[0].extract_text()
    assert "Test Candidate" in pdf_text
    assert "ML Intern" in pdf_text
    assert "PyTorch" in pdf_text


def test_render_cover_pdf_is_one_page_and_ats_readable(tmp_path):
    profile = {
        "name": "Jane Candidate",
        "email": "jane@example.com",
        "phone": "555-0100",
        "linkedin": "https://linkedin.com/in/jane",
        "education": {"location": "Austin, TX"},
    }
    letter = (
        "This company-specific opening explains why the mission matters.\n\n"
        "I built a Python and PyTorch vision pipeline and measured its performance.\n\n"
        "That experience maps directly to the role's model-evaluation work.\n\n"
        "Thank you for your consideration."
    )
    out = tmp_path / "cover.pdf"

    rendered = cover.render_cover_pdf(
        letter, "Acme AI", "Machine Learning Intern", profile, str(out),
    )

    assert rendered == str(out)
    assert out.read_bytes().startswith(b"%PDF")
    reader = PdfReader(str(out))
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    for expected in (
        "Jane Candidate", "jane@example.com", "555-0100",
        "linkedin.com/in/jane", "Acme AI", "Machine Learning Intern",
        "Dear Acme AI Hiring Team",
        "Python and PyTorch",
    ):
        assert expected in text


def test_render_cover_pdf_rejects_overflow_instead_of_shipping(tmp_path):
    profile = {"name": "Jane Candidate", "email": "jane@example.com"}
    letter = "\n\n".join(
        "This deliberately long paragraph contains enough words to force a second page. " * 18
        for _ in range(12)
    )
    out = tmp_path / "overflow.pdf"

    rendered = cover.render_cover_pdf(letter, "Acme", "ML Intern", profile, str(out))

    assert rendered is None
    assert not os.path.exists(out)


# ---------------------------------------------------------------------------
# immutable_facts — non-negotiable identity block, guards against credential
# fabrication / upgrade (the live run once claimed a Master's the candidate
# does not hold because the JD asked for 'MS or PhD')
# ---------------------------------------------------------------------------


def _full_profile() -> dict:
    """A profile that carries the education / experience / eligibility sections
    immutable_facts reads (the _profile() fixture above deliberately lacks them)."""
    return {
        "name": "Test Candidate",
        "facts": {"location": "Austin, TX", "availability": "May 2028"},
        "skills": {"languages": ["Python"], "frameworks": ["PyTorch"]},
        "education": {
            "degree": "Bachelor of Science in Statistics and Data Science",
            "school": "University of Texas at Austin",
            "graduation": "May 2028",
        },
        "experience": [
            {"role": "Machine Learning Intern", "company": "ChargeScape"},
        ],
        "answer_bank": {
            "eligibility": {"work_authorized_us": True, "requires_sponsorship": False},
        },
        "grad_date": "May 2028",
        "preferences": {"tracks": ["ml"]},
    }


def test_immutable_facts_renders_real_identity_and_guards():
    block = cover.immutable_facts(_full_profile())
    # the real degree string, verbatim.
    assert "Bachelor of Science in Statistics and Data Science" in block
    assert "University of Texas at Austin" in block
    assert "May 2028" in block
    # an explicit NEVER-change/upgrade instruction on the degree.
    assert "NEVER change/upgrade" in block
    # an explicit 'NOT a Master's or PhD' standing statement.
    assert "NOT a Master's or PhD" in block
    # real current role + work authorization.
    assert "Machine Learning Intern at ChargeScape" in block
    assert "authorized to work in the US, no sponsorship required" in block
    # stays honest without volunteering the gap: don't apologize for / flag a missing credential.
    assert "do NOT apologize for or draw attention to lacking it" in block


def test_immutable_facts_defensive_on_missing_sections():
    # an empty profile must never raise; it degrades to an empty block.
    assert cover.immutable_facts({}) == ""
    # a partial profile (degree only) renders what's present and omits the rest.
    partial = {"education": {"degree": "Bachelor of Science in Statistics and Data Science"}}
    block = cover.immutable_facts(partial)
    assert "Bachelor of Science in Statistics and Data Science" in block
    assert "NOT a Master's or PhD" in block  # still guards from the degree alone
    assert "Current role" not in block       # no experience section -> omitted
    assert "Work authorization" not in block  # no eligibility section -> omitted


def test_prompt_embeds_immutable_facts_and_anti_upgrade_rule(tmp_path, monkeypatch):
    """The prompt sent to llm.complete must embed the immutable-facts block (real BS
    degree) AND the anti-Master's / anti-'even if the job asks' rule, so the model
    can never upgrade the credential to satisfy an 'MS or PhD' requirement."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured: dict = {}
    monkeypatch.setattr(cover.llm, "complete", _fake_complete(captured, _CANNED_LETTER))
    # profile.yml is not committed, so force the load guard True and feed the full profile.
    monkeypatch.setattr(cover.os.path, "exists", lambda p: True)
    monkeypatch.setattr(cover, "_load_yaml", lambda p: _full_profile() if p.endswith("profile.yml") else {})
    monkeypatch.setattr(cover, "_read_optional", lambda p: "")
    monkeypatch.setattr(cover.verify_facts, "load_sources", lambda: [])
    monkeypatch.setattr(cover, "OUT_DIR", str(tmp_path / "out"))

    jd = "Machine Learning Intern. MS or PhD in CS preferred. Python and PyTorch for retrieval."
    cover.tailor_cover("Acme", "ML Intern", jd, archetype="startup")

    prompt = captured["prompt"]
    # the real BS degree is embedded verbatim.
    assert "Bachelor of Science in Statistics and Data Science" in prompt
    # the anti-Master's guard is present.
    assert "NOT a Master's or PhD" in prompt
    assert "Master's" in prompt
    # the 'even if the job asks' rule is present.
    assert "EVEN IF THE JOB ASKS FOR ONE" in prompt


def test_tailor_cover_fact_gate_regenerates_then_drops(tmp_path, monkeypatch):
    """If the model keeps fabricating a number across both tries, the offending
    sentence is dropped rather than shipped."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    captured: dict = {}
    fabricated = (
        "Your mission resonates with me.\n\n"
        "I grew a side project to 9000% month-over-month growth.\n\n"
        "I'm in Boston and free in June. Thanks."
    )
    monkeypatch.setattr(cover.llm, "complete", _fake_complete(captured, fabricated))
    monkeypatch.setattr(cover, "_load_yaml", lambda p: _profile() if p.endswith("profile.yml") else {})
    monkeypatch.setattr(cover, "_read_optional", lambda p: "")
    monkeypatch.setattr(cover.verify_facts, "load_sources", lambda: ["no such metric in ground truth"])
    monkeypatch.setattr(cover, "OUT_DIR", str(tmp_path / "out"))

    letter = cover.tailor_cover("Acme", "ML Intern", "Python role.", archetype="startup")

    assert letter is not None
    assert "9000%" not in letter
    from src import verify_facts
    ok, _ = verify_facts.verify(letter, ["no such metric in ground truth"])
    assert ok is True


# ---------------------------------------------------------------------------
# tailor_cover — LLM unavailable => fail closed (None, no file, no ✓)
# ---------------------------------------------------------------------------


def test_tailor_cover_llm_unavailable_writes_nothing(tmp_path, monkeypatch, capsys):
    """When llm.complete returns None (no provider / unavailable), tailor_cover
    must return None, write NO output file, and never print a ✓ success line —
    the fix for the 0-byte 'success' silent-failure bug."""
    monkeypatch.setattr(cover.llm, "complete", lambda prompt, *, max_tokens=1200: None)
    monkeypatch.setattr(cover, "_load_yaml", lambda p: _profile() if p.endswith("profile.yml") else {})
    monkeypatch.setattr(cover, "_read_optional", lambda p: "")
    monkeypatch.setattr(cover.verify_facts, "load_sources", lambda: [])
    monkeypatch.setattr(cover, "OUT_DIR", str(tmp_path / "out"))

    result = cover.tailor_cover("Acme", "ML Intern", "Python and PyTorch role.", archetype="startup")

    assert result is None
    # no file written at the slugified path
    out_file = tmp_path / "out" / "acme_ml_intern_cover.md"
    assert not out_file.exists()
    assert not (tmp_path / "out" / "acme_ml_intern_cover.pdf").exists()
    # never a success marker on failure
    assert "✓" not in capsys.readouterr().out
