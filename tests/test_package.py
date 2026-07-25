"""Unit tests for the on-demand application-package generator.

Offline only: every network / LLM / file boundary is monkeypatched, so nothing
hits the wire, no real ANTHROPIC/OPENAI key is needed, and nothing depends on the
gitignored profile/ directory. We cover: package assembly (dir + files + master
doc), snapshot TODO rendering for missing profile fields, the graceful LLM-
unavailable degradation (MESSAGE + STAR render a visible placeholder, never crash
or fabricate), CSV --id resolution, explicit-flag override, the values section's
presence/omission, and the clean error when an --id is absent from the CSV.
"""
from __future__ import annotations

import os

import pytest

from src import package


FAKE_PROFILE = {
    "name": "Jane Doe",
    "facts": {
        "email": "jane@example.com",
        "phone": "555-0100",
        "location": "New York, NY",
        "work_authorization": "US citizen",
        "graduation_date": "May 2027",
        "links": {"github": "github.com/janedoe", "linkedin": "linkedin.com/in/janedoe"},
    },
    "skills": {"languages": ["Python"], "frameworks": ["PyTorch"]},
    "projects": ["Built a retrieval system with PyTorch and evaluated it"],
    "answer_bank": {
        "availability": "June 2026",
        "target_salary": "$8k/month",
    },
}

DEFAULT_JD = "We want Python and PyTorch. A machine learning role on retrieval."


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Redirect the output dir to a tmp path and stub every external boundary."""
    apps = tmp_path / "applications"
    monkeypatch.setattr(package, "APPS_DIR", str(apps))

    monkeypatch.setattr(package.jdfetch, "fetch_jd", lambda *a, **k: DEFAULT_JD)

    fake_pdf = tmp_path / "resume_src.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake resume")
    monkeypatch.setattr(package.tailor, "tailor", lambda *a, **k: str(fake_pdf))

    monkeypatch.setattr(package.cover, "tailor_cover",
                        lambda *a, **k: "Dear team, I admire your retrieval work.")
    monkeypatch.setattr(package, "_load_profile", lambda: FAKE_PROFILE)
    return tmp_path, str(apps), str(fake_pdf)


def _read_md(pkg_dir: str) -> str:
    with open(os.path.join(pkg_dir, "application.md")) as f:
        return f.read()


# ---------------------------------------------------------------------------
# happy path — dir, files, and master-doc section markers
# ---------------------------------------------------------------------------


def test_build_package_creates_dir_files_and_master_doc(patched, monkeypatch):
    monkeypatch.setattr(package.llm, "complete",
                        lambda *a, **k: "I built retrieval systems and love this mission.")
    pkg = package.build_package(company="Anthropic", title="ML Intern",
                                url="http://x/apply", source="greenhouse")

    assert os.path.isdir(pkg)
    assert os.path.exists(os.path.join(pkg, "application.md"))
    assert os.path.exists(os.path.join(pkg, "resume.pdf"))
    assert os.path.exists(os.path.join(pkg, "jd.txt"))
    assert os.path.exists(os.path.join(pkg, "cover_letter.md"))

    md = _read_md(pkg)
    assert "HEADER" in md
    assert "SNAPSHOT ANSWERS" in md
    assert "JD KEYWORD COVERAGE" in md
    assert "Anthropic" in md
    assert "ML Intern" in md


def test_jd_txt_holds_fetched_description(patched, monkeypatch):
    monkeypatch.setattr(package.llm, "complete", lambda *a, **k: "text")
    pkg = package.build_package(company="Co", title="ML Intern", url="u", source="greenhouse")
    with open(os.path.join(pkg, "jd.txt")) as f:
        assert f.read() == DEFAULT_JD


# ---------------------------------------------------------------------------
# snapshot answers — real values shown, missing ones render TODO: <field>
# ---------------------------------------------------------------------------


def test_snapshot_shows_present_values(patched, monkeypatch):
    monkeypatch.setattr(package.llm, "complete", lambda *a, **k: "text")
    pkg = package.build_package(company="Co", title="SWE Intern", url="u", source="lever")
    md = _read_md(pkg)
    assert "Jane Doe" in md
    assert "jane@example.com" in md
    assert "github.com/janedoe" in md
    assert "US citizen" in md


def test_snapshot_todo_for_missing_fields(patched, monkeypatch):
    monkeypatch.setattr(package, "_load_profile", lambda: {"name": "Jane Doe"})
    monkeypatch.setattr(package.llm, "complete", lambda *a, **k: "text")
    pkg = package.build_package(company="Co", title="SWE Intern", url="u", source="lever")
    md = _read_md(pkg)
    assert "Jane Doe" in md
    assert "TODO: Email" in md
    assert "TODO: Phone" in md
    assert "TODO: GitHub" in md


# ---------------------------------------------------------------------------
# graceful degradation — llm.complete None => visible placeholder, never crash
# ---------------------------------------------------------------------------


def test_llm_unavailable_degrades_message_and_star(patched, monkeypatch):
    monkeypatch.setattr(package.llm, "complete", lambda *a, **k: None)
    monkeypatch.setattr(package.cover, "tailor_cover", lambda *a, **k: None)

    pkg = package.build_package(company="Co", title="ML Intern", url="u", source="greenhouse")
    md = _read_md(pkg)

    assert "MESSAGE TO HIRING TEAM" in md
    assert "STAR BEHAVIORAL ANSWERS" in md
    # placeholder appears for the message AND at least one STAR competency
    assert md.count(package.LLM_UNAVAILABLE) >= 2
    # cover was None -> no file written, master doc still built
    assert not os.path.exists(os.path.join(pkg, "cover_letter.md"))


def test_missing_resume_path_skips_copy_without_crashing(patched, monkeypatch):
    monkeypatch.setattr(package.tailor, "tailor", lambda *a, **k: None)
    monkeypatch.setattr(package.llm, "complete", lambda *a, **k: "text")
    pkg = package.build_package(company="Co", title="ML Intern", url="u", source="greenhouse")
    assert os.path.exists(os.path.join(pkg, "application.md"))
    assert not os.path.exists(os.path.join(pkg, "resume.pdf"))


# ---------------------------------------------------------------------------
# listing resolution — CSV lookup by id, flag override, clean not-found error
# ---------------------------------------------------------------------------


def _csv_rows():
    return [{
        "id": "abc123", "company": "CsvCo", "title": "Data Intern", "url": "csvurl",
        "source": "ashby", "score": "88", "posted_at": "2026-07-01", "location": "Remote",
    }]


def test_resolve_by_id_from_csv(patched, monkeypatch):
    monkeypatch.setattr(package.store, "load_listings", _csv_rows)
    monkeypatch.setattr(package.llm, "complete", lambda *a, **k: "text")
    pkg = package.build_package(listing_id="abc123")
    md = _read_md(pkg)
    assert "CsvCo" in md
    assert "Data Intern" in md
    assert "88" in md


def test_explicit_flags_override_csv(patched, monkeypatch):
    monkeypatch.setattr(package.store, "load_listings", _csv_rows)
    monkeypatch.setattr(package.llm, "complete", lambda *a, **k: "text")
    pkg = package.build_package(listing_id="abc123", company="OverrideCo")
    md = _read_md(pkg)
    assert "OverrideCo" in md
    assert "CsvCo" not in md


def test_missing_id_errors_cleanly(patched, monkeypatch):
    monkeypatch.setattr(package.store, "load_listings",
                        lambda: [{"id": "other", "company": "X", "title": "Y"}])
    with pytest.raises(SystemExit):
        package.build_package(listing_id="nope")


def test_bypass_csv_without_company_errors_cleanly(patched, monkeypatch):
    monkeypatch.setattr(package.llm, "complete", lambda *a, **k: "text")
    with pytest.raises(SystemExit):
        package.build_package(title="ML Intern")


# ---------------------------------------------------------------------------
# values alignment — present when the JD lists values, omitted otherwise
# ---------------------------------------------------------------------------


def test_values_section_present_when_jd_lists_values(patched, monkeypatch):
    monkeypatch.setattr(package.jdfetch, "fetch_jd",
                        lambda *a, **k: "Our values:\n- Ownership\n- Craft\n\nResponsibilities: ship.")
    monkeypatch.setattr(package.llm, "complete", lambda *a, **k: "text")
    pkg = package.build_package(company="Co", title="ML Intern", url="u", source="greenhouse")
    md = _read_md(pkg)
    assert "VALUES ALIGNMENT" in md
    assert "Ownership" in md


def test_values_section_omitted_when_no_values(patched, monkeypatch):
    monkeypatch.setattr(package.llm, "complete", lambda *a, **k: "text")
    pkg = package.build_package(company="Co", title="ML Intern", url="u", source="greenhouse")
    md = _read_md(pkg)
    assert "VALUES ALIGNMENT" not in md
