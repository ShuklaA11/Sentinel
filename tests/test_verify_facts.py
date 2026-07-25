"""Unit tests for the numeric fact-check gate.

Self-contained and offline: the core assertions pass `source_texts` directly and
never touch the real profile/ files. `load_sources` is exercised against a temp
directory via monkeypatched module paths, so nothing here depends on the live
profile.yml / resume.tex.
"""
from __future__ import annotations

from src import verify_facts as vf

# ---------------------------------------------------------------------------
# extract_metrics — shape coverage
# ---------------------------------------------------------------------------


def test_extract_percentage():
    assert vf.extract_metrics("improved accuracy by 94%") == {"94%"}


def test_extract_percentage_with_comma_normalized():
    assert vf.extract_metrics("cut latency 3,050%") == {"3050%"}


def test_extract_currency_with_suffix():
    assert vf.extract_metrics("raised $1.2M in seed") == {"$1.2m"}


def test_extract_currency_with_commas():
    assert vf.extract_metrics("a grant of $50,000") == {"$50000"}


def test_extract_multiplier_lowercase():
    assert vf.extract_metrics("3x faster inference") == {"3x"}


def test_extract_multiplier_uppercase_normalized():
    assert vf.extract_metrics("scaled 10X") == {"10x"}


def test_extract_count_with_plus():
    assert vf.extract_metrics("logged 250+ hours") == {"250 hours"}


def test_extract_count_with_commas():
    assert vf.extract_metrics("classified 2,000 crops") == {"2000 crops"}


def test_extract_count_with_k_suffix():
    assert vf.extract_metrics("trained on 100K samples") == {"100k samples"}


def test_extract_count_bare():
    assert vf.extract_metrics("coordinated 15 departments") == {"15 departments"}


def test_extract_empty_string():
    assert vf.extract_metrics("") == set()


def test_extract_no_metrics():
    assert vf.extract_metrics("built a scalable data pipeline") == set()


# ---------------------------------------------------------------------------
# extract_metrics — normalization equivalence
# ---------------------------------------------------------------------------


def test_comma_and_plus_variants_compare_equal():
    a = vf.extract_metrics("2,000 crops")
    b = vf.extract_metrics("2000 crops")
    c = vf.extract_metrics("2,000+ crops")
    assert a == b == c == {"2000 crops"}


def test_multiple_metrics_in_one_string():
    metrics = vf.extract_metrics("raised $50,000, grew 3x, over 250+ hours")
    assert metrics == {"$50000", "3x", "250 hours"}


def test_gpu_name_not_split_into_count():
    # 'A100' is a chip name, not '100 samples' — a digit glued to letters is
    # not a free-standing number.
    assert vf.extract_metrics("ran on an A100 GPU") == set()


# ---------------------------------------------------------------------------
# verify — grounding logic
# ---------------------------------------------------------------------------


def test_verify_grounded_metric_passes():
    ok, violations = vf.verify("boosted revenue 3x", ["we grew the account 3x"])
    assert ok is True
    assert violations == []


def test_verify_fabricated_metric_flagged():
    ok, violations = vf.verify(
        "improved accuracy by 94%", ["improved model accuracy notably"]
    )
    assert ok is False
    assert violations == ["94%"]


def test_verify_union_across_sources():
    ok, violations = vf.verify(
        "raised $1.2M and grew 3x",
        ["closed a $1.2M round", "usage grew 3x last year"],
    )
    assert ok is True
    assert violations == []


def test_verify_partial_fabrication_only_flags_ungrounded():
    ok, violations = vf.verify(
        "grew 3x on $50,000 budget",
        ["scaled the pipeline 3x"],
    )
    assert ok is False
    assert violations == ["$50000"]


def test_verify_violations_are_sorted():
    ok, violations = vf.verify("saw 94%, 3x, and $50,000", ["baseline text"])
    assert ok is False
    assert violations == sorted(violations)
    assert violations == ["$50000", "3x", "94%"]


def test_verify_allow_whitelists_metric():
    ok, violations = vf.verify(
        "logged 250+ hours", ["no numbers here"], allow={"250 hours"}
    )
    assert ok is True
    assert violations == []


def test_verify_allow_accepts_human_form():
    # allow entries in raw human form ('$1.2M') are canonicalized like sources.
    ok, violations = vf.verify(
        "raised $1.2M", ["no numbers here"], allow={"$1.2M"}
    )
    assert ok is True
    assert violations == []


def test_verify_empty_generated_is_ok():
    assert vf.verify("", ["anything 5x"]) == (True, [])


def test_verify_whitespace_generated_is_ok():
    assert vf.verify("   \n\t ", ["anything 5x"]) == (True, [])


def test_verify_no_metrics_in_generated_is_ok():
    ok, violations = vf.verify("great qualitative summary", ["source text"])
    assert ok is True
    assert violations == []


def test_verify_normalization_bridges_comma_forms():
    # generated uses '2,000+', source uses '2000' — same canonical metric.
    ok, violations = vf.verify("shipped 2,000+ crops", ["labeled 2000 crops"])
    assert ok is True
    assert violations == []


# ---------------------------------------------------------------------------
# load_sources — defensive file gathering (temp dir, no real profile deps)
# ---------------------------------------------------------------------------


def test_load_sources_reads_both_files(tmp_path, monkeypatch):
    yml = tmp_path / "profile.yml"
    yml.write_text("facts:\n  metric: 3x\n")
    tex = tmp_path / "resume.tex"
    tex.write_text("Shipped 250+ hours of work.")
    monkeypatch.setattr(vf, "PROFILE_YML", str(yml))
    monkeypatch.setattr(vf, "RESUME_TEX", str(tex))

    sources = vf.load_sources()

    assert len(sources) == 2
    joined = "\n".join(sources)
    assert "3x" in joined
    assert "250+ hours" in joined


def test_load_sources_missing_files_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(vf, "PROFILE_YML", str(tmp_path / "nope.yml"))
    monkeypatch.setattr(vf, "RESUME_TEX", str(tmp_path / "nope.tex"))

    assert vf.load_sources() == []


def test_load_sources_partial_readable(tmp_path, monkeypatch):
    tex = tmp_path / "resume.tex"
    tex.write_text("Only the resume is here.")
    monkeypatch.setattr(vf, "PROFILE_YML", str(tmp_path / "missing.yml"))
    monkeypatch.setattr(vf, "RESUME_TEX", str(tex))

    sources = vf.load_sources()

    assert sources == ["Only the resume is here."]
