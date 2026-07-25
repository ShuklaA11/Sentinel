"""Unit tests for the JD keyword-coverage engine.

Pure/offline: self-contained fixtures (no config/, no profile.yml, no network).
AAA style, mirroring tests/test_filter.py.
"""
from src import keywords as kw


# --- extract_jd_keywords -----------------------------------------------------

def test_extract_lowercases_and_drops_stopwords():
    # Arrange
    text = "We are looking for a strong Python and Kubernetes engineer"

    # Act
    result = kw.extract_jd_keywords(text)

    # Assert: filler/common words gone, real skills kept + lowercased.
    assert "python" in result
    assert "kubernetes" in result
    for filler in ("we", "are", "looking", "for", "a", "strong", "and"):
        assert filler not in result


def test_extract_dedupes_preserving_first_seen_order():
    # Arrange
    text = "Python python PYTHON Rust python"

    # Act
    result = kw.extract_jd_keywords(text)

    # Assert
    assert result == ["python", "rust"]


def test_extract_captures_curated_bigram_when_adjacent():
    # Arrange
    text = "Experience with machine learning and computer vision required"

    # Act
    result = kw.extract_jd_keywords(text)

    # Assert
    assert "machine learning" in result
    assert "computer vision" in result


def test_extract_all_required_bigrams():
    # Arrange
    required = [
        "machine learning", "computer vision", "deep learning",
        "data pipeline", "reinforcement learning", "time series",
    ]
    text = "; ".join(required)

    # Act
    result = kw.extract_jd_keywords(text)

    # Assert
    for bigram in required:
        assert bigram in result


def test_extract_does_not_capture_bigram_when_words_not_adjacent():
    # Arrange: 'machine' and 'learning' present but separated.
    text = "the machine performs learning tasks"

    # Act
    result = kw.extract_jd_keywords(text)

    # Assert
    assert "machine learning" not in result


def test_extract_empty_text_returns_empty_list():
    # Arrange / Act / Assert
    assert kw.extract_jd_keywords("") == []
    assert kw.extract_jd_keywords("   ,;.  ") == []


def test_extract_drops_job_posting_filler():
    # Arrange
    text = "Join our team to work in a role that values experience and skills"

    # Act
    result = kw.extract_jd_keywords(text)

    # Assert: every required filler stopword is absent.
    for filler in ("join", "team", "work", "role", "experience", "skills", "ability"):
        assert filler not in result


def test_extract_drops_proper_nouns_and_corporate_noise():
    # Arrange: company/person/place proper nouns + boilerplate around real skills.
    text = (
        "PlusAI, backed by Scania and Hyundai and Traton, headquartered in "
        "Silicon Valley, builds deep learning and object detection with PyTorch"
    )

    # Act
    result = kw.extract_jd_keywords(text)

    # Assert: mid-sentence proper nouns + corporate/geographic filler are gone.
    for noise in (
        "plusai", "scania", "hyundai", "traton", "silicon", "valley",
        "headquartered",
    ):
        assert noise not in result

    # Real skills survive: curated bigrams + allowlisted tech term.
    assert "deep learning" in result
    assert "object detection" in result
    assert "pytorch" in result


# --- coverage ----------------------------------------------------------------

def test_coverage_splits_present_and_missing():
    # Arrange
    cv = "Built services in Python and Go with Docker."
    jd = ["python", "go", "rust", "docker"]

    # Act
    frac, present, missing = kw.coverage(cv, jd)

    # Assert
    assert present == ["python", "go", "docker"]
    assert missing == ["rust"]
    assert frac == 3 / 4


def test_coverage_empty_keywords_is_zero_no_division():
    # Arrange / Act
    frac, present, missing = kw.coverage("anything", [])

    # Assert
    assert frac == 0.0
    assert present == []
    assert missing == []


def test_coverage_uses_word_boundary_matching():
    # Arrange: 'intern' must NOT match inside 'international'.
    cv = "Worked on international logistics platforms."
    jd = ["intern"]

    # Act
    frac, present, missing = kw.coverage(cv, jd)

    # Assert
    assert present == []
    assert missing == ["intern"]
    assert frac == 0.0


def test_coverage_matches_multiword_bigram_phrase():
    # Arrange
    cv = "Deployed a machine learning model to production."
    jd = ["machine learning", "kubernetes"]

    # Act
    frac, present, missing = kw.coverage(cv, jd)

    # Assert
    assert present == ["machine learning"]
    assert missing == ["kubernetes"]


def test_coverage_is_case_insensitive():
    # Arrange
    cv = "PYTHON and pyTorch experts"
    jd = ["python", "pytorch"]

    # Act
    frac, present, _ = kw.coverage(cv, jd)

    # Assert
    assert frac == 1.0
    assert present == ["python", "pytorch"]


# --- truthful_injection_candidates -------------------------------------------

def test_injection_surfaces_genuinely_held_keywords():
    # Arrange
    missing = ["pytorch", "tensorflow", "machine learning"]
    profile = ["PyTorch", "machine learning frameworks"]

    # Act
    result = kw.truthful_injection_candidates(missing, profile)

    # Assert: pytorch + phrase-in-longer-skill surfaced; tensorflow not held.
    assert result == ["pytorch", "machine learning"]


def test_injection_never_returns_unbacked_keyword():
    # Arrange: candidate knows JavaScript, JD wants 'java' — not genuinely held.
    missing = ["java"]
    profile = ["JavaScript", "TypeScript"]

    # Act
    result = kw.truthful_injection_candidates(missing, profile)

    # Assert
    assert result == []


def test_injection_preserves_missing_order_and_dedupes():
    # Arrange
    missing = ["go", "python", "go"]
    profile = ["Go", "Python"]

    # Act
    result = kw.truthful_injection_candidates(missing, profile)

    # Assert
    assert result == ["go", "python"]


def test_injection_empty_inputs_return_empty():
    # Arrange / Act / Assert
    assert kw.truthful_injection_candidates([], ["python"]) == []
    assert kw.truthful_injection_candidates(["python"], []) == []


# --- profile_skill_terms -----------------------------------------------------

def test_profile_skill_terms_flattens_all_sources_lowercased():
    # Arrange: mirrors profile.yml's real structure.
    profile = {
        "skills": {
            "languages": ["Python", "Go"],
            "frameworks": ["PyTorch"],
            "tools": ["Docker"],
        },
        "preferences": {"skills": ["Kubernetes"]},
        "domain_tags": ["Computer Vision"],
    }

    # Act
    terms = kw.profile_skill_terms(profile)

    # Assert
    for expected in ("python", "go", "pytorch", "docker", "kubernetes", "computer vision"):
        assert expected in terms
    assert all(t == t.lower() for t in terms)


def test_profile_skill_terms_missing_keys_do_not_raise():
    # Arrange / Act / Assert: empty and partial profiles never raise.
    assert kw.profile_skill_terms({}) == []
    assert kw.profile_skill_terms({"skills": {}}) == []
    partial = kw.profile_skill_terms({"skills": {"languages": ["Rust"]}})
    assert partial == ["rust"]


def test_profile_skill_terms_dedupes_across_sources():
    # Arrange
    profile = {
        "skills": {"languages": ["Python"], "tools": ["python"]},
        "preferences": {"skills": ["Python"]},
    }

    # Act
    terms = kw.profile_skill_terms(profile)

    # Assert
    assert terms == ["python"]


def test_profile_terms_feed_injection_end_to_end():
    # Arrange: the intended pipeline — profile terms back injection candidates.
    profile = {"skills": {"frameworks": ["PyTorch"]}, "domain_tags": ["NLP"]}
    missing = ["pytorch", "tensorflow"]

    # Act
    terms = kw.profile_skill_terms(profile)
    result = kw.truthful_injection_candidates(missing, terms)

    # Assert
    assert result == ["pytorch"]
