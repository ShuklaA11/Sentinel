"""Unit tests for the essay generator. No network: the LLM client is faked and
injected. The em-dash guarantee is the load-bearing test.
"""
from src import essays


# --- Deterministic sanitizer (layer 3) --------------------------------------

def test_has_em_dash_detects_all_forms():
    assert essays.has_em_dash("a — b")      # em
    assert essays.has_em_dash("a – b")      # en
    assert essays.has_em_dash("a -- b")     # ascii double hyphen
    assert not essays.has_em_dash("a - b")  # plain hyphen is fine
    assert not essays.has_em_dash("clean text.")


def test_strip_em_dashes_pause_becomes_comma():
    assert essays.strip_em_dashes("I built it — and it worked.") == "I built it, and it worked."


def test_strip_em_dashes_number_range_becomes_hyphen():
    assert essays.strip_em_dashes("Interned there in 2024–2025.") == "Interned there in 2024-2025."


def test_strip_em_dashes_cleans_artifacts():
    # A dash bumped against a period must not leave a dangling comma.
    assert essays.strip_em_dashes("I shipped it —.") == "I shipped it."
    # Double hyphen too.
    assert essays.strip_em_dashes("fast -- reliable") == "fast, reliable"


def test_strip_em_dashes_leaves_clean_text_untouched():
    s = "I trained a model. It hit 94% precision. That mattered."
    assert essays.strip_em_dashes(s) == s


# --- Fake client -------------------------------------------------------------

class _Resp:
    def __init__(self, text):
        self.content = [type("C", (), {"text": text})()]


class _FakeClient:
    """Returns queued texts in order; records each create() call."""
    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = []

    class _Messages:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kw):
            self.outer.calls.append(kw)
            return _Resp(self.outer._texts.pop(0) if self.outer._texts else "")

    @property
    def messages(self):
        return _FakeClient._Messages(self)


class _RaisingClient:
    class _Messages:
        def create(self, **kw):
            raise RuntimeError("api down")

    @property
    def messages(self):
        return _RaisingClient._Messages()


PROFILE = {"education": {"school": "UT Austin", "degree": "BS Stats", "graduation": "May 2028"},
           "experience": [{"role": "ML Intern", "company": "ChargeScape", "dates": "2026",
                           "bullets": ["Built grid-optimization models"]}]}
VOICE = "Short punchy opener. Concrete detail. No em dashes."


def test_draft_returns_none_without_client(monkeypatch):
    # Simulate no API key: _client() yields nothing, so drafting gracefully skips.
    monkeypatch.setattr(essays, "_client", lambda: None)
    assert essays.draft_answer("Why us?", PROFILE, VOICE) is None


def test_draft_guarantee_holds_even_if_model_never_complies():
    # Model returns em-dash text on BOTH the first draft and the retry.
    fake = _FakeClient(["I love this — truly. Interned 2026 — present.",
                        "Still dashy — see? 2024–2025."])
    out = essays.draft_answer("Why us?", PROFILE, VOICE, client=fake, max_retries=1)
    assert out is not None
    assert not essays.has_em_dash(out)          # guarantee: zero dashes ship
    assert "2024-2025" in out                    # range collapsed to hyphen
    assert len(fake.calls) == 2                   # one draft + one retry


def test_draft_no_retry_when_first_draft_is_clean():
    fake = _FakeClient(["I built grid models at ChargeScape. That work drew me here."])
    out = essays.draft_answer("Why us?", PROFILE, VOICE, client=fake)
    assert len(fake.calls) == 1                   # clean first draft, no retry
    assert not essays.has_em_dash(out)


def test_draft_returns_none_on_api_error():
    assert essays.draft_answer("Why us?", PROFILE, VOICE, client=_RaisingClient()) is None


def test_draft_returns_none_on_skip_sentinel():
    # Non-question label -> model declines with SKIP -> None (never types a reply into the form).
    fake = _FakeClient(["SKIP"])
    assert essays.draft_answer("Note to Hiring Manager:", PROFILE, VOICE, client=fake) is None
    fake = _FakeClient(["Skip."])  # tolerant of trailing punctuation / case
    assert essays.draft_answer("Note to Hiring Manager:", PROFILE, VOICE, client=fake) is None
