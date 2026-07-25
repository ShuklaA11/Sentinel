"""Unit tests for the shared Anthropic→OpenAI completion helper.

All provider calls are monkeypatched — no network, no API key required. Covers the
provider-fallback semantics (Claude primary, GPT fallback) and the unavailable-vs-transient
heuristic that decides drop-provider vs. retry-next.
"""
from src import llm


def _fake_provider(name, fn):
    """A provider whose .call(client, prompt, model, max_tokens) delegates to fn(prompt)."""
    return llm._Provider(
        name, object(),
        lambda _client, prompt, _model, _max, _fn=fn: _fn(prompt),
    )


def _billing_error(_prompt):
    raise Exception("Error code: 400 - Your credit balance is too low")


def test_anthropic_success_returns_its_text(monkeypatch):
    monkeypatch.setattr(llm, "_providers", lambda: [
        _fake_provider("anthropic", lambda _p: "  hello from claude  "),
        _fake_provider("openai", lambda _p: "should not be reached"),
    ])
    assert llm.complete("prompt") == "hello from claude"  # first success, stripped


def test_unavailable_primary_falls_back_to_openai(monkeypatch):
    monkeypatch.setattr(llm, "_providers", lambda: [
        _fake_provider("anthropic", _billing_error),          # primary out of credits
        _fake_provider("openai", lambda _p: "gpt answer"),    # fallback works
    ])
    assert llm.complete("prompt") == "gpt answer"


def test_both_unavailable_returns_none(monkeypatch):
    monkeypatch.setattr(llm, "_providers", lambda: [
        _fake_provider("anthropic", _billing_error),
        _fake_provider("openai", _billing_error),
    ])
    assert llm.complete("prompt") is None


def test_no_providers_returns_none(monkeypatch):
    monkeypatch.setattr(llm, "_providers", lambda: [])
    assert llm.complete("prompt") is None


def test_transient_primary_falls_through_to_openai(monkeypatch):
    def _transient(_p):
        raise Exception("Connection reset by peer")
    monkeypatch.setattr(llm, "_providers", lambda: [
        _fake_provider("anthropic", _transient),              # non-unavailable blip
        _fake_provider("openai", lambda _p: "gpt answer"),
    ])
    assert llm.complete("prompt") == "gpt answer"


def test_empty_text_falls_through_to_next_provider(monkeypatch):
    monkeypatch.setattr(llm, "_providers", lambda: [
        _fake_provider("anthropic", lambda _p: "   "),        # non-empty stripped -> ""
        _fake_provider("openai", lambda _p: "gpt answer"),
    ])
    assert llm.complete("prompt") == "gpt answer"


def test_is_unavailable_distinguishes_billing_from_transient():
    assert llm._is_unavailable(Exception("Your credit balance is too low")) is True
    assert llm._is_unavailable(Exception("insufficient_quota for this org")) is True
    assert llm._is_unavailable(Exception("Connection reset by peer")) is False
