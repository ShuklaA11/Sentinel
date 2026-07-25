"""Unit tests for scoring logic and the high-fit / rest partition. Pure logic; no network
or API key (partition_by_fit, _composite, and _parse_batch don't touch Anthropic).
"""
import json

from src import rank


def L(id, score):
    return {"id": id, "score": score, "company": "X", "title": "ML Intern",
            "location": "Remote", "track": "ml"}


def _row(i=0, veto=False, reason="ok", **dims):
    base = {"track": 80, "skill": 80, "logistics": 80, "growth": 80}
    return {"i": i, "veto": veto, "reason": reason, **base, **dims}


def test_composite_is_the_weighted_mean():
    # 0.35*100 + 0.30*100 + 0.20*50 + 0.15*0 = 75
    assert rank._composite(_row(track=100, skill=100, logistics=50, growth=0)) == 75


def test_composite_clamps_into_0_100():
    assert rank._composite(_row(track=200, skill=200, logistics=200, growth=200)) == 100
    assert rank._composite(_row(track=0, skill=0, logistics=0, growth=0)) == 0


def test_veto_forces_zero_and_flags_reason():
    out = rank._parse_batch(json.dumps([_row(i=0, veto=True, reason="full-time role")]))
    assert out[0] == (0, "VETO: full-time role")


def test_veto_without_reason_still_labels():
    out = rank._parse_batch(json.dumps([_row(i=0, veto=True, reason="")]))
    assert out[0] == (0, "VETO")


def test_parse_batch_scores_non_vetoed_row():
    out = rank._parse_batch(json.dumps([_row(i=3, reason="strong ML match")]))
    assert out[3] == (80, "strong ML match")  # all dims 80 -> composite 80


def test_parse_batch_skips_malformed_rows_without_crashing():
    text = json.dumps([{"i": 0, "track": 90}, _row(i=1)])  # first row missing dims
    out = rank._parse_batch(text)
    assert 0 not in out          # malformed -> skipped (listing stays unscored)
    assert out[1] == (80, "ok")


def test_parse_batch_trims_prose_around_the_array():
    text = "Here you go:\n" + json.dumps([_row(i=0)]) + "\nHope that helps!"
    out = rank._parse_batch(text)
    assert out[0][0] == 80


def test_partition_is_inclusive_at_threshold():
    high, rest = rank.partition_by_fit([L("a", 90), L("b", 85), L("c", 84), L("d", 10)], 85)
    assert {l["id"] for l in high} == {"a", "b"}   # >= 85
    assert {l["id"] for l in rest} == {"c", "d"}


def test_unscored_listings_go_to_rest_without_crashing():
    # score == "" must not be compared to an int (TypeError) nor counted as high-fit.
    high, rest = rank.partition_by_fit([L("a", 95), L("b", "")], 85)
    assert [l["id"] for l in high] == ["a"]
    assert [l["id"] for l in rest] == ["b"]


def test_no_high_fit_returns_everything_as_rest():
    high, rest = rank.partition_by_fit([L("a", 50), L("b", 70)], 85)
    assert high == []
    assert len(rest) == 2


# --- provider fallback + loud-failure signal ---------------------------------

READY_PROFILE = {"background": "real ML background", "preferences": {"track": "ml"}}
GOOD_JSON = json.dumps([_row(i=0, reason="match")])  # all dims 80 -> composite 80


def _fake_provider(name, fn):
    """A provider whose .call(client, prompt) delegates to fn(prompt)."""
    return rank._Provider(name, object(), lambda _client, prompt, _fn=fn: _fn(prompt))


def _billing_error(_prompt):
    raise Exception("Error code: 400 - Your credit balance is too low")


def test_is_unavailable_distinguishes_billing_from_transient():
    assert rank._is_unavailable(Exception("Your credit balance is too low")) is True
    assert rank._is_unavailable(Exception("insufficient_quota for this org")) is True
    assert rank._is_unavailable(Exception("Connection timed out")) is False


def test_fallback_scores_when_primary_is_unavailable(monkeypatch):
    monkeypatch.setattr(rank, "_providers", lambda: [
        _fake_provider("anthropic", _billing_error),      # primary: out of credits
        _fake_provider("openai", lambda _p: GOOD_JSON),   # fallback: works
    ])
    scored, warning = rank.score_listings([L("a", "")], READY_PROFILE)
    assert warning is None                 # a working fallback = no alarm
    assert scored[0]["score"] == 80        # scored by GPT


def test_all_providers_unavailable_returns_loud_warning(monkeypatch):
    monkeypatch.setattr(rank, "_providers", lambda: [
        _fake_provider("anthropic", _billing_error),
        _fake_provider("openai", _billing_error),
    ])
    scored, warning = rank.score_listings([L("a", "")], READY_PROFILE)
    assert scored[0]["score"] == ""        # unscored fallthrough (no crash)
    assert warning and "unavailable" in warning


def test_no_providers_is_a_quiet_skip(monkeypatch):
    monkeypatch.setattr(rank, "_providers", lambda: [])
    scored, warning = rank.score_listings([L("a", "")], READY_PROFILE)
    assert warning is None                 # no keys = expected, not an alarm
    assert scored[0]["score"] == ""


def test_transient_failure_does_not_trip_the_warning(monkeypatch):
    def _transient(_p):
        raise Exception("Connection reset by peer")
    monkeypatch.setattr(rank, "_providers", lambda: [_fake_provider("anthropic", _transient)])
    scored, warning = rank.score_listings([L("a", "")], READY_PROFILE)
    assert warning is None                 # provider stays alive -> not "unavailable"
    assert scored[0]["score"] == ""
