"""Unit tests for the high-fit / rest partition. Pure logic; no network or API key
(rank.partition_by_fit doesn't touch Anthropic).
"""
from src import rank


def L(id, score):
    return {"id": id, "score": score, "company": "X", "title": "ML Intern", "track": "ml"}


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
