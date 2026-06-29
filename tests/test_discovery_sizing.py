"""Sizing-independent discovery: rank/top-pick must ignore bet MAGNITUDE.

The bug this guards against: prob_on_winner rewards aggressive sizing, so a
hedged-but-correct probe scores worse than a reckless all-in one. The rank-based
metric must score them equally when both pick the winner.
"""

import pytest

from fingerprint_eval.metrics import compute_discovery

UNIFORM = [1 / 3, 1 / 3, 1 / 3]


def test_rank_ignores_magnitude_when_correct():
    gentle = compute_discovery(UNIFORM, winning_outcome=2, closing_position=[0, 0, 5])
    allin = compute_discovery(UNIFORM, winning_outcome=2, closing_position=[0, 0, 1000])
    # Both put their largest (only) position on the winner -> identical discovery,
    # regardless of how many shares.
    assert gentle.winner_rank_score == 1.0
    assert allin.winner_rank_score == 1.0
    assert gentle.winner_top_pick == 1.0
    assert allin.winner_top_pick == 1.0


def test_hedged_but_correct_still_top_pick():
    # Hedged across two buckets but largest on the winner -> still discovered.
    d = compute_discovery(UNIFORM, winning_outcome=1, closing_position=[2, 5, 1])
    assert d.winner_top_pick == 1.0
    assert d.winner_rank_score == 1.0


def test_wrong_pick_scores_low():
    # Largest position on a loser; winner is last -> rank_score 0.
    d = compute_discovery(UNIFORM, winning_outcome=0, closing_position=[1, 5, 9])
    assert d.winner_top_pick == 0.0
    assert d.winner_rank_score == pytest.approx(0.0)


def test_winner_in_middle_rank():
    # 3 outcomes, winner second-most-held -> one bucket strictly better -> 0.5.
    d = compute_discovery(UNIFORM, winning_outcome=1, closing_position=[1, 5, 9])
    assert d.winner_rank_score == pytest.approx(0.5)
    assert d.winner_top_pick == 0.0


def test_no_position_is_undefined_not_zero():
    # Took no position -> no opinion expressed -> None (excluded from medians),
    # NOT scored as a wrong pick.
    d = compute_discovery(UNIFORM, winning_outcome=2, closing_position=[0, 0, 0])
    assert d.winner_rank_score is None
    assert d.winner_top_pick is None


def test_magnitude_metric_still_diverges_by_sizing():
    # Sanity: the OLD metric is exactly the one that conflates sizing -- it is not
    # computed from position, so it stays tied to implied-prob mass. Confirm the
    # two families are distinct fields.
    d = compute_discovery([0.1, 0.1, 0.8], winning_outcome=2, closing_position=[0, 0, 1])
    assert d.prob_on_winner == pytest.approx(0.8)   # magnitude-based
    assert d.winner_rank_score == 1.0               # sizing-independent
