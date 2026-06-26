"""Isolation tests for the DPM math against hand-computed values.

Per spec.md build order step 1: a known buy must move price by the known
amount, settlement must pay out exactly the pool, and prices must obey the
verified invariants. If any of this is wrong, every downstream result is
silently wrong, so these are deliberately explicit and hand-checked.
"""

import math

import pytest

from agent_eval_harness.dpm import (
    buy_cost,
    collateral,
    implied_probabilities,
    liquidation_reward,
    sell_proceeds,
    settlement_payout,
    spot_prices,
    sum_term,
)

SQRT7 = math.sqrt(7.0)  # 2.6457513110645907


def test_sum_term_and_collateral():
    q = [1.0, 1.0, 1.0, 1.0]
    assert sum_term(q) == 4.0
    assert collateral(q, k=1.0) == 2.0
    assert collateral(q, k=3.0) == 6.0


def test_uniform_spot_prices_and_probabilities():
    q = [1.0, 1.0, 1.0, 1.0]
    prices = spot_prices(q, k=1.0)
    assert prices == pytest.approx([0.5, 0.5, 0.5, 0.5])
    # prices live on a sphere of radius k: sum of squares == k**2
    assert sum(p * p for p in prices) == pytest.approx(1.0)
    probs = implied_probabilities(q)
    assert probs == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert sum(probs) == pytest.approx(1.0)


def test_prices_sum_of_squares_equals_k_squared_nonuniform():
    q = [3.0, 1.0, 7.0, 2.0]
    k = 2.5
    prices = spot_prices(q, k)
    assert sum(p * p for p in prices) == pytest.approx(k * k)


def test_implied_probability_is_price_squared_over_k_squared():
    # The critical correctness point: probability = (price / k)**2, NOT price.
    q = [3.0, 1.0, 7.0, 2.0]
    k = 2.5
    prices = spot_prices(q, k)
    probs = implied_probabilities(q)
    assert probs == pytest.approx([(p / k) ** 2 for p in prices])
    assert sum(probs) == pytest.approx(1.0)


def test_implied_probability_documented_examples():
    # From dpm_spec.md: 0.51/share -> ~26% (not 51%), 0.65/share -> ~42% (not 65%).
    assert (0.51 / 1.0) ** 2 == pytest.approx(0.2601, abs=1e-4)
    assert (0.65 / 1.0) ** 2 == pytest.approx(0.4225, abs=1e-4)


def test_known_buy_cost_and_price_move():
    q = [1.0, 1.0, 1.0, 1.0]
    # Buy 1 share of outcome 0, no fee. newSumTerm = 4 + 1*(2*1+1) = 7.
    cost = buy_cost(q, k=1.0, j=0, delta=1.0, fee=0.0)
    assert cost == pytest.approx(SQRT7 - 2.0)  # 0.6457513110645907

    # Apply the move by hand and check the new spot price of outcome 0.
    q_after = [2.0, 1.0, 1.0, 1.0]
    assert sum_term(q_after) == 7.0
    p0_before = spot_prices(q, 1.0)[0]
    p0_after = spot_prices(q_after, 1.0)[0]
    assert p0_before == pytest.approx(0.5)
    assert p0_after == pytest.approx(2.0 / SQRT7)  # 0.7559289460184544


def test_buy_fee_inflates_cost():
    q = [1.0, 1.0, 1.0, 1.0]
    gross = buy_cost(q, k=1.0, j=0, delta=1.0, fee=0.0)
    with_fee = buy_cost(q, k=1.0, j=0, delta=1.0, fee=0.02)
    assert with_fee == pytest.approx(gross / 0.98)


def test_sell_reverses_buy_without_fee():
    q_after_buy = [2.0, 1.0, 1.0, 1.0]
    # Selling the share just bought returns exactly the gross buy cost.
    proceeds = sell_proceeds(q_after_buy, k=1.0, j=0, delta=1.0, fee=0.0)
    assert proceeds == pytest.approx(SQRT7 - 2.0)


def test_sell_fee_reduces_proceeds():
    q_after_buy = [2.0, 1.0, 1.0, 1.0]
    gross = sell_proceeds(q_after_buy, k=1.0, j=0, delta=1.0, fee=0.0)
    with_fee = sell_proceeds(q_after_buy, k=1.0, j=0, delta=1.0, fee=0.02)
    assert with_fee == pytest.approx(gross * 0.98)


def test_sell_more_than_outstanding_raises():
    with pytest.raises(ValueError):
        sell_proceeds([2.0, 1.0, 1.0, 1.0], k=1.0, j=0, delta=3.0)


def test_settlement_splits_pool_exactly():
    q = [2.0, 1.0, 1.0, 1.0]
    pool = collateral(q, k=1.0)  # sqrt(7)
    # Two holders of outcome 0 with 1 share each split the pool evenly.
    a = settlement_payout(pool, winning_shares=1.0, total_winning_shares=2.0)
    b = settlement_payout(pool, winning_shares=1.0, total_winning_shares=2.0)
    assert a == pytest.approx(pool / 2.0)
    assert a + b == pytest.approx(pool)  # pays out exactly the pool


def test_settlement_no_winning_shares_is_zero():
    assert settlement_payout(pool=10.0, winning_shares=0.0, total_winning_shares=0.0) == 0.0


def test_liquidation_recovers_spot_value():
    q = [2.0, 1.0, 1.0, 1.0]
    held = [1.0, 0.0, 0.0, 0.0]
    reward = liquidation_reward(q, k=1.0, held=held)
    assert reward == pytest.approx(1.0 / SQRT7)  # 0.3779644730092272


def test_spot_price_undefined_on_empty_pool():
    with pytest.raises(ValueError):
        spot_prices([0.0, 0.0], k=1.0)
