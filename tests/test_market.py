"""Isolation tests for PriceBucketMarket state + interface methods."""

import math

import pytest

from fingerprint_eval.dpm import MarketConfig, Order, PriceBucketMarket
from fingerprint_eval.dpm.market import CREATOR_ADDRESS

SQRT7 = math.sqrt(7.0)


def make_market(fee=0.0):
    config = MarketConfig(k=1.0, fee=fee, band_edges=[0.3, 0.6])
    return PriceBucketMarket(config, initial_shares=[1.0, 1.0, 1.0])


def four_outcome_market(fee=0.0):
    config = MarketConfig(k=1.0, fee=fee)
    return PriceBucketMarket(config, initial_shares=[1.0, 1.0, 1.0, 1.0])


def test_initial_state_invariants():
    m = four_outcome_market()
    assert m.pool == pytest.approx(2.0)
    assert m.current_prices() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert sum(p * p for p in m.current_prices()) == pytest.approx(m.k ** 2)
    assert m.current_probabilities() == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert sum(m.current_probabilities()) == pytest.approx(1.0)


def test_cost_to_buy_matches_curve():
    m = four_outcome_market()
    assert m.cost_to_buy(0, 1.0) == pytest.approx(SQRT7 - 2.0)


def test_apply_buy_moves_price_and_records_position():
    m = four_outcome_market()
    result = m.apply_trade(Order("buy", 0, 1.0))
    assert result.ok
    assert result.cost == pytest.approx(SQRT7 - 2.0)
    assert result.fill_price == pytest.approx(SQRT7 - 2.0)
    assert m.q == pytest.approx([2.0, 1.0, 1.0, 1.0])
    assert result.prices_after[0] == pytest.approx(2.0 / SQRT7)
    assert m.position("agent")[0] == pytest.approx(1.0)
    # probabilities still sum to 1 after the trade
    assert sum(result.probabilities_after) == pytest.approx(1.0)


def test_buy_then_sell_round_trip_costs_only_fees():
    m = four_outcome_market(fee=0.0)
    buy = m.apply_trade(Order("buy", 0, 1.0))
    sell = m.apply_trade(Order("sell", 0, 1.0))
    assert sell.proceeds == pytest.approx(buy.cost)
    assert m.q == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert m.position("agent")[0] == pytest.approx(0.0)


def test_hold_is_a_successful_noop():
    m = four_outcome_market()
    before = list(m.q)
    result = m.apply_trade(Order("hold"))
    assert result.ok
    assert result.side == "hold"
    assert m.q == before


def test_oversell_is_rejected_not_raised():
    m = four_outcome_market()
    result = m.apply_trade(Order("sell", 0, 1.0))  # agent holds nothing
    assert not result.ok
    assert result.error is not None
    assert m.q == pytest.approx([1.0, 1.0, 1.0, 1.0])  # state unchanged


def test_bad_outcome_and_side_are_rejected():
    m = four_outcome_market()
    assert not m.apply_trade(Order("buy", 9, 1.0)).ok
    assert not m.apply_trade(Order("nonsense", 0, 1.0)).ok
    assert not m.apply_trade(Order("buy", 0, -5.0)).ok


def test_settlement_pays_out_exactly_the_pool():
    m = four_outcome_market()
    m.apply_trade(Order("buy", 0, 1.0))  # agent now holds 1 of outcome 0; creator holds 1
    pool = m.pool
    payouts = m.settle(winning_outcome=0)
    assert payouts["agent"] == pytest.approx(pool / 2.0)
    assert payouts[CREATOR_ADDRESS] == pytest.approx(pool / 2.0)
    assert sum(payouts.values()) == pytest.approx(pool)


def test_settlement_losing_holders_get_nothing():
    m = four_outcome_market()
    m.apply_trade(Order("buy", 0, 1.0))
    payouts = m.settle(winning_outcome=1)  # agent bet on 0, loses
    assert "agent" not in payouts
    assert payouts[CREATOR_ADDRESS] == pytest.approx(m.pool)


def test_liquidation_recovers_spot_value():
    m = four_outcome_market()
    m.apply_trade(Order("buy", 0, 1.0))
    assert m.liquidate("agent") == pytest.approx(1.0 / SQRT7)


def test_bucket_for_price():
    m = make_market()  # band_edges = [0.3, 0.6]
    assert m.bucket_for_price(0.1) == 0
    assert m.bucket_for_price(0.3) == 1  # edge is exclusive lower bound
    assert m.bucket_for_price(0.45) == 1
    assert m.bucket_for_price(0.6) == 2
    assert m.bucket_for_price(0.9) == 2


def test_config_validation():
    with pytest.raises(ValueError):
        MarketConfig(k=0.0)
    with pytest.raises(ValueError):
        MarketConfig(k=1.0, fee=1.0)
    with pytest.raises(ValueError):
        PriceBucketMarket(MarketConfig(k=1.0), initial_shares=[1.0])  # need >=2
    with pytest.raises(ValueError):
        PriceBucketMarket(MarketConfig(k=1.0), initial_shares=[1.0, 0.0])  # positive seed


def test_band_edges_length_validation():
    with pytest.raises(ValueError):
        PriceBucketMarket(
            MarketConfig(k=1.0, band_edges=[0.5]),  # needs 2 edges for 3 outcomes
            initial_shares=[1.0, 1.0, 1.0],
        )
