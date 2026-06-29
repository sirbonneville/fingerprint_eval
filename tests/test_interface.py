"""Tests for the thin Market seam (spec.md piece 2) and the notional<->shares
inversion the harness relies on."""

import math

import pytest

from fingerprint_eval.dpm import (
    MarketConfig,
    Order,
    PriceBucketMarket,
    shares_for_buy_spend,
    shares_for_sell_proceeds,
)
from fingerprint_eval.market import Market, MarketState, OutcomeInfo

SQRT7 = math.sqrt(7.0)


def banded_market(fee=0.0, trading_close_min=None):
    config = MarketConfig(
        k=1.0,
        fee=fee,
        band_edges=[0.3, 0.6],
        settlement_rule_text="Settles to the band containing the reference price at close.",
        trading_close_min=trading_close_min,
    )
    return PriceBucketMarket(config, initial_shares=[1.0, 1.0, 1.0])


# ----------------------------------------------------------------- conformance


def test_price_bucket_market_is_a_market():
    assert issubclass(PriceBucketMarket, Market)
    assert isinstance(banded_market(), Market)


def test_outcomes_describe_bands():
    infos = banded_market().outcomes()
    assert [o.index for o in infos] == [0, 1, 2]
    assert all(isinstance(o, OutcomeInfo) for o in infos)
    # edges [0.3, 0.6] -> (-inf,0.3), [0.3,0.6), [0.6,+inf)
    assert infos[0].band_low is None and infos[0].band_high == pytest.approx(0.3)
    assert infos[1].band_low == pytest.approx(0.3) and infos[1].band_high == pytest.approx(0.6)
    assert infos[2].band_low == pytest.approx(0.6) and infos[2].band_high is None


def test_outcomes_use_explicit_labels_when_given():
    config = MarketConfig(k=1.0, outcome_labels=["lo", "mid", "hi"], band_edges=[0.3, 0.6])
    infos = PriceBucketMarket(config, [1.0, 1.0, 1.0]).outcomes()
    assert [o.label for o in infos] == ["lo", "mid", "hi"]


def test_current_state_shape_and_values():
    m = banded_market()
    state = m.current_state()
    assert isinstance(state, MarketState)
    assert state.prices == pytest.approx([1 / math.sqrt(3)] * 3)
    assert sum(state.probabilities) == pytest.approx(1.0)
    assert state.pool == pytest.approx(math.sqrt(3.0))
    assert state.my_position == [0.0, 0.0, 0.0]
    assert state.time_to_close is None  # no clock configured


def test_settlement_rule_text_passes_through():
    m = banded_market()
    assert "reference price" in m.settlement_rule_text()


# ----------------------------------------------------------------- clock


def test_time_to_close_tracks_the_clock():
    m = banded_market(trading_close_min=120.0)
    assert m.current_state().time_to_close == pytest.approx(120.0)
    m.set_time(90.0)
    assert m.current_state().time_to_close == pytest.approx(30.0)
    m.set_time(200.0)  # past close clamps to 0, never negative
    assert m.current_state().time_to_close == pytest.approx(0.0)


# ----------------------------------------------------------------- inversion (math)


@pytest.mark.parametrize("fee", [0.0, 0.02])
@pytest.mark.parametrize("spend", [0.1, 0.5, 1.0, 5.0])
def test_buy_spend_inversion_round_trips(fee, spend):
    q = [3.0, 1.0, 7.0, 2.0]
    k = 2.5
    from fingerprint_eval.dpm import buy_cost

    shares = shares_for_buy_spend(q, k, j=1, spend=spend, fee=fee)
    assert shares > 0.0
    assert buy_cost(q, k, j=1, delta=shares, fee=fee) == pytest.approx(spend)


def test_known_buy_spend_inversion():
    # Spending exactly the cost of 1 share should return 1 share.
    q = [1.0, 1.0, 1.0, 1.0]
    shares = shares_for_buy_spend(q, k=1.0, j=0, spend=SQRT7 - 2.0, fee=0.0)
    assert shares == pytest.approx(1.0)


@pytest.mark.parametrize("fee", [0.0, 0.02])
def test_sell_proceeds_inversion_round_trips(fee):
    from fingerprint_eval.dpm import sell_proceeds

    q = [2.0, 1.0, 1.0, 1.0]
    target = sell_proceeds(q, k=1.0, j=0, delta=0.5, fee=fee)
    shares = shares_for_sell_proceeds(q, k=1.0, j=0, proceeds=target, fee=fee)
    assert shares == pytest.approx(0.5)


def test_sell_inversion_clamps_to_outstanding_shares():
    q = [2.0, 1.0, 1.0, 1.0]
    huge = shares_for_sell_proceeds(q, k=1.0, j=0, proceeds=1e9, fee=0.0)
    assert huge == pytest.approx(2.0)  # cannot sell more of outcome 0 than exists


# ----------------------------------------------------------------- inversion (market quote)


def test_market_quote_then_trade_consumes_the_budget():
    # The flow the harness will use: notional -> shares (quote) -> Order -> fill.
    m = banded_market(fee=0.02)
    budget = 5.0
    shares = m.shares_for_spend(outcome=2, spend=budget)
    result = m.apply_trade(Order("buy", 2, shares))
    assert result.ok
    assert result.cost == pytest.approx(budget)  # the fill costs exactly the budget
