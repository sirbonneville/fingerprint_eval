"""End-to-end and unit tests for the harness loop (spec.md piece 4)."""

import json
import re

import pytest

from agent_eval_harness.action import parse
from agent_eval_harness.dpm import MarketConfig, PriceBucketMarket
from agent_eval_harness.harness import AGENT_ADDRESS, run_episode
from agent_eval_harness.market import MarketState, Order
from agent_eval_harness.model import HoldModel, RandomTrader, ScriptedModel
from agent_eval_harness.prompt import build_prompt
from agent_eval_harness.price_path import generate_path
from agent_eval_harness.wallet import Wallet, plan_order


def make_market(trading_close_min=120.0):
    config = MarketConfig(
        k=10.0,
        fee=0.02,
        band_edges=[95.0, 105.0],
        outcome_labels=["below 95", "95-105", "above 105"],
        settlement_rule_text="Settles to the band containing ZQX's price at close.",
        trading_close_min=trading_close_min,
    )
    return PriceBucketMarket(config, initial_shares=[100.0, 100.0, 100.0])


def make_path():
    return generate_path(
        anchor=100.0, hours=2, interval_min=30, volatility=0.1, drift=0.0,
        post_close_min=60, seed=42,
    )


# ----------------------------------------------------------------- wallet / planning


def _state(prices, position):
    n = len(prices)
    return MarketState(
        prices=prices,
        probabilities=[1.0 / n] * n,
        pool=1.0,
        my_position=position,
        time_to_close=10.0,
    )


def test_plan_order_rejects_overspend():
    m = make_market()
    w = Wallet.with_budget(50.0)
    action = parse('{"action":"buy","outcome":1,"size":100}')  # wants 100, has 50
    order, reason = plan_order(m, w, action, m.current_state())
    assert order is None
    assert "insufficient cash" in reason


def test_plan_order_buy_resolves_notional_to_shares():
    m = make_market()
    w = Wallet.with_budget(1000.0)
    action = parse('{"action":"buy","outcome":1,"size":50}')
    order, reason = plan_order(m, w, action, m.current_state())
    assert reason is None
    assert order.side == "buy" and order.outcome == 1 and order.shares > 0
    # the resolved share count should cost ~the requested notional
    assert m.cost_to_buy(1, order.shares) == pytest.approx(50.0)


def test_plan_order_sell_clamps_to_position():
    m = make_market()
    w = Wallet.with_budget(1000.0)
    m.apply_trade(Order("buy", 1, 2.0))
    state = m.current_state()
    held = state.my_position[1]
    # ask to sell a notional far larger than the position is worth
    action = parse('{"action":"sell","outcome":1,"size":100000}')
    order, reason = plan_order(m, w, action, state)
    assert reason is None
    assert order.shares == pytest.approx(held)  # clamped to "sell all"


def test_plan_order_sell_with_no_position_rejected():
    m = make_market()
    w = Wallet.with_budget(1000.0)
    action = parse('{"action":"sell","outcome":0,"size":10}')
    order, reason = plan_order(m, w, action, m.current_state())
    assert order is None
    assert "no position" in reason


# ----------------------------------------------------------------- prompt / info discipline


def test_prompt_never_shows_prices_after_t():
    m = make_market()
    path = make_path()
    t = 60.0
    visible = path.prices_up_to(t)
    prompt = build_prompt(
        m.settlement_rule_text(), m.outcomes(), m.current_state(),
        visible, cash=1000.0, trade_history=[], token="ZQX",
    )
    mentioned = [float(x) for x in re.findall(r"t=(\d+(?:\.\d+)?) min", prompt)]
    assert mentioned, "prompt should list observed timestamps"
    assert max(mentioned) <= t  # CARDINAL RULE: nothing after t


def test_prompt_is_dateless_and_uses_fictional_token():
    m = make_market()
    path = make_path()
    prompt = build_prompt(
        m.settlement_rule_text(), m.outcomes(), m.current_state(),
        path.prices_up_to(30.0), cash=1000.0, trade_history=[], token="ZQX",
    )
    assert "ZQX" in prompt
    # no 4-digit calendar years the model could anchor to
    assert not re.search(r"\b(19|20)\d{2}\b", prompt)


def test_prompt_discloses_dead_window_in_relative_minutes():
    m = make_market()
    path = make_path()
    t = 60.0
    prompt = build_prompt(
        m.settlement_rule_text(), m.outcomes(), m.current_state(),
        path.prices_up_to(t), cash=1000.0, trade_history=[], token="ZQX",
        dead_window_min=90.0,
    )
    # the dead window is perceptible ...
    assert "90 min after trading closes" in prompt
    # ... but discipline still holds: no price timestamp after t, no calendar year
    mentioned = [float(x) for x in re.findall(r"t=(\d+(?:\.\d+)?) min", prompt)]
    assert max(mentioned) <= t
    assert not re.search(r"\b(19|20)\d{2}\b", prompt)


def test_prompt_hides_dead_window_when_not_disclosed():
    m = make_market()
    path = make_path()
    disclosed = build_prompt(
        m.settlement_rule_text(), m.outcomes(), m.current_state(),
        path.prices_up_to(30.0), cash=1000.0, trade_history=[], token="ZQX",
        dead_window_min=90.0,
    )
    hidden = build_prompt(
        m.settlement_rule_text(), m.outcomes(), m.current_state(),
        path.prices_up_to(30.0), cash=1000.0, trade_history=[], token="ZQX",
        dead_window_min=None,
    )
    assert "after trading closes" in disclosed
    assert "after trading closes" not in hidden


def test_prompt_zero_dead_window_says_at_close():
    m = make_market()
    path = make_path()
    prompt = build_prompt(
        m.settlement_rule_text(), m.outcomes(), m.current_state(),
        path.prices_up_to(30.0), cash=1000.0, trade_history=[], token="ZQX",
        dead_window_min=0.0,
    )
    assert "at the moment trading closes" in prompt


def test_run_episode_discloses_dead_window_by_default():
    m = make_market(trading_close_min=120.0)
    path = make_path()  # post_close_min=60 -> 60-min dead window
    log = run_episode(m, HoldModel(), path, starting_cash=1000.0)
    assert "60 min after trading closes" in log.turns[0].prompt
    # and the blind condition omits it
    m2 = make_market(trading_close_min=120.0)
    log2 = run_episode(
        m2, HoldModel(), make_path(), starting_cash=1000.0, disclose_dead_window=False
    )
    assert "after trading closes" not in log2.turns[0].prompt


# ----------------------------------------------------------------- end to end


def test_run_episode_with_hold_model_does_nothing():
    m = make_market()
    path = make_path()
    log = run_episode(m, HoldModel(), path, starting_cash=1000.0)
    assert log.footprint.trade_count == 0
    assert log.footprint.n_holds == len(path.trading_points())
    assert log.final_cash == pytest.approx(1000.0)  # never traded -> budget intact
    assert log.pnl == pytest.approx(0.0)


def test_run_episode_executes_scripted_trades_and_logs_io():
    m = make_market()
    path = make_path()  # 5 trading points (t=0,30,60,90,120)
    model = ScriptedModel(
        [
            {"action": "buy", "outcome": 2, "size": 100, "rationale": "bull"},
            {"action": "hold"},
            {"action": "sell", "outcome": 2, "size": 30, "rationale": "trim"},
        ]
    )
    log = run_episode(m, model, path, starting_cash=1000.0)
    assert log.footprint.n_buys == 1
    assert log.footprint.n_sells == 1
    assert log.footprint.trade_count == 2
    assert log.footprint.outcomes_traded == 1
    assert log.footprint.first_trade_t == 0.0
    # full input->output pair retained for every turn
    assert len(log.turns) == len(path.trading_points())
    for turn in log.turns:
        assert turn.prompt and turn.response
        assert "parse_ok" in turn.parsed


def test_run_episode_budget_constrains_trades():
    m = make_market()
    path = make_path()
    # Try to spend 10000 with only 100 budget -> rejected, no fill.
    model = ScriptedModel([{"action": "buy", "outcome": 1, "size": 10000}])
    log = run_episode(m, model, path, starting_cash=100.0)
    assert log.footprint.trade_count == 0
    assert log.footprint.n_rejected == 1
    assert log.final_cash == pytest.approx(100.0)


def test_run_episode_settles_against_reference_price():
    m = make_market()
    path = make_path()
    log = run_episode(m, RandomTrader(n_outcomes=3, seed=1), path, starting_cash=1000.0)
    # winning outcome must match the band the reference price lands in
    assert log.winning_outcome == m.bucket_for_price(path.settlement_price())
    assert log.payout >= 0.0


def test_jsonl_roundtrip(tmp_path):
    m = make_market()
    path = make_path()
    model = ScriptedModel([{"action": "buy", "outcome": 2, "size": 50, "rationale": "x"}])
    log = run_episode(m, model, path)
    out = tmp_path / "episode.jsonl"
    log.to_jsonl(str(out))
    lines = out.read_text().strip().splitlines()
    assert len(lines) == len(path.trading_points()) + 1  # turns + settlement
    records = [json.loads(ln) for ln in lines]
    assert records[0]["type"] == "turn"
    assert "prompt" in records[0] and "response" in records[0]
    assert records[-1]["type"] == "settlement"
    assert "footprint" in records[-1]


def test_parse_failures_counted_and_safe():
    m = make_market()
    path = make_path()
    model = ScriptedModel(["not json at all", "still not json"])
    log = run_episode(m, model, path)
    assert log.footprint.n_parse_failures >= 2
    assert log.footprint.trade_count == 0
