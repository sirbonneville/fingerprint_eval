"""Tests for the v2 synthetic ("noise") trader flow.

These lock the guarantees that make the v1-vs-v2 comparison valid:
- v1 is byte-identical (noise off changes nothing);
- synthetic flow moves the shared board but NOT the agent's position;
- the price path and winning band are seed-aligned across the two arms;
- the three caps hold (per-episode budget, liveness/pacing);
- everything is deterministic per seed, including under parallel execution.
"""

import dataclasses

from agent_eval_harness.dpm.market import DEFAULT_TRADER
from agent_eval_harness.experiment import CellConfig, run_cell, run_one_episode
from agent_eval_harness.harness import run_episode
from agent_eval_harness.model import HoldModel
from agent_eval_harness.synthetic import (
    NOISE_ADDRESS,
    NoiseConfig,
    SyntheticFlow,
)


def base_cell():
    return CellConfig(
        volatility=1.0, knowability_min=0.0, band_width=10.0, anchor=100.0,
        n_outcomes=3, hours=2.0, interval_min=30.0, drift=0.0, k=10.0, fee=0.02,
        initial_shares=100.0, starting_cash=1000.0, temperature=0.7, memory=False,
        disclose_dead_window=True, token="ZQX",
    )


def make_flow(cell, seed, config=None):
    path = cell.build_path(seed=seed)
    return SyntheticFlow(
        seed=seed, n_outcomes=cell.n_outcomes, band_width=cell.band_width,
        config=config or NoiseConfig(), n_turns=len(path.trading_points()),
    )


# --------------------------------------------------------------- v1 parity

def test_noise_off_is_byte_identical_v1():
    """run_episode with noise_flow=None must equal not passing it (v1 untouched)."""
    cell = base_cell()
    a = run_episode(cell.build_market(), HoldModel(), cell.build_path(seed=3),
                    starting_cash=1000.0, token="ZQX", seed=3)
    b = run_episode(cell.build_market(), HoldModel(), cell.build_path(seed=3),
                    starting_cash=1000.0, token="ZQX", seed=3, noise_flow=None)
    assert [t.prompt for t in a.turns] == [t.prompt for t in b.turns]
    assert a.closing_probabilities == b.closing_probabilities
    assert a.winning_outcome == b.winning_outcome


# --------------------------------------------------------------- board moves, position doesn't

def test_synthetic_flow_moves_board_not_agent_position():
    cell = base_cell()
    market = cell.build_market()
    path = cell.build_path(seed=1)
    before = list(market.current_state().probabilities)
    log = run_episode(market, HoldModel(), path, starting_cash=1000.0, token="ZQX",
                      seed=1, noise_flow=make_flow(cell, 1))
    after = list(market.current_state().probabilities)
    # the board moved...
    assert after != before
    # ...the synthetic address accrued shares...
    assert sum(market.position(NOISE_ADDRESS)) > 0.0
    # ...but the model (HoldModel) never traded, so its position stays flat.
    assert all(abs(x) < 1e-9 for x in market.position(DEFAULT_TRADER))
    assert all(abs(x) < 1e-9 for x in log.closing_position)


# --------------------------------------------------------------- seed alignment

def test_seed_alignment_path_and_winner_match_v1():
    """For a seed, v1 and v2 share an identical price path AND winning band."""
    cell = base_cell()
    for seed in range(8):
        v1 = run_episode(cell.build_market(), HoldModel(), cell.build_path(seed=seed),
                         starting_cash=1000.0, token="ZQX", seed=seed)
        v2 = run_episode(cell.build_market(), HoldModel(), cell.build_path(seed=seed),
                         starting_cash=1000.0, token="ZQX", seed=seed,
                         noise_flow=make_flow(cell, seed))
        assert v1.settlement_price == v2.settlement_price
        assert v1.winning_outcome == v2.winning_outcome


# --------------------------------------------------------------- caps + liveness

def test_episode_budget_cap_holds():
    cell = base_cell()
    cfg = NoiseConfig()
    for seed in range(12):
        market = cell.build_market()
        path = cell.build_path(seed=seed)
        flow = make_flow(cell, seed, cfg)
        spent_reports = 0.0
        market.set_time(0.0)
        for t, _ in path.trading_points():
            market.set_time(t)
            rep = flow(market, path.prices_up_to(t))
            spent_reports += rep.notional
        # never exceeds the per-episode budget (tracked spend == summed reports).
        assert flow.spent <= cfg.episode_budget + 1e-6
        assert abs(flow.spent - spent_reports) < 1e-6


def test_flow_is_live_every_interval():
    """Adaptive pacing should place flow on every interval (no dead late board)."""
    cell = base_cell()
    market = cell.build_market()
    path = cell.build_path(seed=0)
    flow = make_flow(cell, 0)
    moved = 0
    for t, _ in path.trading_points():
        market.set_time(t)
        rep = flow(market, path.prices_up_to(t))
        if rep.notional > 0.0:
            moved += 1
    assert moved == len(path.trading_points())


def test_contrarian_drawn_once_per_episode_and_in_range():
    cfg = NoiseConfig()
    f = make_flow(base_cell(), 5, cfg)
    c1 = f.contrarian
    assert cfg.contrarian_min <= c1 <= cfg.contrarian_max
    # same seed -> same draw (reproducible).
    assert make_flow(base_cell(), 5, cfg).contrarian == c1


# --------------------------------------------------------------- determinism

def test_synthetic_episode_is_deterministic_per_seed():
    cell = base_cell()
    a = run_episode(cell.build_market(), HoldModel(), cell.build_path(seed=7),
                    starting_cash=1000.0, token="ZQX", seed=7, noise_flow=make_flow(cell, 7))
    b = run_episode(cell.build_market(), HoldModel(), cell.build_path(seed=7),
                    starting_cash=1000.0, token="ZQX", seed=7, noise_flow=make_flow(cell, 7))
    assert [t.prompt for t in a.turns] == [t.prompt for t in b.turns]
    assert a.closing_probabilities == b.closing_probabilities


def test_parallel_run_cell_matches_serial():
    """Concurrency may only change execution ORDER, never results (Step 5)."""
    cell = base_cell()
    factory = lambda c, run: HoldModel()  # deterministic model
    serial = run_cell(cell, factory, n_runs=6, synthetic_config=NoiseConfig(), max_workers=1)
    parallel = run_cell(cell, factory, n_runs=6, synthetic_config=NoiseConfig(), max_workers=4)
    for name in ("trade_count", "total_volume", "discovery_prob_on_winner", "pnl"):
        assert serial.metrics[name].as_dict() == parallel.metrics[name].as_dict()
    # records re-sorted by run index, so they line up one-to-one.
    assert [r.run for r in parallel.records] == list(range(6))
    assert [r.winning_outcome for r in serial.records] == [r.winning_outcome for r in parallel.records]


def test_v1_control_still_runs_without_synthetic_config():
    """run_cell with synthetic_config=None is the v1 control path (no noise import needed)."""
    cell = base_cell()
    factory = lambda c, run: HoldModel()
    res = run_cell(cell, factory, n_runs=3, synthetic_config=None, max_workers=1)
    assert res.n_runs == 3
    # HoldModel never trades and there is no noise, so every closing position is flat.
    for r in res.records:
        assert r.footprint.trade_count == 0
