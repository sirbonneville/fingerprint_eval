"""Mechanical tests for the experiment runner (spec.md piece 5).

These validate the PLUMBING: cells build correctly, axes stay frozen except the
swept one, N runs produce distributions, the control pair is shaped right, and
the calibration gate behaves. They do NOT (and cannot) validate discovery with
stand-in models -- that waits for a real model behind the Model protocol.
"""

import pytest

from agent_eval_harness.experiment import (
    CellConfig,
    control_pair,
    default_stand_in_factory,
    run_calibration,
    run_cell,
    run_experiment,
    run_grid,
    summarize,
)
from agent_eval_harness.model import HoldModel, RandomTrader


def base_cell():
    return CellConfig(volatility=0.5, knowability_min=0.0, band_width=10.0,
                      anchor=100.0, n_outcomes=3, hours=2.0, interval_min=30.0)


# ----------------------------------------------------------------- cell construction


def test_band_edges_centered_on_anchor():
    cell = CellConfig(volatility=0.5, knowability_min=0.0, band_width=10.0, anchor=100.0, n_outcomes=3)
    assert cell.band_edges() == pytest.approx([95.0, 105.0])  # middle band [95,105)


def test_band_edges_four_outcomes():
    cell = CellConfig(volatility=0.5, knowability_min=0.0, band_width=10.0, anchor=100.0, n_outcomes=4)
    assert cell.band_edges() == pytest.approx([90.0, 100.0, 110.0])


def test_build_market_and_path_are_consistent():
    cell = base_cell()
    m = cell.build_market()
    assert m.n_outcomes == 3
    assert m.uniform_band_width() == pytest.approx(10.0)
    path = cell.build_path(seed=0)
    assert path.band_width == pytest.approx(10.0)
    assert path.trading_close_min == pytest.approx(120.0)


# ----------------------------------------------------------------- distributions


def test_summarize_basic():
    d = summarize([1, 2, 3, 4, 5])
    assert d.n == 5
    assert d.median == 3
    assert d.minimum == 1 and d.maximum == 5


def test_summarize_empty_and_singleton():
    assert summarize([]).n == 0
    one = summarize([7.0])
    assert one.n == 1 and one.median == 7.0 and one.iqr == 0.0


def test_run_cell_produces_distributions_over_n_runs():
    cell = base_cell()
    result = run_cell(cell, default_stand_in_factory, n_runs=8)
    assert result.n_runs == 8
    assert len(result.records) == 8
    assert result.metrics["trade_count"].n == 8
    assert result.metrics["discovery_prob_on_winner"].n == 8


def test_run_cell_is_reproducible():
    cell = base_cell()
    a = run_cell(cell, default_stand_in_factory, n_runs=6)
    b = run_cell(cell, default_stand_in_factory, n_runs=6)
    assert a.median("pnl") == b.median("pnl")
    assert a.median("trade_count") == b.median("trade_count")


# ----------------------------------------------------------------- grid sweep


def test_run_grid_freezes_all_but_swept_axis():
    base = base_cell()
    grid = run_grid(base, "volatility", [0.1, 0.5, 2.0], default_stand_in_factory, n_runs=4)
    assert set(grid.cells.keys()) == {0.1, 0.5, 2.0}
    for v, cell_result in grid.cells.items():
        c = cell_result.cell
        assert c.volatility == v
        # everything else frozen
        assert c.knowability_min == base.knowability_min
        assert c.band_width == base.band_width
        assert c.anchor == base.anchor


def test_run_grid_rejects_unknown_axis():
    with pytest.raises(ValueError):
        run_grid(base_cell(), "not_an_axis", [1, 2], default_stand_in_factory, n_runs=2)


def test_deltas_vs_spread_report_shape():
    grid = run_grid(base_cell(), "volatility", [0.1, 1.0], default_stand_in_factory, n_runs=5)
    report = grid.deltas_vs_spread("trade_count")
    assert set(report["medians"].keys()) == {0.1, 1.0}
    assert "between_cell_range" in report
    assert "looks_like_signal" in report


# ----------------------------------------------------------------- calibration control pair


def test_control_pair_shape():
    easy, hard = control_pair(base_cell())
    assert easy.volatility < hard.volatility
    assert easy.knowability_min == 0.0      # fully knowable at close
    assert hard.knowability_min > 0.0        # long dead window -> unknowable
    assert easy.label == "control-easy" and hard.label == "control-hard"


def test_calibration_with_standin_is_plumbing_only():
    cal = run_calibration(base_cell(), default_stand_in_factory, n_runs=6, trust_discovery=False)
    assert cal.trusted is False
    assert "PLUMBING ONLY" in cal.note
    # both control cells actually ran
    assert cal.easy.n_runs == 6 and cal.hard.n_runs == 6


def test_experiment_standin_runs_grid_for_plumbing():
    res = run_experiment(
        base_cell(), "volatility", [0.1, 0.5, 2.0],
        model_factory=default_stand_in_factory, n_runs=4, trust_discovery=False,
    )
    assert res.ran_grid is True
    assert res.grid is not None
    assert "plumbing" in res.message.lower()


def test_experiment_trusted_gate_stops_when_easy_control_fails():
    # HoldModel never trades, so it cannot discover the easy control. With a
    # trusted gate, the grid must be skipped.
    factory = lambda cell, run: HoldModel()
    res = run_experiment(
        base_cell(), "volatility", [0.1, 0.5, 2.0],
        model_factory=factory, n_runs=4, trust_discovery=True, require_calibration=True,
    )
    assert res.calibration.passed is False
    assert res.ran_grid is False
    assert res.grid is None
    assert "skipped" in res.message.lower()


def test_holdmodel_discovery_is_uninformative_baseline():
    # Sanity: with no trading, closing prob on winner ~ 1/n_outcomes (no discovery).
    cell = base_cell()
    result = run_cell(cell, lambda c, r: HoldModel(), n_runs=6)
    assert result.median("discovery_prob_on_winner") == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert result.median("discovery_edge_over_open") == pytest.approx(0.0, abs=1e-6)
