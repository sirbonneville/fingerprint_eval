"""Temperature as a sweepable, model-side intervention (the subject's own knob)."""

from dataclasses import replace

from agent_eval_harness.experiment import (
    METRIC_NAMES,
    SWEEPABLE_AXES,
    CellConfig,
    default_stand_in_factory,
    run_grid,
)
from agent_eval_harness.model import HoldModel, openrouter_factory


def _base_cell(**kw) -> CellConfig:
    return CellConfig(volatility=0.5, knowability_min=0.0, band_width=10.0, **kw)


def test_temperature_is_sweepable_and_a_cell_field():
    assert "temperature" in SWEEPABLE_AXES
    cell = _base_cell(temperature=0.3)
    assert cell.temperature == 0.3
    assert replace(cell, temperature=1.1).temperature == 1.1


def test_factory_reads_cell_temperature():
    # The factory must take temperature FROM the cell so the sweep varies the model.
    factory = openrouter_factory("x/y", transport=lambda payload: {})
    hot = factory(_base_cell(temperature=1.3), 0)
    cold = factory(_base_cell(temperature=0.0), 0)
    assert hot.temperature == 1.3
    assert cold.temperature == 0.0


def test_temperature_passed_into_request_payload():
    seen = {}

    def transport(payload):
        seen.update(payload)
        return {"choices": [{"message": {"content": '{"action":"hold"}'}}]}

    factory = openrouter_factory("x/y", transport=transport)
    model = factory(_base_cell(temperature=0.9), 0)
    model("prompt")
    assert seen["temperature"] == 0.9


def test_environment_is_held_fixed_across_temperature_cells():
    # Temperature must not touch the price path: identical seed -> identical path,
    # so any behavior delta is attributable to temperature alone.
    base = _base_cell(temperature=0.2)
    p_lo = base.build_path(seed=0).prices
    p_hi = replace(base, temperature=1.5).build_path(seed=0).prices
    assert p_lo == p_hi


def test_run_grid_over_temperature_runs_mechanically():
    base = _base_cell()
    grid = run_grid(
        base, "temperature", [0.0, 0.7, 1.2],
        lambda c, r: HoldModel(), n_runs=3, store_logs=False,
    )
    assert set(grid.cells.keys()) == {0.0, 0.7, 1.2}
    for cr in grid.cells.values():
        assert cr.n_runs == 3


def test_hold_rate_metric_exists_and_reads():
    assert "hold_rate" in METRIC_NAMES
    base = _base_cell()
    grid = run_grid(base, "temperature", [0.0, 1.0],
                    lambda c, r: HoldModel(), n_runs=2)
    # HoldModel holds every turn -> hold_rate == 1.0 for every run.
    vals = grid.cells[0.0].metric_values("hold_rate")
    assert vals and all(v == 1.0 for v in vals)
