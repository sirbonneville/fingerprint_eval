"""Tests for the analysis honesty layer (spec.md piece 6).

These check that the layer foregrounds effect-vs-spread + overlap, gates on
small N, calls overlapping distributions 'no effect', refuses to over-read, and
keeps the plumbing-only caveat. They validate that the layer is HONEST, not that
it finds anything (a stand-in finds nothing by construction).
"""

import pytest

from agent_eval_harness.analysis import (
    PLUMBING_BANNER,
    V_NO_EFFECT,
    V_NO_VARIATION,
    V_POSSIBLE,
    V_TOO_THIN,
    analyze_axis_metric,
    cles,
    format_report,
    mann_whitney,
    overlap_plot,
    pairwise_effect,
    range_overlap_fraction,
)
from agent_eval_harness.experiment import CellConfig, default_stand_in_factory, run_grid


# ----------------------------------------------------------------- effect-size primitives


def test_cles_identical_is_half():
    assert cles([1, 2, 3], [1, 2, 3]) == pytest.approx(0.5)


def test_cles_full_separation():
    assert cles([1, 2, 3], [10, 11, 12]) == pytest.approx(1.0)
    assert cles([10, 11, 12], [1, 2, 3]) == pytest.approx(0.0)


def test_range_overlap_fraction():
    assert range_overlap_fraction([0, 1, 2], [10, 11, 12]) == pytest.approx(0.0)
    assert range_overlap_fraction([0, 5, 10], [0, 5, 10]) == pytest.approx(1.0)


# ----------------------------------------------------------------- pairwise verdicts


def test_overlapping_distributions_are_no_detectable_effect():
    a = [5, 6, 7, 6, 5, 7, 6, 5, 6, 7]
    b = [6, 5, 7, 6, 7, 5, 6, 7, 5, 6]  # same-ish, heavy overlap
    e = pairwise_effect(a, b, "trade_count", "low", "high")
    assert e.verdict == V_NO_EFFECT
    assert e.effect_over_spread < 1.0
    assert any("Spread exceeds delta" in n for n in e.notes)


def test_cleanly_separated_distributions_are_possible_effect():
    a = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    b = [20, 21, 20, 21, 20, 21, 20, 21, 20, 21]
    e = pairwise_effect(a, b, "trade_count", "low", "high")
    assert e.verdict == V_POSSIBLE
    assert e.effect_over_spread >= 1.0
    assert e.cles == pytest.approx(1.0)


def test_small_n_is_gated_as_too_thin():
    e = pairwise_effect([1, 2], [10, 20], "trade_count", "low", "high")
    assert e.verdict == V_TOO_THIN
    assert any("N too small" in n for n in e.notes)


def test_zero_variation_reported_not_hidden():
    a = [3.0] * 10
    b = [3.0] * 10
    e = pairwise_effect(a, b, "trade_count", "low", "high")
    assert e.verdict == V_NO_VARIATION
    assert e.delta_median == 0.0


def test_plumbing_caveat_present_unless_trusted():
    untrusted = pairwise_effect([1, 2, 3, 4], [1, 2, 3, 4], "x", "a", "b", trusted=False)
    assert any(PLUMBING_BANNER == n for n in untrusted.notes)
    trusted = pairwise_effect([1, 2, 3, 4], [1, 2, 3, 4], "x", "a", "b", trusted=True)
    assert not any(PLUMBING_BANNER == n for n in trusted.notes)


def test_below_recommended_n_flagged_provisional():
    e = pairwise_effect([1, 2, 3, 4], [10, 11, 12, 13], "x", "a", "b", trusted=True)
    assert any("provisional" in n for n in e.notes)


# ----------------------------------------------------------------- overlap plot


def test_overlap_plot_contains_both_labels_and_axis():
    plot = overlap_plot([1, 2, 3], [2, 3, 4], "low", "high")
    assert "low" in plot and "high" in plot
    assert "\n" in plot  # multi-row


def test_overlap_plot_handles_no_variation():
    plot = overlap_plot([5, 5, 5], [5, 5, 5], "a", "b")
    assert "identical" in plot


# ----------------------------------------------------------------- significance is secondary + caveated


def test_mann_whitney_is_caveated_underpowered():
    mw = mann_whitney([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert "UNDERPOWERED" in mw.caveat
    assert 0.0 <= mw.p_two_sided_approx <= 1.0


# ----------------------------------------------------------------- grid integration


def test_analyze_axis_metric_on_standin_grid():
    base = CellConfig(volatility=0.5, knowability_min=0.0, band_width=10.0, n_outcomes=3)
    grid = run_grid(base, "volatility", [0.1, 0.5, 2.0], default_stand_in_factory, n_runs=10)
    am = analyze_axis_metric(grid, "trade_count", trusted=False)
    assert am.baseline_value == 0.1
    assert set(am.per_cell.keys()) == {0.1, 0.5, 2.0}
    assert len(am.effects) == 2  # each non-baseline cell vs baseline


def test_format_report_foregrounds_caveat_and_verdicts():
    base = CellConfig(volatility=0.5, knowability_min=0.0, band_width=10.0, n_outcomes=3)
    grid = run_grid(base, "volatility", [0.1, 2.0], default_stand_in_factory, n_runs=8)
    report = format_report(grid, metrics=["trade_count", "total_volume"], trusted=False)
    assert PLUMBING_BANNER in report
    assert "effect/spread" in report
    assert "METRIC: trade_count" in report
