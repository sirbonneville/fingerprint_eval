"""Tests for the price generator (spec.md piece 3).

The path is an intervention instrument, so the tests focus on what must hold for
clean experiments: determinism (seeding), the vol/drift knobs behaving as knobs,
the post-close dead window that sets knowability, and -- the load-bearing one --
that band-relative volatility keeps the volatility axis orthogonal to both anchor
price and band width.
"""

import math
import statistics

import pytest

from fingerprint_eval.price_path import (
    MODEL_ARITHMETIC,
    MODEL_GBM,
    generate_path,
)


# ----------------------------------------------------------------- determinism


def test_same_seed_is_identical():
    a = generate_path(100.0, hours=2, interval_min=15, volatility=0.1, drift=0.0, seed=7)
    b = generate_path(100.0, hours=2, interval_min=15, volatility=0.1, drift=0.0, seed=7)
    assert a.points == b.points


def test_different_seed_differs():
    a = generate_path(100.0, hours=2, interval_min=15, volatility=0.1, drift=0.0, seed=1)
    b = generate_path(100.0, hours=2, interval_min=15, volatility=0.1, drift=0.0, seed=2)
    assert a.prices != b.prices


# ----------------------------------------------------------------- structure / timestamps


def test_anchor_is_first_point():
    path = generate_path(50.0, hours=1, interval_min=10, volatility=0.05, drift=0.01, seed=0)
    assert path.points[0] == (0.0, 50.0)


def test_timestamps_and_trading_close_grid():
    path = generate_path(
        100.0, hours=2, interval_min=30, volatility=0.0, drift=0.0,
        post_close_min=60, seed=0,
    )
    assert path.trading_close_min == pytest.approx(120.0)
    assert path.times == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    assert len(path.trading_points()) == 5      # t = 0..120
    assert len(path.post_close_points()) == 2   # t = 150, 180
    assert path.knowability_window_min() == pytest.approx(60.0)


def test_no_dead_window_means_settlement_at_close():
    path = generate_path(
        100.0, hours=1, interval_min=20, volatility=0.2, drift=0.0,
        post_close_min=0, seed=3,
    )
    assert path.knowability_window_min() == pytest.approx(0.0)
    assert path.settlement_price() == pytest.approx(path.price_at_close())
    assert path.post_close_points() == []


# ----------------------------------------------------------------- cardinal info rule


def test_prices_up_to_never_leaks_the_future():
    path = generate_path(
        100.0, hours=2, interval_min=30, volatility=0.1, drift=0.0,
        post_close_min=60, seed=5,
    )
    visible = path.prices_up_to(60.0)
    assert [t for t, _ in visible] == [0.0, 30.0, 60.0]
    assert max(t for t, _ in visible) <= 60.0


# ----------------------------------------------------------------- vol / drift knobs (default additive)


def test_zero_vol_zero_drift_is_flat():
    path = generate_path(100.0, hours=2, interval_min=15, volatility=0.0, drift=0.0, seed=9)
    assert all(p == pytest.approx(100.0) for p in path.prices)


def test_zero_vol_drift_is_linear_for_additive_model():
    # Additive with no noise: price_i = anchor + drift * dt * i (absolute units).
    anchor, interval_min, drift = 100.0, 60.0, 2.0
    path = generate_path(anchor, hours=3, interval_min=interval_min, volatility=0.0, drift=drift, seed=0)
    dt = interval_min / 60.0
    for i, (_, price) in enumerate(path.points):
        assert price == pytest.approx(anchor + drift * dt * i)


def test_gbm_zero_vol_drift_compounds():
    anchor, interval_min, drift = 100.0, 60.0, 0.05
    path = generate_path(anchor, hours=3, interval_min=interval_min, volatility=0.0,
                         drift=drift, seed=0, model=MODEL_GBM)
    dt = interval_min / 60.0
    sigma_log = 0.0
    mu_log = drift / anchor
    for i, (_, price) in enumerate(path.points):
        assert price == pytest.approx(anchor * math.exp(mu_log * dt * i))


def test_higher_volatility_widens_endpoint_spread():
    def endpoint_std(vol):
        ends = [
            generate_path(100.0, hours=4, interval_min=30, volatility=vol, drift=0.0, seed=s)
            .settlement_price()
            for s in range(400)
        ]
        return statistics.pstdev(ends)

    assert endpoint_std(3.0) > endpoint_std(0.5) > endpoint_std(0.0)


def test_additive_endpoint_std_matches_volatility_times_sqrt_time():
    # Absolute additive: std(end - anchor) ~ volatility * sqrt(T_hours).
    vol, hours = 2.0, 4.0
    ends = [
        generate_path(100.0, hours=hours, interval_min=15, volatility=vol, drift=0.0, seed=s)
        .settlement_price()
        for s in range(3000)
    ]
    measured = statistics.pstdev([e - 100.0 for e in ends])
    assert measured == pytest.approx(vol * math.sqrt(hours), rel=0.1)


def test_interval_does_not_change_horizon_variance():
    def endpoint_std(interval):
        ends = [
            generate_path(100.0, hours=4, interval_min=interval, volatility=2.0, drift=0.0, seed=s)
            .settlement_price()
            for s in range(800)
        ]
        return statistics.pstdev(ends)

    assert endpoint_std(60) == pytest.approx(endpoint_std(15), rel=0.15)


# ----------------------------------------------------------------- band-relative orthogonality (the point)


def _band_travel_std(band_width, anchor, vol, hours, n=1500):
    """Std of price travel measured in band-widths over the horizon."""
    travels = [
        (
            generate_path(
                anchor, hours=hours, interval_min=15, volatility=vol, drift=0.0,
                band_width=band_width, seed=s,
            ).settlement_price()
            - anchor
        )
        / band_width
        for s in range(n)
    ]
    return statistics.pstdev(travels)


def test_band_relative_travel_independent_of_band_width():
    # Same vol, different band widths -> same number of band-widths travelled.
    vol, hours = 0.5, 4.0
    narrow = _band_travel_std(band_width=5.0, anchor=100.0, vol=vol, hours=hours)
    wide = _band_travel_std(band_width=50.0, anchor=100.0, vol=vol, hours=hours)
    assert narrow == pytest.approx(wide, rel=0.05)


def test_band_relative_travel_independent_of_anchor():
    # Same vol + band width, very different anchors -> same band-width travel.
    vol, hours, bw = 0.5, 4.0, 10.0
    low = _band_travel_std(band_width=bw, anchor=50.0, vol=vol, hours=hours)
    high = _band_travel_std(band_width=bw, anchor=5000.0, vol=vol, hours=hours)
    assert low == pytest.approx(high, rel=0.05)


def test_band_relative_travel_equals_vol_times_sqrt_time():
    # The clean meaning: vol band-widths/sqrt-hour over T hours => vol*sqrt(T) band-widths.
    vol, hours = 0.5, 4.0
    measured = _band_travel_std(band_width=10.0, anchor=100.0, vol=vol, hours=hours)
    assert measured == pytest.approx(vol * math.sqrt(hours), rel=0.08)


# ----------------------------------------------------------------- models / validation


def test_gbm_stays_positive():
    path = generate_path(10.0, hours=8, interval_min=5, volatility=0.5, drift=-0.2,
                         band_width=1.0, seed=11, model=MODEL_GBM)
    assert all(p > 0.0 for p in path.prices)


def test_default_model_is_arithmetic():
    path = generate_path(100.0, hours=1, interval_min=30, volatility=0.1, drift=0.0, seed=4)
    assert path.model == MODEL_ARITHMETIC


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hours": 0, "interval_min": 10, "volatility": 0.1, "drift": 0.0},
        {"hours": 1, "interval_min": 0, "volatility": 0.1, "drift": 0.0},
        {"hours": 1, "interval_min": 10, "volatility": -0.1, "drift": 0.0},
        {"hours": 1, "interval_min": 10, "volatility": 0.1, "drift": 0.0, "post_close_min": -5},
        {"hours": 1, "interval_min": 10, "volatility": 0.1, "drift": 0.0, "band_width": 0.0},
    ],
)
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        generate_path(100.0, **kwargs)


def test_gbm_requires_positive_anchor():
    with pytest.raises(ValueError):
        generate_path(0.0, hours=1, interval_min=10, volatility=0.1, drift=0.0, model=MODEL_GBM)
