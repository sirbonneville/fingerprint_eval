"""Price generator -- the intervention instrument (spec.md piece 3).

A parametric synthetic price path: a seeded random walk with explicit
**volatility** and **drift** knobs (experimental variables, not constants). The
whole path is generated up front, creation -> settlement reference, and crucially
*includes the post-trading-close "dead window"* -- the segment the trader can
never act in. How much the price moves there is how you deliberately set outcome
**knowability**.

Band-relative volatility (the load-bearing design decision)
-----------------------------------------------------------
The headline experiment measures whether implied probability discovers the true
outcome, and outcomes are price *bands*. If volatility were expressed in absolute
price units, how much the price moves *relative to the bands* would silently
depend on the anchor price (and, for a multiplicative process, on price level) --
tangling the volatility axis with anchor and band width. That is a confound baked
into the generator.

So volatility is expressed in **band-width units**: ``volatility`` = expected
number of band-widths the price travels per sqrt-hour. With the default additive
process this makes the number of band-crossings a function of ``volatility``
*alone* -- independent of ``anchor`` and ``band_width`` -- keeping the volatility
axis orthogonal to the band-width axis. That orthogonality is the whole point of
a controlled instrument.

- Default model ``"arithmetic"``: additive, constant-dollar increments
  ``sigma_$ = volatility * band_width``. Band-crossing dynamics are anchor- and
  band-width-invariant. This is the recommended, clean choice.
- ``"gbm"``: multiplicative; only anchor-normalized band-relative vol is
  supported (``sigma_log = volatility * band_width / anchor``), which is clean
  *only if the anchor is held fixed across all cells*. Provided for realism; use
  with that caveat.

If ``band_width`` is omitted, ``volatility``/``drift`` fall back to absolute price
units (legacy/escape hatch); prefer the band-relative form for experiments.

Determinism: pass ``seed`` and a path is exactly reproducible (clean experiments
and the eventual REE replay). Built on the stdlib Mersenne Twister, no third-party
dependency.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

Point = Tuple[float, float]  # (t_minutes, price)

MODEL_GBM = "gbm"
MODEL_ARITHMETIC = "arithmetic"


@dataclass
class PricePath:
    """A generated price path with a trading window and a post-close dead window.

    ``points`` is the full ``[(t_min, price)]`` series from t=0 to the settlement
    reference. ``trading_close_min`` marks the boundary: the harness lets the
    model act only at points with ``t <= trading_close_min``; everything after is
    the dead window. The settlement reference price is the final point.
    """

    points: List[Point]
    interval_min: float
    trading_close_min: float
    anchor: float
    volatility: float
    drift: float
    model: str = MODEL_ARITHMETIC
    band_width: Optional[float] = None
    seed: Optional[int] = None

    @property
    def times(self) -> List[float]:
        return [t for t, _ in self.points]

    @property
    def prices(self) -> List[float]:
        return [p for _, p in self.points]

    @property
    def band_relative(self) -> bool:
        return self.band_width is not None

    def trading_points(self) -> List[Point]:
        """Points the model is allowed to act on (t <= trading_close_min)."""
        return [(t, p) for t, p in self.points if t <= self.trading_close_min + 1e-9]

    def post_close_points(self) -> List[Point]:
        """The dead window: points strictly after trading close."""
        return [(t, p) for t, p in self.points if t > self.trading_close_min + 1e-9]

    def prices_up_to(self, t_min: float) -> List[Point]:
        """CARDINAL RULE helper: only points at or before ``t_min`` (strict ``<=``)."""
        return [(t, p) for t, p in self.points if t <= t_min + 1e-9]

    def settlement_price(self) -> float:
        """The reference price the market settles against (end of the path)."""
        return self.points[-1][1]

    def price_at_close(self) -> float:
        """Price at the moment trading closes (last point the trader could see)."""
        for t, p in reversed(self.points):
            if t <= self.trading_close_min + 1e-9:
                return p
        return self.points[0][1]

    def knowability_window_min(self) -> float:
        """Length of the dead window in minutes (0 => settlement is fully knowable)."""
        return self.points[-1][0] - self.trading_close_min


def generate_path(
    anchor: float,
    hours: float,
    interval_min: float,
    volatility: float,
    drift: float,
    band_width: Optional[float] = None,
    post_close_min: float = 0.0,
    seed: Optional[int] = None,
    model: str = MODEL_ARITHMETIC,
) -> PricePath:
    """Generate a synthetic price path (see module docstring).

    Parameters
    ----------
    anchor : starting price at t=0 (must be > 0 for the gbm model).
    hours : length of the **trading** window in hours.
    interval_min : timestep in minutes.
    volatility : volatility knob. In **band-width units per sqrt-hour** when
        ``band_width`` is given (recommended); otherwise absolute price units.
    drift : drift knob, in band-width units per hour (or absolute if no band).
    band_width : dollar width of one outcome band. Providing it enables the
        band-relative interpretation that keeps volatility orthogonal to anchor
        and band width.
    post_close_min : length of the post-close dead window in minutes -- the
        **knowability** knob (0 => settlement reference is the price at close).
    seed : RNG seed for reproducibility (``None`` => nondeterministic).
    model : ``"arithmetic"`` (default) or ``"gbm"``.
    """
    if hours <= 0.0:
        raise ValueError("hours must be positive")
    if interval_min <= 0.0:
        raise ValueError("interval_min must be positive")
    if post_close_min < 0.0:
        raise ValueError("post_close_min must be non-negative")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    if band_width is not None and band_width <= 0.0:
        raise ValueError("band_width must be positive when provided")
    if model not in (MODEL_GBM, MODEL_ARITHMETIC):
        raise ValueError("model must be %r or %r" % (MODEL_GBM, MODEL_ARITHMETIC))
    if model == MODEL_GBM and anchor <= 0.0:
        raise ValueError("anchor must be positive for the gbm model")

    n_trading = int(round((hours * 60.0) / interval_min))
    n_post = int(round(post_close_min / interval_min))
    if n_trading < 1:
        raise ValueError("trading window is shorter than one interval")
    total_steps = n_trading + n_post
    trading_close_min = n_trading * interval_min

    # Resolve the volatility/drift knobs to absolute price units.
    if band_width is not None:
        vol_abs = volatility * band_width        # dollars per sqrt-hour
        drift_abs = drift * band_width           # dollars per hour
    else:
        vol_abs = volatility
        drift_abs = drift

    rng = random.Random(seed)
    dt = interval_min / 60.0
    sqrt_dt = math.sqrt(dt)

    if model == MODEL_GBM:
        # Anchor-normalized so a unit move near the anchor matches the additive
        # case; clean only if the anchor is held fixed across cells.
        sigma_log = vol_abs / anchor
        mu_log = drift_abs / anchor
    else:
        sigma_log = mu_log = 0.0  # unused

    price = float(anchor)
    points: List[Point] = [(0.0, price)]
    for i in range(1, total_steps + 1):
        z = rng.gauss(0.0, 1.0)
        if model == MODEL_GBM:
            price = price * math.exp(mu_log * dt + sigma_log * sqrt_dt * z)
        else:
            price = price + drift_abs * dt + vol_abs * sqrt_dt * z
        points.append((i * interval_min, price))

    return PricePath(
        points=points,
        interval_min=interval_min,
        trading_close_min=trading_close_min,
        anchor=float(anchor),
        volatility=volatility,
        drift=drift,
        model=model,
        band_width=band_width,
        seed=seed,
    )
