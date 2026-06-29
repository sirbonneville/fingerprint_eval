"""Synthetic ("noise") trader flow -- the SINGLE behavioral difference between
the v1 battery (sole-trader control) and the v2 battery (live-board treatment).

Why this exists
---------------
In v1 the model is the only trader, so the implied-probability board moves *only*
when the model itself trades and is frozen between its turns. A real Delphi market
has other participants whose flow moves the odds and is itself information. This
module adds that flow so v2 measures behavior against a live board, holding
everything else byte-identical to v1.

Design guarantees (these are what make the v1-vs-v2 comparison clean)
--------------------------------------------------------------------
* **Same DPM path, no parallel math.** Synthetic traders place orders through the
  exact same :meth:`PriceBucketMarket.apply_trade` the model uses. They trade under
  a distinct address (:data:`NOISE_ADDRESS`) so their flow moves the shared share
  pool ``q`` -- and therefore the implied-probability board everyone sees -- WITHOUT
  being counted as the model's own position (``my_position`` reads the agent
  address only).
* **Informed by the observable, never by the answer.** They read only the revealed
  price path (the same points the model can see) and never the settlement outcome /
  winning band. The target band is "which band is the price in now", nudged by a
  small momentum tilt -- not trend extrapolation (the walk is centered, so
  extrapolating would be systematically wrong and one-directionally exploitable).
* **Per-episode contrarian fraction.** Each trade goes *with* the inferred band with
  probability ``1 - c`` and *against* it with probability ``c``, where ``c`` is drawn
  ONCE per episode from ~5-30%. A fixed contrarian rate would itself be a learnable
  regularity; drawing it per episode keeps the board a market, not a price mirror.
* **Three load-bearing caps** prevent board saturation (see :class:`NoiseConfig`):
  per-trade ($150-350, one conviction unit), per-interval (count + a notional cap
  that keeps flow off the steep upper part of the cost curve), and per-episode
  (~$1000, ~the model's own budget, so the model stays a co-equal participant
  rather than a price-taker).
* **Seed discipline.** The flow is seeded from the SAME per-episode seed as the
  price path. Because the path is generated up front (its own RNG, fully consumed
  before the episode loop), a separate noise RNG cannot perturb it: for any seed,
  v1 and v2 share an IDENTICAL price path and winning band. The only difference is
  whether the board is alive.

Disclosure to the model is IMPLICIT only: the moved board shows up through the
existing implied-probability line in the prompt. No narration is added, so the
prompt template stays byte-identical to v1 -- only the probability numbers differ.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from .market import SIDE_BUY, Order

# Synthetic traders trade under this address so apply_trade moves the shared pool
# (the board) without crediting the model's own (DEFAULT_TRADER="agent") position.
NOISE_ADDRESS = "noise"


@dataclass(frozen=True)
class NoiseConfig:
    """Calibrated synthetic-flow knobs (see module docstring for rationale).

    Defaults are calibrated against the real market depth (k=10, seed
    q0=[100,100,100], seed pool ~1732 tokens, budget/pool ~57.7%; a ~$200-333 buy
    moves one band ~13-20 points; the curve caps a single band near ~73% within
    budget). The calibration gate (one episode, four checks) is what confirms these
    land at ~10-25 net points/interval without pinning the ceiling.
    """

    # Per episode: total synthetic notional, ~the model's own budget, so the model
    # remains a co-equal participant (the ~57.7% regime), not a price-taker. This is
    # PACED evenly across the episode's intervals (target = episode_budget/n_turns)
    # so every interval -- including the model's last, highest-stakes turns -- sees
    # ~one other trader's worth of flow instead of the budget front-loading and the
    # board freezing late.
    episode_budget: float = 1000.0
    # Per-interval spend is drawn as this fraction-range x the paced target, so the
    # board move jitters turn-to-turn (a fixed amount would itself be a regularity).
    interval_fraction_min: float = 0.6
    interval_fraction_max: float = 1.4
    # Hard notional cap per interval -- keeps a single interval off the steep upper
    # part of the cost curve (no stampede; ~one other trader's worth per interval).
    per_interval_cap: float = 380.0

    # Per synthetic trade: hard cap (one conviction unit, matching the model's own
    # scale) and a dust floor below which we don't bother placing.
    trade_max: float = 350.0
    min_trade: float = 50.0
    # Probability an interval's flow is split into TWO trades rather than one
    # (occasional bursts; the interval notional is unchanged either way).
    two_trade_prob: float = 0.30

    # Contrarian fraction range; drawn ONCE per episode.
    contrarian_min: float = 0.05
    contrarian_max: float = 0.30

    # Momentum tilt: cap on P(nudge the target band by +/-1 toward the move
    # direction). Scaled by |price move| / band_width so a big intra-interval move
    # tilts more, a tiny one barely at all. Capped well below 1 so the target stays
    # "mostly which band the price is in now".
    momentum_tilt_cap: float = 0.40


@dataclass
class IntervalReport:
    """What one interval of synthetic flow did (consumed by the calibration gate)."""

    notional: float                 # tokens of synthetic flow placed this interval
    n_trades: int                   # synthetic trades actually filled this interval
    inferred_band: int              # the observable-informed target band
    prob_before: List[float] = field(default_factory=list)
    prob_after: List[float] = field(default_factory=list)

    @property
    def net_point_move(self) -> float:
        """Net board movement in percentage points.

        Implied probabilities sum to 1, so the total absolute change is double the
        net shift; halving it (an L1/2 norm) makes a clean one-band shift of +x on
        the favored band / -x spread across the rest read as ~x points -- the
        "conviction unit" the calibration gate targets at ~10-25.
        """
        if not self.prob_before or not self.prob_after:
            return 0.0
        l1 = sum(abs(a - b) for a, b in zip(self.prob_after, self.prob_before))
        return 100.0 * l1 / 2.0


class SyntheticFlow:
    """Per-episode synthetic trader flow. Construct ONE per episode, seeded from the
    episode seed, then call it once per turn (after the clock advances, before the
    model trades) to move the board by ~one other trader's worth.

    Stateful across the episode on purpose: the contrarian fraction is drawn once
    here, and the running synthetic spend is tracked so the per-episode cap binds.
    """

    def __init__(
        self,
        seed: Optional[int],
        n_outcomes: int,
        band_width: float,
        config: NoiseConfig = NoiseConfig(),
        n_turns: int = 5,
    ):
        # Same per-episode SEED as the price path (the path uses its own Random and
        # is already fully generated, so this independent stream cannot change it).
        self.rng = random.Random(seed)
        self.n_outcomes = int(n_outcomes)
        self.band_width = float(band_width)
        self.config = config
        self.n_turns = max(1, int(n_turns))
        self.intervals_done = 0  # for ADAPTIVE pacing (remaining budget / turns left)
        # Contrarian fraction: drawn ONCE per episode.
        self.contrarian = self.rng.uniform(config.contrarian_min, config.contrarian_max)
        self.spent = 0.0  # cumulative synthetic notional this episode (cap state)

    # The spec's named entry point; the instance is callable for convenience.
    def __call__(self, market, revealed_prices) -> IntervalReport:
        return apply_noise_flow(market, revealed_prices, self.rng, state=self)

    def _inferred_band(self, market, revealed_prices) -> int:
        """Observable-informed target band: which band the price is in NOW, nudged
        by a small, magnitude-scaled momentum tilt. Reads only revealed prices."""
        price_now = revealed_prices[-1][1]
        band = market.resolve_outcome(price_now)
        if len(revealed_prices) >= 2 and self.band_width > 0.0:
            delta = price_now - revealed_prices[-2][1]
            tilt = min(self.config.momentum_tilt_cap, abs(delta) / self.band_width)
            if self.rng.random() < tilt:
                band += 1 if delta > 0.0 else -1
        return max(0, min(self.n_outcomes - 1, band))

    def _other_band(self, inferred: int) -> int:
        """A band other than the inferred one (the contrarian target)."""
        if self.n_outcomes <= 1:
            return inferred
        choice = self.rng.randrange(self.n_outcomes - 1)
        return choice if choice < inferred else choice + 1


def apply_noise_flow(market, revealed_prices, episode_rng, *, state: SyntheticFlow) -> IntervalReport:
    """Apply one inter-turn interval of synthetic flow to ``market``.

    Parameters
    ----------
    market : the live :class:`PriceBucketMarket` (mutated in place via the same
        ``apply_trade`` the model uses, under :data:`NOISE_ADDRESS`).
    revealed_prices : ``[(t, price)]`` the model can see so far -- the ONLY signal
        the synthetic traders read (never the settlement outcome).
    episode_rng : the per-episode RNG (``state.rng``); passed explicitly to match
        the documented ``apply_noise_flow(market, revealed_prices, episode_rng)``
        signature. ``state`` carries the per-episode contrarian draw + spend cap.
    """
    cfg = state.config
    prob_before = list(market.current_state().probabilities)
    if not revealed_prices:
        return IntervalReport(0.0, 0, 0, prob_before, prob_before)

    inferred = state._inferred_band(market, revealed_prices)

    # This interval's notional: ADAPTIVE paced target (remaining budget spread over
    # the turns that remain, so early jitter can't starve the late, highest-stakes
    # turns), jittered by a fraction draw, then bounded by the per-interval cap and
    # whatever episode budget is actually left.
    turns_left = max(1, state.n_turns - state.intervals_done)
    target = (cfg.episode_budget - state.spent) / turns_left
    interval_spend = target * episode_rng.uniform(cfg.interval_fraction_min, cfg.interval_fraction_max)
    interval_spend = min(interval_spend, cfg.per_interval_cap, cfg.episode_budget - state.spent)
    state.intervals_done += 1
    if interval_spend < cfg.min_trade:
        return IntervalReport(0.0, 0, inferred, prob_before, prob_before)

    # Realize the interval as one trade, or occasionally a two-trade burst (same
    # total notional). Each leg is capped at one conviction unit (trade_max).
    if interval_spend >= 2 * cfg.min_trade and episode_rng.random() < cfg.two_trade_prob:
        first = interval_spend * episode_rng.uniform(0.4, 0.6)
        legs = [first, interval_spend - first]
    else:
        legs = [interval_spend]

    placed = 0.0
    n_trades = 0
    for size in legs:
        size = min(size, cfg.trade_max)
        if size < cfg.min_trade:
            continue
        # With prob c trade AGAINST the inferred band, else WITH it.
        band = state._other_band(inferred) if episode_rng.random() < state.contrarian else inferred
        shares = market.shares_for_spend(band, size)
        if shares <= 0.0:
            continue
        result = market.apply_trade(Order(side=SIDE_BUY, outcome=band, shares=shares), address=NOISE_ADDRESS)
        if not result.ok:
            continue
        placed += result.cost
        n_trades += 1

    state.spent += placed
    prob_after = list(market.current_state().probabilities)
    return IntervalReport(placed, n_trades, inferred, prob_before, prob_after)
