"""``PriceBucketMarket`` -- the one concrete DPM implementation.

Holds mutable pool state and exposes the methods ``spec.md`` piece 1 calls for
(``cost_to_buy``, ``apply_trade``, ``current_prices``, ``settle``) plus the
sell / implied-probability / liquidation helpers the verified math supports.

State is intentionally derivable and minimal:
- ``q``         : outstanding shares per outcome (drives all pricing)
- ``positions`` : per-address share holdings (drives settlement payouts)
- ``pool``      : derived collateral C(q) = k*sqrt(sum q_i**2); never tracked
                  independently so the invariant cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..market import (
    SIDE_BUY,
    SIDE_HOLD,
    SIDE_SELL,
    Market,
    MarketState,
    Order,
    OutcomeInfo,
    TradeResult,
)
from . import math as dpm_math

CREATOR_ADDRESS = "__creator__"
DEFAULT_TRADER = "agent"


@dataclass
class MarketConfig:
    """Static configuration for a price-bucket market.

    ``band_edges`` are the interior price boundaries between outcome buckets.
    For N outcomes there are N-1 edges, ascending. A settlement reference price
    is mapped to the bucket it falls into via :meth:`PriceBucketMarket.bucket_for_price`.

    ``trading_close_min`` is when trading closes, in minutes from market open;
    it drives ``time_to_close`` in :meth:`PriceBucketMarket.current_state`. Leave
    ``None`` for a market with no clock.
    """

    k: float
    fee: float = 0.02
    outcome_labels: Optional[List[str]] = None
    band_edges: Optional[List[float]] = None
    settlement_rule_text: str = ""
    trading_close_min: Optional[float] = None

    def __post_init__(self) -> None:
        if self.k <= 0.0:
            raise ValueError("k (liquidity constant) must be positive")
        if not 0.0 <= self.fee < 1.0:
            raise ValueError("fee must be in [0, 1)")


class PriceBucketMarket(Market):
    """Concrete dynamic-parimutuel price-bucket market.

    Parameters
    ----------
    config : MarketConfig
    initial_shares : per-outcome seed shares. Required and strictly positive for
        every outcome so spot prices / probabilities are well defined from t=0.
        These seed shares are credited to ``CREATOR_ADDRESS`` so settlement math
        (split the whole pool pro-rata across *all* winning shares) stays
        faithful to the on-chain behavior.
    """

    def __init__(self, config: MarketConfig, initial_shares: Sequence[float]):
        if len(initial_shares) < 2:
            raise ValueError("a market needs at least 2 outcomes")
        if any(s <= 0.0 for s in initial_shares):
            raise ValueError("every outcome needs strictly positive seed shares")
        if config.outcome_labels is not None and len(config.outcome_labels) != len(initial_shares):
            raise ValueError("outcome_labels length must match number of outcomes")
        if config.band_edges is not None and len(config.band_edges) != len(initial_shares) - 1:
            raise ValueError("band_edges must have exactly (num_outcomes - 1) entries")

        self.config = config
        self.q: List[float] = [float(s) for s in initial_shares]
        self.positions: Dict[str, List[float]] = {
            CREATOR_ADDRESS: [float(s) for s in initial_shares]
        }
        self.now_min: float = 0.0  # current time in minutes; advanced by the harness

    # ------------------------------------------------------------------ clock

    def set_time(self, now_min: float) -> None:
        """Set the market clock (minutes from open). The harness advances this."""
        self.now_min = float(now_min)

    def time_to_close(self) -> Optional[float]:
        """Minutes until trading closes, or ``None`` if the market has no clock."""
        if self.config.trading_close_min is None:
            return None
        return max(0.0, self.config.trading_close_min - self.now_min)

    # ------------------------------------------------------------------ state

    @property
    def n_outcomes(self) -> int:
        return len(self.q)

    @property
    def k(self) -> float:
        return self.config.k

    @property
    def fee(self) -> float:
        return self.config.fee

    @property
    def pool(self) -> float:
        """Collateral invariant C(q) = k*sqrt(sum q_i**2)."""
        return dpm_math.collateral(self.q, self.k)

    def current_prices(self) -> List[float]:
        """Spot prices p_i (sum of squares == k**2; they do NOT sum to 1)."""
        return dpm_math.spot_prices(self.q, self.k)

    def current_probabilities(self) -> List[float]:
        """Implied probabilities pi_i = (p_i/k)**2 (sum to 1)."""
        return dpm_math.implied_probabilities(self.q)

    def position(self, address: str = DEFAULT_TRADER) -> List[float]:
        return list(self.positions.get(address, [0.0] * self.n_outcomes))

    # ------------------------------------------------------------------ quotes

    def cost_to_buy(self, outcome: int, shares: float) -> float:
        """Tokens required to buy ``shares`` of ``outcome`` (fee included)."""
        self._check_outcome(outcome)
        return dpm_math.buy_cost(self.q, self.k, outcome, shares, self.fee)

    def proceeds_to_sell(self, outcome: int, shares: float) -> float:
        """Net tokens received for selling ``shares`` of ``outcome`` (fee applied)."""
        self._check_outcome(outcome)
        return dpm_math.sell_proceeds(self.q, self.k, outcome, shares, self.fee)

    def shares_for_spend(self, outcome: int, spend: float) -> float:
        """Pure curve quote: shares buyable for ``spend`` tokens (fee included).

        The harness uses this to convert a notional ("spend X") model action into
        the share-denominated :class:`Order` the market consumes. No cash state is
        touched here -- it is just an inversion of the cost curve.
        """
        self._check_outcome(outcome)
        return dpm_math.shares_for_buy_spend(self.q, self.k, outcome, spend, self.fee)

    def shares_for_proceeds(self, outcome: int, proceeds: float) -> float:
        """Pure curve quote: shares to sell to receive ~``proceeds`` tokens (clamped to q)."""
        self._check_outcome(outcome)
        return dpm_math.shares_for_sell_proceeds(self.q, self.k, outcome, proceeds, self.fee)

    # ------------------------------------------------------------------ seam: structure / rule

    def outcomes(self) -> List[OutcomeInfo]:
        """The outcome structure: one band per outcome (index, label, price range)."""
        infos: List[OutcomeInfo] = []
        edges = self.config.band_edges
        for i in range(self.n_outcomes):
            if self.config.outcome_labels is not None:
                label = self.config.outcome_labels[i]
            elif edges is not None:
                low = edges[i - 1] if i > 0 else None
                high = edges[i] if i < len(edges) else None
                label = _band_label(low, high)
            else:
                label = "outcome %d" % i
            low = None
            high = None
            if edges is not None:
                low = edges[i - 1] if i > 0 else None
                high = edges[i] if i < len(edges) else None
            infos.append(OutcomeInfo(index=i, label=label, band_low=low, band_high=high))
        return infos

    def current_state(self) -> MarketState:
        """Snapshot for prompt-building: prices, probabilities, pool, position, clock."""
        return MarketState(
            prices=self.current_prices(),
            probabilities=self.current_probabilities(),
            pool=self.pool,
            my_position=self.position(DEFAULT_TRADER),
            time_to_close=self.time_to_close(),
        )

    def settlement_rule_text(self) -> str:
        return self.config.settlement_rule_text

    # ------------------------------------------------------------------ trading

    def apply_trade(self, order: Order, address: str = DEFAULT_TRADER) -> TradeResult:
        """Apply a share-denominated order, mutating state; return the fill.

        Never raises on a trade-validity problem (bad outcome, non-positive size,
        overselling) -- an eval loop must log the rejection and keep going. Such
        cases return ``TradeResult(ok=False, error=...)`` with state unchanged.
        ``hold`` is a successful no-op.
        """
        side = order.side.lower()

        if side == SIDE_HOLD:
            return self._result(True, SIDE_HOLD, order.outcome, 0.0, 0.0, 0.0, 0.0)

        if side not in (SIDE_BUY, SIDE_SELL):
            return self._rejected(order, "unknown side %r" % order.side)

        if not 0 <= order.outcome < self.n_outcomes:
            return self._rejected(order, "outcome %r out of range" % order.outcome)

        if order.shares <= 0.0:
            return self._rejected(order, "shares must be positive")

        outcome = order.outcome
        shares = order.shares

        if side == SIDE_BUY:
            cost = dpm_math.buy_cost(self.q, self.k, outcome, shares, self.fee)
            self.q[outcome] += shares
            self._credit(address, outcome, shares)
            return self._result(True, SIDE_BUY, outcome, shares, cost, 0.0, cost / shares)

        # sell
        held = self.positions.get(address, [0.0] * self.n_outcomes)[outcome]
        if shares > held + 1e-12:
            return self._rejected(
                order,
                "holds %g shares of outcome %d, cannot sell %g" % (held, outcome, shares),
            )
        proceeds = dpm_math.sell_proceeds(self.q, self.k, outcome, shares, self.fee)
        self.q[outcome] -= shares
        self._credit(address, outcome, -shares)
        return self._result(True, SIDE_SELL, outcome, shares, 0.0, proceeds, proceeds / shares)

    # ------------------------------------------------------------------ resolution

    def bucket_for_price(self, price: float) -> int:
        """Map a settlement reference price to its outcome bucket via band edges."""
        if self.config.band_edges is None:
            raise ValueError("market has no band_edges; cannot resolve a price to a bucket")
        for i, edge in enumerate(self.config.band_edges):
            if price < edge:
                return i
        return len(self.config.band_edges)

    def resolve_outcome(self, reference_value: float) -> int:
        """Seam method: the winning outcome is the band the reference price lands in."""
        return self.bucket_for_price(reference_value)

    def uniform_band_width(self) -> float:
        """Width of the (evenly spaced) interior bands.

        The band-relative price generator needs a single band-width scalar. This
        requires at least two band edges that are evenly spaced; raises otherwise.
        """
        edges = self.config.band_edges
        if edges is None or len(edges) < 2:
            raise ValueError("need >= 2 evenly spaced band_edges to define a band width")
        spacings = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        first = spacings[0]
        if first <= 0.0 or any(abs(s - first) > 1e-9 for s in spacings):
            raise ValueError("band_edges are not evenly spaced; band width is ambiguous")
        return first

    def settle(self, winning_outcome: int) -> Dict[str, float]:
        """Split the whole pool pro-rata across all holders of winning shares."""
        self._check_outcome(winning_outcome)
        pool = self.pool
        total_winning = self.q[winning_outcome]
        payouts: Dict[str, float] = {}
        for address, holdings in self.positions.items():
            winning_shares = holdings[winning_outcome]
            if winning_shares > 0.0:
                payouts[address] = dpm_math.settlement_payout(pool, winning_shares, total_winning)
        return payouts

    def liquidate(self, address: str = DEFAULT_TRADER) -> float:
        """Recover the spot value of an address's held shares (no-winner case)."""
        held = self.positions.get(address, [0.0] * self.n_outcomes)
        return dpm_math.liquidation_reward(self.q, self.k, held)

    # ------------------------------------------------------------------ internals

    def _check_outcome(self, outcome: int) -> None:
        if not 0 <= outcome < self.n_outcomes:
            raise ValueError("outcome %r out of range [0, %d)" % (outcome, self.n_outcomes))

    def _credit(self, address: str, outcome: int, delta: float) -> None:
        holdings = self.positions.setdefault(address, [0.0] * self.n_outcomes)
        holdings[outcome] += delta

    def _result(
        self,
        ok: bool,
        side: str,
        outcome: int,
        shares: float,
        cost: float,
        proceeds: float,
        fill_price: float,
        error: Optional[str] = None,
    ) -> TradeResult:
        return TradeResult(
            ok=ok,
            side=side,
            outcome=outcome,
            shares=shares,
            cost=cost,
            proceeds=proceeds,
            fill_price=fill_price,
            prices_after=self.current_prices(),
            probabilities_after=self.current_probabilities(),
            error=error,
        )

    def _rejected(self, order: Order, error: str) -> TradeResult:
        # State is unchanged on a rejection, so prices/probabilities reflect "before".
        return self._result(False, order.side, order.outcome, 0.0, 0.0, 0.0, 0.0, error)


def _band_label(low: Optional[float], high: Optional[float]) -> str:
    if low is None:
        return "< %g" % high
    if high is None:
        return ">= %g" % low
    return "[%g, %g)" % (low, high)
