"""The thin Market interface -- the seam (spec.md piece 2).

The harness only ever calls these five methods; everything market-type-specific
(settlement rule, outcome structure, payout math) lives behind them. There is
exactly one concrete implementation in v1: ``dpm.PriceBucketMarket``.

Design contract for the seam:
- The market is **pure pricing + position state**. It speaks in *shares* (its
  native unit) and *tokens* (cost/proceeds). It never knows about a trader's
  cash, budget, or affordability -- those are harness/portfolio concerns, kept
  out of the seam so the verified contract math inside the market stays
  untouched and a second market type drops in without untangling economics.
- The model thinks in money (notional). The harness converts notional -> shares
  via the market's pure cost curve, then hands the market a share-denominated
  :class:`Order`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Optional

# Trade sides the seam understands. "hold" is a valid no-op so the model can
# explicitly decline to trade and we still log a turn.
SIDE_BUY = "buy"
SIDE_SELL = "sell"
SIDE_HOLD = "hold"


@dataclass
class Order:
    """A share-denominated instruction at the market boundary.

    The harness builds this from the model's notional action (it owns the
    currency->shares conversion and affordability checks). ``shares`` is ignored
    for ``hold``.
    """

    side: str
    outcome: int = 0
    shares: float = 0.0


@dataclass
class OutcomeInfo:
    """Describes one outcome (the 'outcome structure' that varies by market type).

    For a price-bucket market each outcome is a price band; ``band_low`` /
    ``band_high`` are the inclusive-low / exclusive-high edges (``None`` = open).
    """

    index: int
    label: str
    band_low: Optional[float] = None
    band_high: Optional[float] = None


@dataclass
class MarketState:
    """Snapshot the harness reads each timestep to build the prompt.

    Note: cash is deliberately absent -- it is harness/portfolio-owned and merged
    in by the harness when constructing the prompt.
    """

    prices: List[float]
    probabilities: List[float]
    pool: float
    my_position: List[float]
    time_to_close: Optional[float] = None  # minutes; None if market has no clock


@dataclass
class TradeResult:
    """Outcome of a single applied order.

    ``ok`` is False for rejected orders (e.g. selling more than held, bad
    outcome index). The seam never raises on a trade-validity problem -- an eval
    loop must log the rejection and continue rather than crash.
    """

    ok: bool
    side: str
    outcome: int
    shares: float                 # shares actually filled
    cost: float                   # tokens paid in (buy); 0 otherwise
    proceeds: float               # tokens paid out (sell); 0 otherwise
    fill_price: float             # avg tokens/share for this fill; 0 if no fill
    prices_after: List[float]
    probabilities_after: List[float]
    error: Optional[str] = None


class Market(abc.ABC):
    """The thin interface the market-agnostic harness drives.

    The first five methods are spec.md piece 2 verbatim. The remaining few are
    the minimum the harness needs to stay market-agnostic while supporting a
    money-thinking trader and a price-driven settlement reference:
    - ``set_time``        : advance the market clock (drives ``time_to_close``).
    - ``shares_for_spend`` / ``shares_for_proceeds`` : pure cost-curve quotes the
      harness uses to convert a notional model action into a share Order. They
      touch no cash -- affordability stays in the harness.
    - ``resolve_outcome`` : map a settlement reference value to the winning
      outcome (the price->bucket mapping is market-type-specific).
    """

    @abc.abstractmethod
    def outcomes(self) -> List[OutcomeInfo]:
        """How many outcomes there are and what they mean."""

    @abc.abstractmethod
    def current_state(self) -> MarketState:
        """Prices, probabilities, pool, the trader's position, time to close."""

    @abc.abstractmethod
    def apply_trade(self, order: Order) -> TradeResult:
        """Apply a share-denominated order, mutating state; return the fill."""

    @abc.abstractmethod
    def settle(self, winning_outcome: int) -> Dict[str, float]:
        """Resolve the market; return per-address payouts."""

    @abc.abstractmethod
    def settlement_rule_text(self) -> str:
        """The prompt-facing description of how this market settles."""

    @abc.abstractmethod
    def set_time(self, now_min: float) -> None:
        """Advance the market clock to ``now_min`` minutes from open."""

    @abc.abstractmethod
    def shares_for_spend(self, outcome: int, spend: float) -> float:
        """Pure quote: shares buyable for ``spend`` tokens (fee included)."""

    @abc.abstractmethod
    def shares_for_proceeds(self, outcome: int, proceeds: float) -> float:
        """Pure quote: shares to sell to receive ~``proceeds`` tokens."""

    @abc.abstractmethod
    def resolve_outcome(self, reference_value: float) -> int:
        """Map a settlement reference value to the winning outcome index."""
