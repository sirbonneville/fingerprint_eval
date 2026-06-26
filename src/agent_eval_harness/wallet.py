"""The trader's wallet and order planning (harness-owned economics).

Per the Piece-2 decisions: the market is pure pricing + positions and never sees
cash. The wallet lives here in the harness and owns the budget, affordability,
and the notional->shares conversion. The model's notional action is turned into a
share-denominated :class:`~agent_eval_harness.market.Order` here, using the
market's pure cost-curve quotes.

Conventions chosen for v1 (clean, observable behavior; easy to revisit):
- ``buy``: ``size`` = currency to spend. Rejected if it exceeds cash -- an
  overspend attempt is itself signal, not silently clamped.
- ``sell``: ``size`` = currency to receive, converted to shares and clamped to
  the position actually held ("get me out" semantics). Rejected only if nothing
  is held in that outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .action import ACTION_BUY, ACTION_HOLD, ACTION_SELL, ParsedAction
from .market import Order, MarketState

_EPS = 1e-9


@dataclass
class Wallet:
    """A simple cash account starting at a fixed budget."""

    starting_cash: float
    cash: float

    @classmethod
    def with_budget(cls, budget: float) -> "Wallet":
        if budget < 0.0:
            raise ValueError("budget must be non-negative")
        return cls(starting_cash=float(budget), cash=float(budget))

    def debit(self, amount: float) -> None:
        self.cash -= amount

    def credit(self, amount: float) -> None:
        self.cash += amount


def plan_order(
    market,
    wallet: Wallet,
    action: ParsedAction,
    state: MarketState,
) -> Tuple[Optional[Order], Optional[str]]:
    """Turn a notional model action into a share Order (or a rejection reason).

    Returns ``(order, None)`` to execute, or ``(None, reason)`` to reject. A hold
    returns a hold Order so the turn is still recorded as an explicit no-op.
    ``state`` is the pre-trade snapshot (used for the held position on sells).
    """
    n_outcomes = len(state.prices)

    if action.action == ACTION_HOLD:
        return Order(side=ACTION_HOLD), None

    if not 0 <= action.outcome < n_outcomes:
        return None, "outcome %r out of range" % action.outcome

    if action.size <= 0.0:
        return None, "non-positive size"

    if action.action == ACTION_BUY:
        if action.size > wallet.cash + _EPS:
            return None, "insufficient cash: size %g > cash %g" % (action.size, wallet.cash)
        shares = market.shares_for_spend(action.outcome, action.size)
        if shares <= _EPS:
            return None, "spend resolves to zero shares"
        return Order(side=ACTION_BUY, outcome=action.outcome, shares=shares), None

    if action.action == ACTION_SELL:
        held = state.my_position[action.outcome]
        if held <= _EPS:
            return None, "no position to sell in outcome %d" % action.outcome
        shares = market.shares_for_proceeds(action.outcome, action.size)
        shares = min(shares, held)  # "get me out" -- never sell more than held
        if shares <= _EPS:
            return None, "sell resolves to zero shares"
        return Order(side=ACTION_SELL, outcome=action.outcome, shares=shares), None

    return None, "unknown action %r" % action.action


def settle_fill(wallet: Wallet, result) -> None:
    """Apply a successful trade's cash effect to the wallet."""
    if not result.ok:
        return
    if result.cost:
        wallet.debit(result.cost)
    if result.proceeds:
        wallet.credit(result.proceeds)
