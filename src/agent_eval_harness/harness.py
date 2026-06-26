"""The market-agnostic harness loop (spec.md piece 4).

Drives a :class:`~agent_eval_harness.market.Market` with a model over a
:class:`~agent_eval_harness.price_path.PricePath`, logging the FULL prompt and
response for every timestep (line one -- this is the whole dataset and exactly
what an REE replay needs), then settles against the path's reference price and
computes the run's behavioral footprint.

The loop is agnostic to market type: it only ever calls the thin Market seam and
the wallet/prompt/parse helpers. Information discipline (never show a price after
``t``) is enforced by sourcing every prompt from ``path.prices_up_to(t)``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .action import ACTION_HOLD, parse
from .market import Market, Order
from .metrics import Footprint, compute_footprint
from .price_path import PricePath
from .prompt import DEFAULT_TOKEN, build_prompt
from .wallet import Wallet, plan_order, settle_fill

AGENT_ADDRESS = "agent"


@dataclass
class TurnLog:
    """Everything observed at one timestep -- the full input->output pair."""

    t: float
    time_to_close: Optional[float]
    prompt: str
    response: str
    parsed: dict
    planned_order: Optional[dict]
    trade_result: Optional[dict]
    executed: bool
    rejected: bool
    error: Optional[str]
    notional: float
    cash_before: float
    cash_after: float
    prices_before: List[float]
    probabilities_before: List[float]
    position_before: List[float]


@dataclass
class EpisodeLog:
    """The full record of one run, plus its footprint."""

    turns: List[TurnLog]
    settlement_price: float
    winning_outcome: int
    payout: float
    starting_cash: float
    final_cash: float
    pnl: float
    token: str
    trading_close_min: float
    seed: Optional[int]
    footprint: Footprint
    opening_probabilities: List[float] = field(default_factory=list)
    closing_probabilities: List[float] = field(default_factory=list)

    def to_jsonl(self, path: str) -> None:
        """Persist one JSON line per turn (input->output) plus a settlement line."""
        with open(path, "w", encoding="utf-8") as fh:
            for turn in self.turns:
                fh.write(json.dumps({"type": "turn", **asdict(turn)}) + "\n")
            fh.write(
                json.dumps(
                    {
                        "type": "settlement",
                        "settlement_price": self.settlement_price,
                        "winning_outcome": self.winning_outcome,
                        "payout": self.payout,
                        "starting_cash": self.starting_cash,
                        "final_cash": self.final_cash,
                        "pnl": self.pnl,
                        "opening_probabilities": self.opening_probabilities,
                        "closing_probabilities": self.closing_probabilities,
                        "footprint": asdict(self.footprint),
                    }
                )
                + "\n"
            )


def run_episode(
    market: Market,
    model,
    path: PricePath,
    starting_cash: float = 1000.0,
    token: str = DEFAULT_TOKEN,
    seed: Optional[int] = None,
) -> EpisodeLog:
    """Run one full episode and return its log + footprint."""
    wallet = Wallet.with_budget(starting_cash)
    turns: List[TurnLog] = []
    trade_history: List[dict] = []  # compact, prompt-facing

    market.set_time(0.0)
    opening_probabilities = list(market.current_state().probabilities)

    for (t, _price) in path.trading_points():
        market.set_time(t)
        state = market.current_state()
        visible = path.prices_up_to(t)

        prompt = build_prompt(
            settlement_rule=market.settlement_rule_text(),
            outcomes=market.outcomes(),
            state=state,
            visible_points=visible,
            cash=wallet.cash,
            trade_history=trade_history,
            token=token,
        )
        response = model(prompt)
        parsed = parse(response)

        cash_before = wallet.cash
        order, reject = plan_order(market, wallet, parsed, state)

        trade_result = None
        executed = False
        error = reject
        if order is not None and order.side != ACTION_HOLD:
            trade_result = market.apply_trade(order)
            if trade_result.ok:
                settle_fill(wallet, trade_result)
                executed = True
            else:
                error = trade_result.error

        notional = 0.0
        if executed and trade_result is not None:
            notional = trade_result.cost + trade_result.proceeds

        turns.append(
            TurnLog(
                t=t,
                time_to_close=state.time_to_close,
                prompt=prompt,
                response=response,
                parsed=parsed.as_dict(),
                planned_order=(asdict_order(order) if order is not None else None),
                trade_result=(_result_dict(trade_result) if trade_result is not None else None),
                executed=executed,
                rejected=(order is None) or (trade_result is not None and not trade_result.ok),
                error=error,
                notional=notional,
                cash_before=cash_before,
                cash_after=wallet.cash,
                prices_before=state.prices,
                probabilities_before=state.probabilities,
                position_before=state.my_position,
            )
        )
        trade_history.append(
            {
                "t": t,
                "action": parsed.action,
                "outcome": parsed.outcome,
                "size": parsed.size,
                "ok": executed,
                "error": error,
            }
        )

    # Implied probabilities the market reached by trading close -- the raw material
    # for the discovery metric (did the probe move probability toward the truth?).
    closing_probabilities = list(market.current_state().probabilities)

    # Settlement against the reference price (end of the path, after the dead window).
    market.set_time(path.points[-1][0])
    settlement_price = path.settlement_price()
    winning_outcome = market.resolve_outcome(settlement_price)
    payouts = market.settle(winning_outcome)
    payout = payouts.get(AGENT_ADDRESS, 0.0)
    wallet.credit(payout)
    final_cash = wallet.cash

    footprint = compute_footprint(
        turns=turns,
        starting_cash=starting_cash,
        trading_close_min=path.trading_close_min,
        payout=payout,
        final_cash=final_cash,
        winning_outcome=winning_outcome,
    )

    return EpisodeLog(
        turns=turns,
        settlement_price=settlement_price,
        winning_outcome=winning_outcome,
        payout=payout,
        starting_cash=starting_cash,
        final_cash=final_cash,
        pnl=final_cash - starting_cash,
        token=token,
        trading_close_min=path.trading_close_min,
        seed=seed,
        footprint=footprint,
        opening_probabilities=opening_probabilities,
        closing_probabilities=closing_probabilities,
    )


def asdict_order(order: Order) -> dict:
    return {"side": order.side, "outcome": order.outcome, "shares": order.shares}


def _result_dict(result) -> dict:
    return {
        "ok": result.ok,
        "side": result.side,
        "outcome": result.outcome,
        "shares": result.shares,
        "cost": result.cost,
        "proceeds": result.proceeds,
        "fill_price": result.fill_price,
        "error": result.error,
    }
