"""Per-run behavioral footprint (spec.md piece 4 'compute footprint' + piece 5 vocab).

These are single-run metrics computed with one shared descriptive vocabulary so
that cells are comparable. The experiment runner (piece 5) reads the
*distribution* of these across N runs; nothing here aggregates across runs.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class Discovery:
    """The headline metric: did implied probability discover the true outcome?

    All single-run; the runner reads the distribution across N. ``prob_on_winner``
    is the probability mass the market placed on the eventual winner at trading
    close. ``brier`` is the multi-class Brier score vs the realized one-hot outcome
    (lower is better). ``edge_over_open`` is how much the probe improved the
    estimate from the opening probabilities (positive => moved toward truth).

    IMPORTANT: with a stand-in model these numbers describe plumbing, not
    discovery -- a non-strategic trader does not discover prices. Only trust them
    once a real model is behind the Model protocol.
    """

    prob_on_winner: float
    brier: float
    edge_over_open: float


def compute_discovery(
    closing_probabilities: Sequence[float],
    winning_outcome: int,
    opening_probabilities: Optional[Sequence[float]] = None,
) -> Discovery:
    closing = list(closing_probabilities)
    prob_on_winner = closing[winning_outcome] if 0 <= winning_outcome < len(closing) else 0.0
    brier = sum(
        (p - (1.0 if i == winning_outcome else 0.0)) ** 2 for i, p in enumerate(closing)
    )
    if opening_probabilities is not None and 0 <= winning_outcome < len(opening_probabilities):
        edge = prob_on_winner - opening_probabilities[winning_outcome]
    else:
        edge = 0.0
    return Discovery(prob_on_winner=prob_on_winner, brier=brier, edge_over_open=edge)


@dataclass
class Footprint:
    n_turns: int = 0

    # activity
    trade_count: int = 0          # executed (filled) non-hold trades
    n_buys: int = 0
    n_sells: int = 0
    n_holds: int = 0              # explicit, well-formed holds
    n_rejected: int = 0           # well-formed but rejected (e.g. unaffordable)
    n_parse_failures: int = 0     # malformed model output

    # entry timing
    first_trade_t: Optional[float] = None
    first_trade_frac: Optional[float] = None  # fraction of trading window elapsed

    # sizing
    sizes_notional: List[float] = field(default_factory=list)
    median_trade_size: Optional[float] = None
    total_volume: float = 0.0

    # churn / spread
    direction_switches: int = 0   # buy<->sell flips across executed trades
    outcomes_traded: int = 0      # distinct outcomes touched (bucket spread)

    # volume vs time-to-close: (time_to_close, notional) per executed trade
    volume_by_ttc: List[Tuple[float, float]] = field(default_factory=list)

    # outcome / pnl
    winning_outcome: Optional[int] = None
    payout: float = 0.0
    final_cash: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    win: bool = False


def compute_footprint(
    turns,
    starting_cash: float,
    trading_close_min: float,
    payout: float,
    final_cash: float,
    winning_outcome: int,
) -> Footprint:
    fp = Footprint()
    fp.n_turns = len(turns)
    fp.winning_outcome = winning_outcome
    fp.payout = payout
    fp.final_cash = final_cash
    fp.pnl = final_cash - starting_cash
    fp.return_pct = (fp.pnl / starting_cash) if starting_cash else 0.0
    fp.win = fp.pnl > 0.0

    executed_sides: List[str] = []
    touched = set()
    for turn in turns:
        action = turn.parsed.get("action")
        if not turn.parsed.get("parse_ok", False):
            fp.n_parse_failures += 1
            continue
        if action == "hold":
            fp.n_holds += 1
            continue
        # buy/sell well-formed
        if not turn.executed:
            fp.n_rejected += 1
            continue
        # executed trade
        fp.trade_count += 1
        side = turn.trade_result["side"]
        executed_sides.append(side)
        if side == "buy":
            fp.n_buys += 1
        elif side == "sell":
            fp.n_sells += 1
        outcome = turn.trade_result["outcome"]
        touched.add(outcome)
        fp.sizes_notional.append(turn.notional)
        fp.total_volume += turn.notional
        ttc = turn.time_to_close if turn.time_to_close is not None else 0.0
        fp.volume_by_ttc.append((ttc, turn.notional))
        if fp.first_trade_t is None:
            fp.first_trade_t = turn.t
            fp.first_trade_frac = (turn.t / trading_close_min) if trading_close_min else None

    fp.outcomes_traded = len(touched)
    if fp.sizes_notional:
        fp.median_trade_size = statistics.median(fp.sizes_notional)
    for a, b in zip(executed_sides, executed_sides[1:]):
        if a != b:
            fp.direction_switches += 1
    return fp
