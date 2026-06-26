"""Prompt construction with information discipline (spec.md piece 4).

CARDINAL VALIDITY RULE: at timestep ``t`` the model must never see any price data
after ``t``. This builder is deliberately paranoid about it:
- It only ever receives the already-trimmed visible series (``visible_points``);
  it does no look-ahead itself.
- Time is expressed purely as "minutes since open" -- no calendar dates the model
  could anchor to a real-world price.
- The asset is a fictional ticker with no inferable real-world value.
- The generative parameters (volatility, drift, the path beyond ``t``) are never
  disclosed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .market import MarketState, OutcomeInfo

Point = Tuple[float, float]

DEFAULT_TOKEN = "ZQX"

_SCHEMA = (
    '{ "action": "buy" | "sell" | "hold", '
    '"outcome": <integer outcome index>, '
    '"size": <number, amount of CURRENCY to spend (buy) or receive (sell)>, '
    '"rationale": "<free text>" }'
)


def _format_outcomes(outcomes: Sequence[OutcomeInfo], token: str) -> str:
    lines = []
    for o in outcomes:
        if o.band_low is None and o.band_high is None:
            rng = o.label
        elif o.band_low is None:
            rng = "%s price < %g" % (token, o.band_high)
        elif o.band_high is None:
            rng = "%s price >= %g" % (token, o.band_low)
        else:
            rng = "%g <= %s price < %g" % (o.band_low, token, o.band_high)
        lines.append("  [%d] %s  (%s)" % (o.index, o.label, rng))
    return "\n".join(lines)


def _format_history(visible_points: Sequence[Point]) -> str:
    return "\n".join("  t=%g min: %.4f" % (t, p) for t, p in visible_points)


def _format_market(state: MarketState, token: str) -> str:
    prob = ", ".join(
        "[%d] %.1f%%" % (i, 100.0 * p) for i, p in enumerate(state.probabilities)
    )
    pos = ", ".join("[%d] %.4f" % (i, s) for i, s in enumerate(state.my_position))
    ttc = "unknown" if state.time_to_close is None else ("%g min" % state.time_to_close)
    return (
        "Implied probabilities: %s\n"
        "Time until trading closes: %s\n"
        "Your share position:   %s" % (prob, ttc, pos)
    )


def _format_trade_history(history: Sequence[dict]) -> str:
    if not history:
        return "  (no trades yet)"
    lines = []
    for h in history:
        if h.get("action") == "hold":
            lines.append("  t=%g min: hold" % h["t"])
        else:
            status = "filled" if h.get("ok") else ("rejected: %s" % h.get("error"))
            lines.append(
                "  t=%g min: %s size %g on outcome %s -> %s"
                % (h["t"], h["action"], h.get("size", 0.0), h.get("outcome"), status)
            )
    return "\n".join(lines)


def build_prompt(
    settlement_rule: str,
    outcomes: Sequence[OutcomeInfo],
    state: MarketState,
    visible_points: Sequence[Point],
    cash: float,
    trade_history: Optional[Sequence[dict]] = None,
    token: str = DEFAULT_TOKEN,
) -> str:
    """Build the prompt for one timestep from already-trimmed visible data."""
    trade_history = trade_history or []
    return (
        "You are a trader on a prediction market for the price of a fictional "
        "asset called %s. You cannot see the future; decide using only the "
        "information below.\n\n"
        "HOW THIS MARKET SETTLES:\n%s\n\n"
        "OUTCOMES (you bet on which price band %s lands in at settlement):\n%s\n\n"
        "OBSERVED %s PRICE SO FAR (minutes since market open):\n%s\n\n"
        "CURRENT MARKET STATE:\n%s\n"
        "Your available cash: %.4f\n\n"
        "YOUR TRADES SO FAR:\n%s\n\n"
        "Decide your next action. Respond with ONLY a single JSON object in "
        "exactly this schema (no other text):\n%s\n"
        % (
            token,
            settlement_rule or "(settlement rule unspecified)",
            token,
            _format_outcomes(outcomes, token),
            token,
            _format_history(visible_points),
            _format_market(state, token),
            cash,
            _format_trade_history(trade_history),
            _SCHEMA,
        )
    )
