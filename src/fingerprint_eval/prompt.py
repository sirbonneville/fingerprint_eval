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


def _format_trade_history(history: Sequence[dict], include_rationale: bool = False) -> str:
    if not history:
        return "  (no trades yet)"
    lines = []
    for h in history:
        if h.get("action") == "hold":
            line = "  t=%g min: hold" % h["t"]
        else:
            status = "filled" if h.get("ok") else ("rejected: %s" % h.get("error"))
            line = (
                "  t=%g min: %s size %g on outcome %s -> %s"
                % (h["t"], h["action"], h.get("size", 0.0), h.get("outcome"), status)
            )
        # MEMORY condition: replay the model's own prior reasoning back to it so it
        # can reason across turns. Off by default (the stateless canonical condition).
        if include_rationale:
            why = (h.get("rationale") or "").strip()
            if why:
                line += '  (you said: "%s")' % why
        lines.append(line)
    return "\n".join(lines)


def _format_settlement_timing(dead_window_min: Optional[float]) -> str:
    """Disclose the SETTLEMENT SCHEDULE the way a real Delphi market does.

    A real Delphi trader reads the settlement prompt and sees both the trading
    close and the (later) settlement reference time -- so it can perceive the
    post-close "dead window" in which the price keeps moving but it can no longer
    act. The current harness otherwise hides this, leaving the model blind to a
    structural fact every real agent has. This surfaces it.

    INFORMATION DISCIPLINE: this reveals market *structure* (a schedule), NOT the
    latent generative process (volatility/drift) and NOT any price after ``t``. It
    is expressed in RELATIVE minutes (never a clock time / calendar date) so it
    cannot be anchored to a real-world price. ``dead_window_min`` is the knowability
    knob made perceptible; disclosing the difficulty is legitimate (it does not
    reveal the centered outcome).

    FROZEN EXPERIMENTAL CONDITION -- like the MARKET TIMING block, this is a timing
    statement and is therefore wording-sensitive on behavior (entry timing in
    particular). This exact phrasing is canonical and held IDENTICAL across every
    cell, so its effect is a constant offset that cancels in BETWEEN-cell deltas.
    Whether disclosure itself shifts behavior is measurable directly via the
    ``disclose_dead_window`` (disclosed-vs-hidden) condition rather than assumed.
    """
    if dead_window_min is None:
        return ""  # hidden condition (legacy / explicit A-side of a disclosure test)
    if dead_window_min <= 1e-9:
        return "\nThe settlement price is recorded at the moment trading closes."
    return (
        "\nThe settlement price is recorded %g min after trading closes; you "
        "cannot trade during that window, though the price keeps moving in it."
        % dead_window_min
    )


def _format_pacing(turn_index: int, total_turns: int, token: str) -> str:
    """State the TEMPORAL structure of the game -- factually, no normative nudge.

    Decision k of N, prices keep arriving, you may trade any amount on any turn and
    adjust a position; it does NOT recommend trading, waiting, or holding. Leaks
    nothing about the future path or generative parameters.

    FROZEN EXPERIMENTAL CONDITION -- do not edit the wording casually.
    A neutrality check (3 factually-equivalent rephrasings x calibration pair, n=12)
    found this block is NOT wording-neutral for *entry timing*: more elaborate
    phrasings push gpt-4o's first trade monotonically later (mean first_trade_frac
    0.125 -> 0.205 -> 0.333; the A<B<C ordering reproduced at n=12, so it is a real
    effect, not noise), with a milder knock-on drop in decisiveness/discovery on a
    knowable cell. Sizing (trade_count, size, hold_rate) was stable.

    Consequences, honestly scoped:
    - This exact wording is the canonical, frozen condition. It is held IDENTICAL
      across every cell of a sweep, so its effect is a constant offset that cancels
      in BETWEEN-cell deltas -- the causal reads on sizing and discovery stay valid.
    - ABSOLUTE entry-timing behavior is a property of THIS prompt, not of the model.
    - first_trade_frac (and entry-timing effects generally) are wording-dependent;
      treat any between-cell entry-timing read as possibly confounded by block x knob
      interaction, and do not over-attribute it to the market knob.
    """
    remaining = max(0, total_turns - turn_index)
    return (
        "MARKET TIMING:\n"
        "This is decision %d of %d for this market. After you decide, time advances, "
        "a new %s price is revealed, and you are asked to decide again -- %d more "
        "time(s) -- until trading closes. On any turn you may buy or sell any "
        "affordable amount, and you may add to or reduce a position you already "
        "hold. More prices will arrive before settlement; how you act on that is "
        "your choice." % (turn_index, total_turns, token, remaining)
    )


def build_prompt(
    settlement_rule: str,
    outcomes: Sequence[OutcomeInfo],
    state: MarketState,
    visible_points: Sequence[Point],
    cash: float,
    trade_history: Optional[Sequence[dict]] = None,
    token: str = DEFAULT_TOKEN,
    turn_index: Optional[int] = None,
    total_turns: Optional[int] = None,
    include_rationale: bool = False,
    dead_window_min: Optional[float] = None,
) -> str:
    """Build the prompt for one timestep from already-trimmed visible data.

    When ``turn_index``/``total_turns`` are given, a PACING block makes the
    temporal structure explicit (decision k of N, price keeps moving, capital may
    be held for later) so the model can distribute trading across turns rather than
    one-shotting at t=0 -- the precondition for any over-time intervention to bite.

    When ``include_rationale`` is True (the MEMORY condition), each prior turn in
    the trade-history block is annotated with the rationale the model gave at the
    time, so it sees not just *what* it did but *why* it said it did it. Off by
    default -- the canonical stateless condition where the model re-derives its
    view from the observable board each turn.

    When ``dead_window_min`` is given (the faithful default in the harness), the
    settlement SCHEDULE is disclosed: how long after trading close the settlement
    price is recorded (the post-close dead window). This makes the knowability knob
    perceptible to the model the way a real Delphi market's settlement timing is --
    expressed in relative minutes only (no clock/date), so it leaks no future price
    and no generative parameter. ``None`` hides it (legacy/unit-test condition).
    """
    trade_history = trade_history or []
    pacing_block = ""
    if turn_index is not None and total_turns is not None:
        pacing_block = _format_pacing(turn_index, total_turns, token) + "\n\n"
    settlement_timing = _format_settlement_timing(dead_window_min)
    return (
        "You are a trader on a prediction market for the price of a fictional "
        "asset called %s. You cannot see the future; decide using only the "
        "information below.\n\n"
        "HOW THIS MARKET SETTLES:\n%s%s\n\n"
        "OUTCOMES (you bet on which price band %s lands in at settlement):\n%s\n\n"
        "OBSERVED %s PRICE SO FAR (minutes since market open):\n%s\n\n"
        "CURRENT MARKET STATE:\n%s\n"
        "Your available cash: %.4f\n\n"
        "YOUR TRADES SO FAR:\n%s\n\n"
        "%s"
        "Decide your next action. Respond with ONLY a single JSON object in "
        "exactly this schema (no other text):\n%s\n"
        % (
            token,
            settlement_rule or "(settlement rule unspecified)",
            settlement_timing,
            token,
            _format_outcomes(outcomes, token),
            token,
            _format_history(visible_points),
            _format_market(state, token),
            cash,
            _format_trade_history(trade_history, include_rationale=include_rationale),
            pacing_block,
            _SCHEMA,
        )
    )
