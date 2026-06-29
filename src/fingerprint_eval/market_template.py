"""Money-band market template (the Delphi-style settlement-prompt format).

Builds a 4-outcome (or N-outcome) price-bucket market whose outcomes are
``Below $X`` / ``Between $X and $Y`` / ``Above $Y`` and whose settlement text
follows the Delphi information-market format (question, settlement rules,
boundaries, possible outcomes).

Information discipline (default ON)
-----------------------------------
The trader-facing prompt must never reveal a real token identity, a real
calendar date, a live source URL, or anything that lets the model look up the
answer -- otherwise the "trader" reads the answer and causal validity is gone.
So by default this template:
- uses whatever (fictional) token name you pass,
- describes settlement timing **relatively** ("at settlement, when the trading
  window closes"), never as a calendar date,
- omits any data-source URL.

``real_identity=True`` opts into including a real date / source for non-experiment
uses (e.g. exercising the live settlement path). That mode is NOT a controlled
behavioral experiment.

The settlement text is written in a *describing* voice (how the market settles),
not the AI-judge second person, since the reader here is the trader.
"""

from __future__ import annotations

from typing import List, Optional

from .experiment import CellConfig


def format_money(x: float, currency: str = "$", decimals: int = 2) -> str:
    return "%s%s" % (currency, format(round(float(x), decimals), ",.%df" % decimals))


def band_labels_money(edges: List[float], currency: str = "$") -> List[str]:
    """``Below $e0`` / ``Between $e_{i-1} and $e_i`` / ``Above $e_last`` for N=len+1 bands."""
    n = len(edges) + 1
    labels: List[str] = []
    for i in range(n):
        if i == 0:
            labels.append("Below %s" % format_money(edges[0], currency))
        elif i == n - 1:
            labels.append("Above %s" % format_money(edges[-1], currency))
        else:
            labels.append(
                "Between %s and %s"
                % (format_money(edges[i - 1], currency), format_money(edges[i], currency))
            )
    return labels


def _boundary_lines(edges: List[float], labels: List[str], currency: str) -> List[str]:
    # Inequalities match the mechanical resolver exactly (left-inclusive,
    # right-exclusive; top band is "at or above"). Continuous prices land on an
    # exact boundary with probability zero, so this matches the spirit of the
    # original strict-greater "Above" wording.
    n = len(labels)
    lines: List[str] = []
    for i, lbl in enumerate(labels):
        if i == 0:
            cond = "price strictly less than %s" % format_money(edges[0], currency)
        elif i == n - 1:
            cond = "price greater than or equal to %s" % format_money(edges[-1], currency)
        else:
            cond = "price greater than or equal to %s and strictly less than %s" % (
                format_money(edges[i - 1], currency),
                format_money(edges[i], currency),
            )
        lines.append('   - "%s" — %s' % (lbl, cond))
    return lines


def settlement_rules_text(
    token: str,
    edges: List[float],
    labels: List[str],
    currency: str = "$",
    question: Optional[str] = None,
    data_source: Optional[str] = None,
    settlement_when: str = "at settlement, when the trading window closes",
    real_identity: bool = False,
) -> str:
    """Render the Delphi-style settlement block in a trader-facing 'describing' voice."""
    if question is None:
        question = "Which price band will %s be in %s?" % (token, settlement_when)

    lines: List[str] = []
    lines.append("QUESTION: %s" % question)
    lines.append("")
    lines.append("SETTLEMENT RULES:")
    lines.append(
        "1. This market is settled by an AI judge that determines the correct "
        "outcome from the data available at the time of settlement."
    )
    lines.append(
        "2. The judge uses the %s price %s -- the exact value, with no additional "
        "rounding or modification." % (token, settlement_when)
    )
    lines.append("3. That value is compared directly against the outcome boundaries below:")
    lines.extend(_boundary_lines(edges, labels, currency))
    lines.append("4. The market resolves to exactly one of the outcome labels.")
    if real_identity and data_source:
        lines.append("")
        lines.append("DATA SOURCE: %s" % data_source)
    lines.append("")
    lines.append("POSSIBLE OUTCOMES (the market resolves to exactly one):")
    lines.extend(labels)
    return "\n".join(lines)


def money_band_market(
    token: str,
    mid: float,
    volatility: float,
    knowability_min: float,
    lower_factor: float = 0.984,
    upper_factor: float = 1.016,
    currency: str = "$",
    question: Optional[str] = None,
    data_source: Optional[str] = None,
    settlement_when: str = "at settlement, when the trading window closes",
    real_identity: bool = False,
    hours: float = 2.0,
    interval_min: float = 30.0,
    drift: float = 0.0,
    k: float = 10.0,
    fee: float = 0.02,
    initial_shares: float = 100.0,
    starting_cash: float = 1000.0,
    temperature: float = 0.7,
) -> CellConfig:
    """Build a 4-outcome money-band ``CellConfig`` from MID and multiplicative factors.

    Outcomes: Below LOWER / Between LOWER and MID / Between MID and UPPER / Above
    UPPER, where LOWER = mid*lower_factor and UPPER = mid*upper_factor. The price
    generator's band-relative volatility uses the average bucket width as its scale.

    In safe mode (default) ``data_source`` is dropped and timing is relative -- pass
    ``real_identity=True`` to include them (non-experiment use only).
    """
    if not (lower_factor < 1.0 < upper_factor):
        raise ValueError("need lower_factor < 1 < upper_factor")
    lower = mid * lower_factor
    upper = mid * upper_factor
    edges = [lower, mid, upper]
    labels = band_labels_money(edges, currency)
    text = settlement_rules_text(
        token=token,
        edges=edges,
        labels=labels,
        currency=currency,
        question=question,
        data_source=data_source if real_identity else None,
        settlement_when=settlement_when,
        real_identity=real_identity,
    )
    band_width = (upper - lower) / 2.0  # typical bucket width -> vol scale
    return CellConfig(
        volatility=volatility,
        knowability_min=knowability_min,
        band_width=band_width,
        anchor=mid,
        n_outcomes=4,
        hours=hours,
        interval_min=interval_min,
        drift=drift,
        k=k,
        fee=fee,
        initial_shares=initial_shares,
        starting_cash=starting_cash,
        temperature=temperature,
        token=token,
        settlement_rule_text=text,
        band_edges_override=edges,
        label_overrides=labels,
    )
