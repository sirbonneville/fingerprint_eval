"""Verified Delphi DPM math (L2 / Euclidean-norm dynamic parimutuel).

Every formula here is transcribed structurally from the verified
``DynamicParimutuelMath`` contract source (see ``dpm_spec.md``). v1 uses
floating point; the 18-decimal fixed-point representation and the
"always round against the user" rounding can be layered on later without
changing the structure of these formulas.

Notation
--------
q       : sequence of outstanding shares per outcome (q_i)
k       : per-market liquidity constant (config.k)
sum_term: Sigma q_i**2
fee     : proportional trading fee (e.g. 0.02 for 2%)
"""

from __future__ import annotations

import math
from typing import Sequence


def sum_term(q: Sequence[float]) -> float:
    """Sigma q_i**2 (the contract's ``sumTerm36``)."""
    return sum(qi * qi for qi in q)


def collateral(q: Sequence[float], k: float) -> float:
    """Pool collateral invariant: C(q) = k * sqrt(Sigma q_i**2)."""
    return k * math.sqrt(sum_term(q))


def spot_prices(q: Sequence[float], k: float) -> list:
    """Marginal cost / spot price per outcome: p_i = k*q_i / sqrt(sum_term).

    Note: Sigma p_i**2 == k**2. Prices live on a sphere of radius k; they do
    NOT sum to 1. Use :func:`implied_probabilities` for probabilities.
    """
    st = sum_term(q)
    if st <= 0.0:
        raise ValueError("spot price undefined when sum_term is zero (no shares outstanding)")
    root = math.sqrt(st)
    return [k * qi / root for qi in q]


def implied_probabilities(q: Sequence[float]) -> list:
    """Implied probability per outcome: pi_i = q_i**2 / sum_term = (p_i/k)**2.

    Probability is the spot price SQUARED over k**2 (Sigma pi_i == 1). Price is
    NOT probability: a 0.51/share spot price implies ~26% probability, not 51%.
    """
    st = sum_term(q)
    if st <= 0.0:
        raise ValueError("implied probability undefined when sum_term is zero")
    return [qi * qi / st for qi in q]


def buy_cost(q: Sequence[float], k: float, j: int, delta: float, fee: float = 0.0) -> float:
    """Gross-plus-fee cost to buy ``delta`` shares of outcome ``j`` (quoteBuyExactOut).

    newSumTerm = sum_term + delta*(2*q_j + delta)   [== (q_j+delta)**2 - q_j**2]
    gross      = k*(sqrt(newSumTerm) - sqrt(sum_term))
    tokensIn   = gross / (1 - fee)                   [fee applied, rounded up on-chain]
    """
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    st = sum_term(q)
    new_st = st + delta * (2.0 * q[j] + delta)
    gross = k * (math.sqrt(new_st) - math.sqrt(st))
    return gross / (1.0 - fee)


def sell_proceeds(q: Sequence[float], k: float, j: int, delta: float, fee: float = 0.0) -> float:
    """Net payout for selling ``delta`` shares of outcome ``j`` (quoteSellExactIn).

    newSumTerm = sum_term - delta*(2*q_j - delta)    [reverses the buy]
    gross      = k*(sqrt(sum_term) - sqrt(newSumTerm))
    tokensOut  = gross * (1 - fee)                    [fee applied, rounded down on-chain]
    """
    if delta <= 0.0:
        raise ValueError("delta must be positive")
    if delta > q[j]:
        raise ValueError("cannot sell more shares than outstanding for the outcome")
    st = sum_term(q)
    new_st = st - delta * (2.0 * q[j] - delta)
    if new_st < 0.0:
        new_st = 0.0
    gross = k * (math.sqrt(st) - math.sqrt(new_st))
    return gross * (1.0 - fee)


def shares_for_buy_spend(
    q: Sequence[float], k: float, j: int, spend: float, fee: float = 0.0
) -> float:
    """Inverse of :func:`buy_cost`: shares of ``j`` buyable for ``spend`` tokens.

    Closed-form inversion of C(q) (no search needed). With gross = spend*(1-fee):
        sqrt(newSumTerm) = sqrt(sum_term) + gross/k
        (q_j + delta)**2 = newSumTerm - sum_term + q_j**2
    so delta = sqrt(newSumTerm - sum_term + q_j**2) - q_j.

    Round-trips with ``buy_cost`` to floating-point precision.
    """
    if spend <= 0.0:
        return 0.0
    st = sum_term(q)
    gross = spend * (1.0 - fee)
    new_root = math.sqrt(st) + gross / k
    new_st = new_root * new_root
    inner = new_st - st + q[j] * q[j]
    if inner < 0.0:
        inner = 0.0
    return math.sqrt(inner) - q[j]


def shares_for_sell_proceeds(
    q: Sequence[float], k: float, j: int, proceeds: float, fee: float = 0.0
) -> float:
    """Inverse of :func:`sell_proceeds`: shares of ``j`` to sell for ~``proceeds`` tokens.

    Closed-form, clamped to ``[0, q_j]`` (you cannot drive an outcome's shares
    negative). With gross = proceeds/(1-fee):
        sqrt(newSumTerm) = sqrt(sum_term) - gross/k
        (q_j - delta)**2 = newSumTerm - sum_term + q_j**2
    so delta = q_j - sqrt(newSumTerm - sum_term + q_j**2).
    """
    if proceeds <= 0.0:
        return 0.0
    st = sum_term(q)
    gross = proceeds / (1.0 - fee)
    new_root = math.sqrt(st) - gross / k
    if new_root < 0.0:
        new_root = 0.0
    new_st = new_root * new_root
    inner = new_st - st + q[j] * q[j]
    if inner < 0.0:
        inner = 0.0
    delta = q[j] - math.sqrt(inner)
    if delta < 0.0:
        return 0.0
    if delta > q[j]:
        return q[j]
    return delta


def settlement_payout(pool: float, winning_shares: float, total_winning_shares: float) -> float:
    """Pure-parimutuel redemption: pool * (winning_shares / total_winning_shares).

    Winners split the whole pool pro-rata by winning shares (rounded down
    on-chain; ``unclaimedWinningShares`` decrements as people redeem, making it
    order-independent).
    """
    if total_winning_shares <= 0.0:
        return 0.0
    return pool * (winning_shares / total_winning_shares)


def liquidation_reward(q: Sequence[float], k: float, held: Sequence[float]) -> float:
    """Recovery for an expired/unsettled market with no winner.

    liquidatorReward = k * Sigma(held q_i) / sqrt(Sigma q_j**2)
    i.e. you recover the spot value of the shares you hold.
    """
    st = sum_term(q)
    if st <= 0.0:
        return 0.0
    return k * sum(held) / math.sqrt(st)
