"""Market-depth / conviction-cost tables for interpreting the v2 (live-board) arm.

Two tables, both computed straight from the verified DPM math (``dpm.math``) on
the battery's constants (k=10, seed q0=[100,100,100], 2% fee, 3 bands):

  TABLE A -- SEED-RELATIVE depth. Starting from the untouched seed pool, how many
    dollars does it take to push ONE band to a target implied probability, and what
    is the marginal $/point along the way. This is the turn-1 picture (what a trade
    costs when nobody has moved the board yet).

  TABLE B -- MARGINAL depth CONDITIONED ON THE BAND ALREADY BEING PUSHED. In a live
    v2 episode the synthetic traders and the model both move the same band, so by
    turn 3 the board is already skewed and the marginal cost the model faces is NOT
    the turn-1 seed picture. For a band already sitting at p0 in {33,45,55,65}%,
    this shows the marginal $/point to push it a little further.

Why table B matters for analysis: it lets us later separate "the model traded less
in v2 because it deferred to the signal" from "the model traded less because the
board was already pushed and the next point of conviction got expensive." Have it
in hand before reading the v2-vs-v1 trade-size deltas.

Usage:  python3 eval_runs/depth_table.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from agent_eval_harness.dpm import math as dpm  # noqa: E402

K = 10.0
SEED_Q = [100.0, 100.0, 100.0]
FEE = 0.02
BAND = 0  # the band we push (symmetric, so any single band is representative)


def implied(q):
    return dpm.implied_probabilities(q)


def cost_to_reach_prob(q, k, j, target_p, fee):
    """Tokens to buy enough of band j to raise its implied prob from current to
    ``target_p``. Closed-form: implied_p = q_j^2 / sum_term, so solve for the q_j
    that hits target_p with the OTHER bands held fixed, then price that buy."""
    others = sum(q[i] * q[i] for i in range(len(q)) if i != j)
    if target_p >= 1.0:
        return float("inf"), float("inf")
    # q_j^2 / (q_j^2 + others) = target_p  ->  q_j = sqrt(target_p*others/(1-target_p))
    qj_target = math.sqrt(target_p * others / (1.0 - target_p))
    delta = qj_target - q[j]
    if delta <= 0.0:
        return 0.0, qj_target
    cost = dpm.buy_cost(q, k, j, delta, fee)
    return cost, qj_target


def fmt_money(x):
    if x == float("inf"):
        return "  unreachable"
    return "$%9.2f" % x


def table_a():
    print("=" * 78)
    print("TABLE A -- SEED-RELATIVE depth (turn-1 board, q0=[100,100,100], k=10, fee=2%)")
    print("=" * 78)
    p0 = implied(SEED_Q)
    pool = dpm.collateral(SEED_Q, K)
    print("seed pool C(q) = %.2f tokens   |   each band starts at %.1f%%   |   budget $1000 = %.1f%% of pool"
          % (pool, 100.0 * p0[BAND], 100.0 * 1000.0 / pool))
    print()
    print("  target band prob |   total $ to get there |   $ / point (avg from seed)")
    print("  -----------------+------------------------+----------------------------")
    prev_p = 100.0 * p0[BAND]
    for target in (0.40, 0.45, 0.50, 0.5774, 0.60, 0.6667, 0.70, 0.7280):
        cost, _ = cost_to_reach_prob(SEED_Q, K, BAND, target, FEE)
        pts = 100.0 * target - 100.0 * p0[BAND]
        per_pt = (cost / pts) if pts > 0 else float("inf")
        note = ""
        if abs(target - 0.7280) < 1e-3:
            note = "  <- ~full $1000 budget ceiling"
        if abs(target - 0.5774) < 1e-3:
            note = "  <- spot price = k/sqrt(3)"
        print("        %5.1f%%       | %s            | %s%s"
              % (100.0 * target, fmt_money(cost), fmt_money(per_pt).strip(), note))


def table_b():
    print()
    print("=" * 78)
    print("TABLE B -- MARGINAL depth CONDITIONED ON THE BAND ALREADY BEING PUSHED")
    print("=" * 78)
    print("For a band already at p0 (others split the rest evenly), the marginal cost")
    print("to push it +5 points further -- i.e. what the model faces mid-episode in v2")
    print("once synthetic flow + its own trades have already skewed the board.")
    print()
    print("  band already at | marginal $ for next +5 pts | marginal $/point")
    print("  ----------------+----------------------------+------------------")
    for p0 in (1.0 / 3.0, 0.45, 0.55, 0.65):
        # Build a q with band 0 at p0 and the other two bands equal, total scale
        # matched to the seed pool's sum_term so the depth is comparable.
        others_each = math.sqrt((1.0 - p0) / 2.0)
        q0_band = math.sqrt(p0)
        # scale so sum_term matches the seed (3*100^2 = 30000) -> comparable depth
        scale = math.sqrt(30000.0 / (q0_band ** 2 + 2 * others_each ** 2))
        q = [q0_band * scale, others_each * scale, others_each * scale]
        cur = implied(q)[0]
        target = min(cur + 0.05, 0.9999)
        cost, _ = cost_to_reach_prob(q, K, 0, target, FEE)
        pts = 100.0 * (target - cur)
        per_pt = (cost / pts) if pts > 0 else float("inf")
        print("       %5.1f%%      | %s               | %s"
              % (100.0 * cur, fmt_money(cost), fmt_money(per_pt).strip()))
    print()
    print("Read: the marginal $/point RISES as the band climbs (convex cost curve),")
    print("so a v2 model arriving at an already-pushed board pays more per point than")
    print("the turn-1 seed picture in Table A -- distinguish 'deferred to signal' from")
    print("'trading got expensive' using this, not Table A.")


def main():
    table_a()
    table_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
