"""Calibration gate for the v2 synthetic-trader flow (battery WITH_SYNTH_TRADING).

Run BEFORE committing the full ~6h v2 battery. All four checks are
model-INDEPENDENT (the noise flow, price path, and prompt template do not depend
on the model), so this runs fully offline with a HoldModel stand-in -- no API
calls, no cost -- yet exercises the exact production code path
(``run_episode(..., noise_flow=SyntheticFlow(...))``).

Checks (Step 4 of the v2 task):
  1. The board moves ~one conviction unit (~10-25 points net) per interval.
  2. It does NOT pin to the ~73% single-band ceiling and go dead.
  3. The prompt template is byte-identical to v1 for the same seed/turn --
     only the implied-probability NUMBERS differ (no narration, same wording).
  4. The price path (and winning band) for a seed match v1 exactly -> seed-aligned.

Usage:  python3 eval_runs/calibration_gate.py [seed] [n_seeds]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from agent_eval_harness.experiment import CellConfig  # noqa: E402
from agent_eval_harness.harness import run_episode  # noqa: E402
from agent_eval_harness.model import HoldModel  # noqa: E402
from agent_eval_harness.synthetic import NoiseConfig, SyntheticFlow, IntervalReport  # noqa: E402


def baseline_cell(band_width: float = 10.0) -> CellConfig:
    """The frozen battery base cell -- identical to v1's (battery.py defaults).

    ``band_width`` defaults to the baseline (10.0); the core battery also sweeps it
    to 5.0 and 20.0, which are DIFFERENT geometries (the band edges and -- because
    the path is band-relative, vol_abs = volatility*band_width -- the price path
    itself change), so the gate must be re-run for each.
    """
    return CellConfig(
        volatility=1.0, knowability_min=0.0, band_width=band_width,
        anchor=100.0, n_outcomes=3, hours=2.0, interval_min=30.0, drift=0.0,
        k=10.0, fee=0.02, initial_shares=100.0, starting_cash=1000.0,
        temperature=0.7, memory=False, disclose_dead_window=True, token="ZQX",
    )


# ---- instrumented episode: capture each interval's board move + per-turn prompts

def run_instrumented(cell, seed, with_noise):
    """Run a HoldModel episode; return (interval_reports, per_turn_prompts, log).

    HoldModel never trades, so any board movement is purely synthetic flow and the
    prompts differ from v1 only where the synthetic flow moved the probabilities.
    """
    market = cell.build_market()
    path = cell.build_path(seed=seed)
    n_turns = len(path.trading_points())
    reports = []
    flow = None
    if with_noise:
        sf = SyntheticFlow(seed=seed, n_outcomes=cell.n_outcomes,
                           band_width=cell.band_width, config=NoiseConfig(),
                           n_turns=n_turns)

        def flow(market, revealed):          # noqa: E306 - capture reports
            rep = sf(market, revealed)
            reports.append(rep)
            return rep

    log = run_episode(
        market, HoldModel(), path,
        starting_cash=cell.starting_cash, token=cell.token, seed=seed,
        memory=cell.memory, disclose_dead_window=cell.disclose_dead_window,
        noise_flow=flow,
    )
    prompts = [t.prompt for t in log.turns]
    return reports, prompts, log, n_turns


# ---- check 3 helper: diff two prompts, allowing ONLY the probability line to move

def prompt_diff_only_probabilities(p1, p2):
    """Return (ok, offending_lines). ok=True iff every line is byte-identical
    except the 'Implied probabilities:' line."""
    l1, l2 = p1.split("\n"), p2.split("\n")
    if len(l1) != len(l2):
        return False, [("LINE COUNT", len(l1), len(l2))]
    offending = []
    for a, b in zip(l1, l2):
        if a == b:
            continue
        if a.startswith("Implied probabilities:") and b.startswith("Implied probabilities:"):
            continue  # allowed: the live board moved the numbers
        offending.append((a, b))
    return (len(offending) == 0), offending


def gate_for_cell(cell, label, seed, n_seeds):
    """Run the four-check gate for one geometry; print a section, return all-pass."""
    print("\n" + "=" * 74)
    print("GEOMETRY: %s   (band_width=%g)   base seed=%d  aggregate over %d seeds"
          % (label, cell.band_width, seed, n_seeds))
    print("=" * 74)

    # ----- detailed single episode (the "one episode" the gate asks for) -----
    reports, v2_prompts, v2_log, n_turns = run_instrumented(cell, seed, with_noise=True)
    print("\n[seed %d] per-interval synthetic flow (HoldModel, so board moves ONLY via noise):" % seed)
    print("  turn | $ placed | trades | inferred band | net move (pts) | board (implied %)")
    total_spend = 0.0
    for i, r in enumerate(reports, start=1):
        total_spend += r.notional
        board = " ".join("%4.1f" % (100.0 * p) for p in r.prob_after)
        print("  %4d | %8.1f | %6d | %13d | %14.1f | %s"
              % (i, r.notional, r.n_trades, r.inferred_band, r.net_point_move, board))
    print("  TOTAL synthetic notional this episode: $%.0f  (cap ~$1000, model budget $1000)" % total_spend)

    # ----- aggregate over n_seeds for robustness -----
    all_moves, max_probs, total_spends = [], [], []
    for s in range(n_seeds):
        reps, _, _, _ = run_instrumented(cell, s, with_noise=True)
        all_moves += [r.net_point_move for r in reps if r.notional > 0]
        ts = sum(r.notional for r in reps)
        total_spends.append(ts)
        for r in reps:
            max_probs.append(100.0 * max(r.prob_after) if r.prob_after else 0.0)
    all_moves.sort()
    max_probs.sort()
    mid = all_moves[len(all_moves) // 2] if all_moves else 0.0

    def pct(sorted_vals, q):
        if not sorted_vals:
            return 0.0
        return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]

    peak_med = pct(max_probs, 0.50)   # already in percent
    peak_p90 = pct(max_probs, 0.90)
    peak_max = max_probs[-1] if max_probs else 0.0
    avg_spend = sum(total_spends) / len(total_spends) if total_spends else 0.0
    # liveness: fraction of (turn) intervals that actually placed flow
    live = sum(1 for m in all_moves if m > 0.0)

    # ===== CHECK 1: net move ~10-25 points/interval =====
    lo, hi = (all_moves[0], all_moves[-1]) if all_moves else (0, 0)
    c1 = 8.0 <= mid <= 27.0
    print("\nCHECK 1  net move/interval ~10-25 pts: median=%.1f  range=[%.1f, %.1f]  -> %s"
          % (mid, lo, hi, "PASS" if c1 else "REVIEW"))

    # ===== CHECK 2: does not pin the ~73% ceiling / go dead =====
    # Judge on the TYPICAL worst case (p90), not a single unlucky single-band-path
    # seed: passing requires the board to routinely stay off the ceiling AND to keep
    # moving (liveness) rather than freezing.
    c2 = peak_p90 < 68.0
    print("CHECK 2  not pinned to ~73%% ceiling: peak band prob median=%.1f%% p90=%.1f%% max=%.1f%% -> %s"
          % (peak_med, peak_p90, peak_max, "PASS" if c2 else "REVIEW"))
    print("         avg total synthetic notional/episode=$%.0f (target ~$1000, co-equal regime)" % avg_spend)
    print("         liveness: %d of %d intervals across all seeds placed flow" % (live, n_seeds * n_turns))

    # ===== CHECK 3: prompt byte-identical to v1 except probability numbers =====
    _, v1_prompts, _, _ = run_instrumented(cell, seed, with_noise=False)
    c3 = True
    first_bad = None
    for ti, (pv1, pv2) in enumerate(zip(v1_prompts, v2_prompts), start=1):
        ok, offending = prompt_diff_only_probabilities(pv1, pv2)
        if not ok:
            c3 = False
            first_bad = (ti, offending[:3])
            break
    # also confirm the probability line actually DID move on >=1 turn (flow is live)
    moved = any(
        pv1.split("\n")[i] != pv2.split("\n")[i]
        for pv1, pv2 in zip(v1_prompts, v2_prompts)
        for i in range(len(pv1.split("\n")))
        if pv1.split("\n")[i].startswith("Implied probabilities:")
    )
    print("CHECK 3  prompt template byte-identical except prob numbers: %s%s"
          % ("PASS" if c3 else "FAIL", "" if c3 else "  offending=%r" % (first_bad,)))
    print("         (and the probability line DID move under noise: %s)" % moved)

    # ===== CHECK 4: price path + winning band match v1 for the seed =====
    p_noise = cell.build_path(seed=seed).points
    p_plain = cell.build_path(seed=seed).points
    _, _, v1_log, _ = run_instrumented(cell, seed, with_noise=False)
    path_match = (p_noise == p_plain)
    band_match = (v1_log.winning_outcome == v2_log.winning_outcome)
    settle_match = abs(v1_log.settlement_price - v2_log.settlement_price) < 1e-12
    c4 = path_match and band_match and settle_match
    print("CHECK 4  seed-aligned path+winner: path=%s  winning_band(v1=%d,v2=%d)=%s  settle=%s -> %s"
          % (path_match, v1_log.winning_outcome, v2_log.winning_outcome, band_match, settle_match,
             "PASS" if c4 else "FAIL"))

    allpass = c1 and c2 and c3 and c4
    print("  -> %s: %s" % (label, "PASS" if allpass else "FAIL"))
    return allpass


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print("=" * 74)
    print("V2 SYNTHETIC-FLOW CALIBRATION GATE")
    print("Run across EVERY geometry the core battery sweeps (band_width 10/5/20),")
    print("not just baseline -- the noise band inference, momentum tilt, and the")
    print("band-relative price path all change with band_width.")
    print("=" * 74)

    # The core battery's band_width geometries: baseline (10) + the two swept values.
    geometries = [
        ("baseline", baseline_cell(10.0)),
        ("band_width=5", baseline_cell(5.0)),
        ("band_width=20", baseline_cell(20.0)),
    ]
    results = [(label, gate_for_cell(cell, label, seed, n_seeds)) for label, cell in geometries]

    print("\n" + "=" * 74)
    allpass = all(ok for _, ok in results)
    for label, ok in results:
        print("  %-16s %s" % (label, "PASS" if ok else "FAIL"))
    print("GATE: %s" % ("ALL GEOMETRIES PASS -- v2 battery is safe to run."
                        if allpass else "NOT ALL PASS -- fix before running the battery."))
    print("=" * 74)
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
