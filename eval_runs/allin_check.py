"""All-in behavior: v1 (frozen, no-synth control) vs v2 (live-board treatment).

Read-only. Built to answer the two disciplined questions raised after noticing
all-in cash rejections in v2 -- WITHOUT prematurely calling it a finding:

  CHECK 1 -- INTENT, not rejections. An "all-in attempt" = a BUY whose requested
    size is >= theta * available cash, counted whether it FILLED or was rejected
    (a rejection is just an all-in that clipped the 2% fee; counting only
    rejections measures a rounding artifact, not the behavior). Compare the
    attempt RATE in v1 vs v2 on the SAME model/memory/cells/seeds. If equal,
    all-in is just v1 behavior we happened to notice; if v2 is materially higher,
    there is something to explain.

  CHECK 2 -- A vs B (only meaningful if v2 is elevated). For each v2 all-in, does
    its target band FOLLOW the board the synthetic flow has been pushing toward?
      Reading A (signal-chasing, interesting): all-in aligns with recent board /
        synth drift -> the model is reading counterparty flow as information.
      Reading B (instability, artifact): all-in direction is uncorrelated with the
        drift -> the moving board just destabilized its sizing.
    We report alignment of the all-in's target band with (a) the current favorite,
    (b) the observed board drift since its last turn, (c) the pure synthetic drift
    (board move between the board the model LEFT and the board it now SEES), each
    vs the 1/3 chance baseline, AND the same alignment for non-all-in buys as a
    within-model control (so "follows the leader" isn't just its general policy).

Usage:  python3 eval_runs/allin_check.py [model] [mem] [theta]
  e.g.  python3 eval_runs/allin_check.py openai-gpt-4o off 0.9
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, "battery_WITHOUT_SYNTH_TRADING/2026-06-27_official-v1")
V2 = os.path.join(HERE, "battery_WITH_SYNTH_TRADING/2026-06-29_official-v2")


def load_cells(session, model, mem):
    """cellname -> [episode dicts] for every COMPLETE cell of this model/memory."""
    out = {}
    for d in sorted(glob.glob(os.path.join(session, "%s__mem-%s__*" % (model, mem)))):
        rj = os.path.join(d, "runs.jsonl")
        if not os.path.isfile(rj):
            continue
        eps = [json.loads(l) for l in open(rj) if l.strip()]
        if len(eps) >= 30:
            out[os.path.basename(d)] = eps
    return out


def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def is_all_in(turn, theta):
    p = turn["parsed"]
    if p.get("action") != "buy":
        return False
    size = p.get("size")
    cash = turn.get("cash_before") or 0.0
    return size is not None and cash > 0 and size >= theta * cash


def left_board(turns, i):
    """The board the model LEFT after turn i (= probs after its trade, or the
    board it saw if it didn't move it)."""
    t = turns[i]
    tr = t.get("trade_result")
    if t.get("executed") and tr and tr.get("probabilities_after"):
        return tr["probabilities_after"]
    return t["probabilities_before"]


# ---------------------------------------------------------------- check 1

def attempt_stats(cells, theta):
    n_turns = n_buys = n_allin = 0
    eps_with_allin = total_eps = 0
    per_cell = {}
    for name, eps in cells.items():
        c_allin = 0
        for e in eps:
            total_eps += 1
            had = False
            for t in e["episode"]["turns"]:
                n_turns += 1
                if t["parsed"].get("action") == "buy":
                    n_buys += 1
                if is_all_in(t, theta):
                    n_allin += 1
                    c_allin += 1
                    had = True
            if had:
                eps_with_allin += 1
        per_cell[name] = c_allin
    return {
        "turns": n_turns, "buys": n_buys, "allin": n_allin,
        "eps": total_eps, "eps_with_allin": eps_with_allin, "per_cell": per_cell,
    }


# ---------------------------------------------------------------- check 2

def direction_stats(cells, theta):
    """For v2: alignment of all-in (and, as control, non-all-in buys) target band
    with favorite / board-drift / synth-drift."""
    def blank():
        return {"n": 0, "fav": 0, "board": 0, "synth": 0, "n_drift": 0}
    allin, other = blank(), blank()
    for eps in cells.values():
        for e in eps:
            turns = e["episode"]["turns"]
            for i, t in enumerate(turns):
                if t["parsed"].get("action") != "buy":
                    continue
                tgt = t["parsed"].get("outcome")
                pb = t["probabilities_before"]
                if tgt is None or not (0 <= tgt < len(pb)):
                    continue
                bucket = allin if is_all_in(t, theta) else other
                bucket["n"] += 1
                if tgt == argmax(pb):
                    bucket["fav"] += 1
                if i >= 1:
                    bucket["n_drift"] += 1
                    prev = turns[i - 1]["probabilities_before"]
                    if tgt == argmax([pb[j] - prev[j] for j in range(len(pb))]):
                        bucket["board"] += 1
                    lb = left_board(turns, i - 1)
                    if tgt == argmax([pb[j] - lb[j] for j in range(len(pb))]):
                        bucket["synth"] += 1
    return allin, other


def pct(a, b):
    return (100.0 * a / b) if b else float("nan")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "openai-gpt-4o"
    mem = sys.argv[2] if len(sys.argv) > 2 else "off"
    theta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9

    c1 = load_cells(V1, model, mem)
    c2 = load_cells(V2, model, mem)
    common = sorted(set(c1) & set(c2))
    c1 = {k: c1[k] for k in common}
    c2 = {k: c2[k] for k in common}

    print("=" * 78)
    print("ALL-IN CHECK   model=%s  mem-%s   (NOT a finding -- numbers only)" % (model, mem))
    print("matched cells (complete in BOTH arms, same seeds): %d" % len(common))
    print("all-in attempt = BUY with size >= %.0f%% of available cash (filled OR rejected)"
          % (100 * theta))
    print("=" * 78)
    if not common:
        print("No cells complete in BOTH arms yet for this model/mem -- come back when they finish.")
        return 0

    # ---- CHECK 1: attempt rate v1 vs v2 (matched) ----
    print("\nCHECK 1 -- all-in ATTEMPT rate (intent), v1 control vs v2 treatment:")
    print("  arm | all-in | buys | turns | %of buys | %of turns | episodes w/ >=1 all-in")
    rows = []
    for label, cells in (("v1", c1), ("v2", c2)):
        for th in (theta, 0.95):
            pass
        s = attempt_stats(cells, theta)
        rows.append((label, s))
        print("  %3s | %6d | %4d | %5d | %7.1f%% | %8.1f%% | %d/%d"
              % (label, s["allin"], s["buys"], s["turns"],
                 pct(s["allin"], s["buys"]), pct(s["allin"], s["turns"]),
                 s["eps_with_allin"], s["eps"]))
    v1s, v2s = rows[0][1], rows[1][1]
    print("  delta v2-v1: all-in %+d attempts  (%+.1f pts of buys, %+.1f pts of turns)"
          % (v2s["allin"] - v1s["allin"],
             pct(v2s["allin"], v2s["buys"]) - pct(v1s["allin"], v1s["buys"]),
             pct(v2s["allin"], v2s["turns"]) - pct(v1s["allin"], v1s["turns"])))
    # per-cell so a single cell can't masquerade as a trend
    print("  per-cell all-in attempts (v1 -> v2):")
    for name in common:
        print("    %-40s %d -> %d" % (name.replace("%s__mem-%s__" % (model, mem), ""),
                                      v1s["per_cell"][name], v2s["per_cell"][name]))

    # ---- CHECK 2: direction correlation (v2 only) ----
    allin, other = direction_stats(c2, theta)
    print("\nCHECK 2 -- v2 all-in DIRECTION (A=follows board/signal vs B=uncorrelated/noise):")
    print("  chance baseline for 3 bands = 33.3%%")
    print("  group            |  N  | == favorite | == board-drift | == synth-drift")
    for label, g in (("all-in buys", allin), ("other buys (ctrl)", other)):
        print("  %-16s | %3d | %9.1f%% | %12.1f%% | %12.1f%%"
              % (label, g["n"], pct(g["fav"], g["n"]),
                 pct(g["board"], g["n_drift"]), pct(g["synth"], g["n_drift"])))
    print("\n  (read: all-in alignment >> 33%% AND >> the 'other buys' control -> Reading A")
    print("   (signal-chasing); all-in alignment ~33%% / ~ control -> Reading B (instability).")
    print("   N is small -- treat as directional, not conclusive; Claude is the real test.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
