"""Master analysis over the full official-v1 battery (52 cells).

Read-only. No model calls. Aggregates every run/cell/condition and prints the
cross-references the master findings file is built on:

  A. PERSONALITY      -- pooled behavioral fingerprint per model x mem.
  B. NOISE FLOOR      -- baseline == vol1.0 == know0 == temp0.7 (4 identical-config
                         draws); their spread is the true within-config noise bar.
  C. MEMORY EFFECT    -- mem-on vs mem-off per model, pooled, permutation p.
  D. AXIS SWEEPS      -- every axis, per model x mem, endpoint perm floor.
  E. DISCOVERY        -- end-state-dependent vs -independent (survivorship recap).
  F. OUTCOMES         -- pnl/win (context only; money is out of scope).

Usage: python3 master_analysis.py [session_dir]
"""
import json
import os
import statistics
import sys
import random

random.seed(0)
N_PERM = 20000
GPT, CLA = "openai-gpt-4o", "anthropic-claude-sonnet-4"
MODELS = [GPT, CLA]
MEMS = ["off", "on"]

BASELINE = "axis-baseline__val-default__n30"
AXES = {
    "volatility": [("0.5", "0.5"), ("1.0", "1"), ("1.5", "1.5"), ("2.0", "2")],
    "temperature": [("0.0", "0"), ("0.7", "0.7"), ("1.2", "1.2")],
    "knowability_min": [("0", "0"), ("60", "60"), ("120", "120")],
    "band_width": [("5", "5"), ("20", "20")],
}
# the four cells that are config-identical to baseline (independent draws)
SAME_AS_BASELINE = [
    ("baseline", BASELINE),
    ("vol=1.0", "axis-volatility__val-1__n30"),
    ("know=0", "axis-knowability_min__val-0__n30"),
    ("temp=0.7", "axis-temperature__val-0.7__n30"),
]


def cell_path(model, mem, cell):
    return "%s__mem-%s__%s" % (model, mem, cell)


def load(session, model, mem, cell):
    p = os.path.join(session, cell_path(model, mem, cell), "runs.jsonl")
    if not os.path.isfile(p):
        return None
    return [json.loads(l)["episode"] for l in open(p)]


def all_core_cells():
    cells = [BASELINE]
    for ax, vals in AXES.items():
        for _, v in vals:
            cells.append("axis-%s__val-%s__n30" % (ax, v))
    return sorted(set(cells))


# ---------- per-run metric extractors ----------
def fp(ep, k):
    return ep["footprint"].get(k)


# NOTE / data-hygiene trap: the parser ZEROES parsed["outcome"] on hold turns, so
# parsed["outcome"] == 0 on any non-buy/sell turn is meaningless (not "band 0").
# Read t["planned_order"]["outcome"] and gate on executed buys/sells, never trust
# parsed["outcome"] on a hold. Nothing here is corrupted (buy_flow uses
# planned_order + executed buys; the top_pick family reads positions), but any
# future metric that reads parsed["outcome"] on non-trade turns will silently see 0.
def exec_sizes(ep):
    return [t["parsed"]["size"] for t in ep["turns"]
            if t["parsed"]["action"] in ("buy", "sell") and t["executed"]]


def held_at_bell(ep):
    """1.0 if the model holds ANY position at settlement, else 0.0 (= not exit-flat).

    win is only possible when held_at_bell == 1 (exiting flat pays 0), so
    win / held_at_bell = P(win | held) exactly -- the accuracy factor, stripped of
    the commitment factor. Decomposing win into held_rate x conditional-accuracy is
    the same end-state-conditioning fix applied to `top_pick` (see F8); a bare `win`
    confounds "commits more often" with "picks better".
    """
    return 0.0 if sum(abs(x) for x in ep["closing_position"]) <= 1e-9 else 1.0


def exit_flat(ep):
    return 1.0 if sum(abs(x) for x in ep["closing_position"]) <= 1e-9 else 0.0


def top_pick(ep):
    pos = list(ep["closing_position"]); w = ep["winning_outcome"]
    if sum(abs(x) for x in pos) <= 1e-9 or not (0 <= w < len(pos)):
        return None
    return 1.0 if sum(1 for x in pos if x > pos[w] + 1e-12) == 0 else 0.0


def peak_top_pick(ep):
    w = ep["winning_outcome"]
    states = [list(t["position_before"]) for t in ep["turns"]] + [list(ep["closing_position"])]
    bg, bp = -1.0, None
    for pos in states:
        g = sum(abs(x) for x in pos)
        if g > bg + 1e-12:
            bg, bp = g, pos
    if bp is None or bg <= 1e-9 or not (0 <= w < len(bp)):
        return None
    return 1.0 if sum(1 for x in bp if x > bp[w] + 1e-12) == 0 else 0.0


def buy_flow(ep):
    w = ep["winning_outcome"]; flow = {}; tot = 0.0
    for t in ep["turns"]:
        if t["parsed"]["action"] == "buy" and t["executed"]:
            o = t["planned_order"].get("outcome", t["parsed"]["outcome"])
            flow[o] = flow.get(o, 0.0) + t["notional"]; tot += t["notional"]
    return None if tot <= 1e-9 else flow.get(w, 0.0) / tot


BEHAVIOR = [
    ("trade_count", lambda e: float(fp(e, "trade_count"))),
    ("hold_rate", lambda e: fp(e, "n_holds") / fp(e, "n_turns")),
    ("total_volume", lambda e: fp(e, "total_volume")),
    ("median_trade_size", lambda e: fp(e, "median_trade_size")),
    ("direction_switches", lambda e: float(fp(e, "direction_switches"))),
    ("outcomes_traded", lambda e: float(fp(e, "outcomes_traded"))),
    ("first_trade_frac", lambda e: fp(e, "first_trade_frac")),
    ("exit_flat_rate", exit_flat),
]
OUTCOME = [("win", lambda e: float(fp(e, "win"))),
           ("return_pct", lambda e: fp(e, "return_pct")),
           ("pnl", lambda e: fp(e, "pnl"))]


def vals(eps, fn):
    return [v for v in (fn(e) for e in eps) if v is not None]


def pooled(session, model, mem, fn, cells=None):
    cells = cells or all_core_cells()
    out = []
    for c in cells:
        eps = load(session, model, mem, c)
        if eps:
            out += vals(eps, fn)
    return out


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def perm2(a, b):
    if not a or not b:
        return float("nan"), float("nan")
    obs = mean(b) - mean(a); pool = a + b; nb = len(b)
    idx = list(range(len(pool))); hit = 0
    for _ in range(N_PERM):
        random.shuffle(idx)
        g = mean([pool[i] for i in idx[:nb]]) - mean([pool[i] for i in idx[nb:]])
        if abs(g) >= abs(obs) - 1e-12:
            hit += 1
    return obs, hit / N_PERM


# =================== A. PERSONALITY ===================
def section_personality(session):
    print("=" * 80)
    print("A. PERSONALITY -- pooled behavioral fingerprint (all 13 core cells)")
    print("=" * 80)
    print("  %-20s %10s %10s   %10s %10s" % ("metric", "gpt off", "gpt on", "cla off", "cla on"))
    for name, fn in BEHAVIOR + OUTCOME:
        row = "  %-20s" % name
        for model in MODELS:
            for mem in MEMS:
                row += " %10.2f" % mean(pooled(session, model, mem, fn))
            if model == GPT:
                row += "  "
        print(row)
    print()


# =================== B. NOISE FLOOR ===================
def section_noise_floor(session):
    print("=" * 80)
    print("B. NOISE FLOOR -- 4 config-IDENTICAL draws (baseline=vol1=know0=temp0.7)")
    print("   spread across these 4 same-config cells = the real within-config noise.")
    print("   Any axis effect must beat THIS to be real.")
    print("=" * 80)
    keymets = [("trade_count", BEHAVIOR[0][1]), ("hold_rate", BEHAVIOR[1][1]),
               ("total_volume", BEHAVIOR[2][1]), ("median_trade_size", BEHAVIOR[3][1]),
               ("direction_switches", BEHAVIOR[4][1])]
    for model in MODELS:
        for mem in MEMS:
            print("  -- %s mem-%s --" % (model, mem))
            for name, fn in keymets:
                ms = []
                for _, cell in SAME_AS_BASELINE:
                    eps = load(session, model, mem, cell)
                    ms.append(mean(vals(eps, fn)) if eps else float("nan"))
                spread = (max(ms) - min(ms)) if all(m == m for m in ms) else float("nan")
                print("     %-20s %s   spread=%.2f"
                      % (name, " ".join("%8.2f" % m for m in ms), spread))
            print()


# =================== C. MEMORY EFFECT ===================
def section_memory(session):
    print("=" * 80)
    print("C. MEMORY EFFECT -- mem-on vs mem-off, pooled over all core cells")
    print("   gap = on - off; perm p two-sided. <== sig if p<0.05")
    print("=" * 80)
    for model in MODELS:
        print("  -- %s --" % model)
        for name, fn in BEHAVIOR + [("top_pick", top_pick), ("peak_top_pick", peak_top_pick),
                                    ("buy_flow", buy_flow)] + OUTCOME:
            off = pooled(session, model, "off", fn)
            on = pooled(session, model, "on", fn)
            obs, p = perm2(off, on)
            flag = "  <== sig" if p == p and p < 0.05 else ""
            print("     %-20s off=%8.2f  on=%8.2f  gap=%+8.2f [p=%.3f]%s"
                  % (name, mean(off), mean(on), obs, p, flag))
        print()


# =================== D. AXIS SWEEPS ===================
def section_axes(session):
    print("=" * 80)
    print("D. AXIS SWEEPS -- endpoint permutation floor per model x mem")
    print("   For each behavior metric: series across the axis + perm p on first->last.")
    print("   <== sig means the endpoint move beats within-cell noise (p<0.05).")
    print("=" * 80)
    sweepmets = [("trade_count", BEHAVIOR[0][1]), ("hold_rate", BEHAVIOR[1][1]),
                 ("total_volume", BEHAVIOR[2][1]), ("median_trade_size", BEHAVIOR[3][1]),
                 ("direction_switches", BEHAVIOR[4][1]), ("outcomes_traded", BEHAVIOR[5][1])]
    for ax, labels in AXES.items():
        print("  #### axis = %s ####" % ax)
        for model in MODELS:
            for mem in MEMS:
                cells = [("axis-%s__val-%s__n30" % (ax, v)) for _, v in labels]
                series_eps = [load(session, model, mem, c) for c in cells]
                if any(e is None for e in series_eps):
                    continue
                hits = []
                for name, fn in sweepmets:
                    seq = [mean(vals(e, fn)) for e in series_eps]
                    a = vals(series_eps[0], fn); b = vals(series_eps[-1], fn)
                    _, p = perm2(a, b)
                    tag = "%s:%s" % (name, "/".join("%.2f" % s for s in seq))
                    if p == p and p < 0.05:
                        tag += "[p=%.3f*]" % p
                    hits.append(tag)
                print("     %-22s mem-%s: %s" % (model, mem, "  ".join(hits)))
        print()


# =================== E. DISCOVERY ===================
def section_discovery(session):
    print("=" * 80)
    print("E. DISCOVERY -- end-state-dependent (top_pick) vs -independent")
    print("   (peak_top_pick, buy_flow). chance=0.33. Pooled core cells.")
    print("   gap = claude - gpt; if edge survives on independent metrics it's real.")
    print("=" * 80)
    for mem in MEMS:
        print("  -- mem-%s --" % mem)
        for name, fn in [("top_pick(end)", top_pick), ("peak_top_pick", peak_top_pick),
                         ("buy_flow", buy_flow)]:
            g = pooled(session, GPT, mem, fn); c = pooled(session, CLA, mem, fn)
            obs, p = perm2(g, c)
            flag = "  <== sig" if p == p and p < 0.05 else ""
            print("     %-16s gpt n=%-3d %.3f   cla n=%-3d %.3f   gap=%+.3f [p=%.3f]%s"
                  % (name, len(g), mean(g), len(c), mean(c), obs, p, flag))
        print()


def section_win_decomp(session):
    print("=" * 80)
    print("F. WIN DECOMPOSITION -- win = held_rate x conditional-accuracy")
    print("   `win` (held the winning band at the bell) is END-STATE-CONDITIONED, the")
    print("   same bias killed in F8/top_pick. Split it: P(win)=P(held)*P(win|held).")
    print("   If P(win|held) is flat across the memory split, the win gain is pure")
    print("   COMMITMENT (more terminal positions), not better accuracy.")
    print("=" * 80)
    print("  %-26s %8s %8s   %8s %8s" % ("", "gpt off", "gpt on", "cla off", "cla on"))
    rows = []
    cells = all_core_cells()
    grid = {}
    for model in MODELS:
        for mem in MEMS:
            wins = pooled(session, model, mem, lambda e: float(fp(e, "win")))
            held = pooled(session, model, mem, held_at_bell)
            # conditional accuracy: win among runs that held a position
            cond = []
            for c in cells:
                eps = load(session, model, mem, c)
                if not eps:
                    continue
                cond += [float(fp(e, "win")) for e in eps if held_at_bell(e) == 1.0]
            grid[(model, mem)] = (mean(held), mean(wins),
                                  (mean(cond) if cond else float("nan")), len(cond))
    for label, key in [("held_rate P(held)", 0), ("win P(win)", 1),
                       ("cond-accuracy P(win|held)", 2)]:
        row = "  %-26s" % label
        for model in MODELS:
            for mem in MEMS:
                row += " %8.3f" % grid[(model, mem)][key]
            if model == GPT:
                row += "  "
        print(row)
    print("  n(held runs):              " + "".join(
        "%8d " % grid[(m, mem)][3] + ("  " if m == GPT and mem == "on" else "")
        for m in MODELS for mem in MEMS))
    print()


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "2026-06-27_official-v1"
    here = os.path.dirname(os.path.abspath(__file__))
    session = os.path.join(here, session)
    print("\nMASTER ANALYSIS  session=%s  N_PERM=%d\n" % (os.path.basename(session), N_PERM))
    section_personality(session)
    section_noise_floor(session)
    section_memory(session)
    section_axes(session)
    section_discovery(session)
    section_win_decomp(session)


if __name__ == "__main__":
    main()
