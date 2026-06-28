"""Read-only diagnostics that turn this battery's two leads into verdicts.

Runs four checks across the finished mem-off cells of a battery session, with NO
writes and NO model calls. Each check targets a specific alternative explanation
for the knowability->discovery drop, plus the still-open volatility within-label
adjudication.

  (1) TOP_PICK SCORABILITY  -- survivorship. top_pick is undefined when a run took
      no position. If the dead window changes how often a run is *scorable*, the
      top_pick drop at 120 could be a change in WHICH runs survive to be scored,
      not accuracy among comparable runs. Reports n / undefined per cell.

  (2) OUTCOME KNOWABILITY (model-independent) -- the path is generated THROUGH the
      dead window, so settlement = price at (close + knowability_min). A longer
      window = more unseen post-close diffusion. This shifts the winning-bucket
      distribution AND decorrelates the close price from the winner. Reports, per
      knowability cell (identical across models -- same seeds, same path): the
      winning-bucket distribution, the mean |settlement - close| move, and how
      often the close-price band == the winning band ("could ANYONE have known
      from observable info"). If outcome-knowability collapses at 120, a top_pick
      drop is the MARKET moving the target out of view, not model skill degrading.

  (3) PERMUTATION FLOOR on the top_pick drop -- among SCORABLE runs only (so it is
      not contaminated by (1)), shuffle the cell label many times and ask how often
      the mean-top_pick gap is at least as large as observed. baseline-vs-k0 (same
      config, independent draw) is the built-in null sanity check.

  (4) WITHIN-LABEL volatility dispersion (claude) -- the still-unrun adjudication
      of the volatility null. Bins one volatility label's episodes by realized
      range (calm vs wild) and tests the size-dispersion / trade-count gap against
      a permutation floor. Until this clears, "claude is vol-insensitive" is the
      label-sweep inference already invalidated for gpt-4o.

Usage:
    python3 knob_diagnostics.py [session_dir] [mem]
    e.g. python3 knob_diagnostics.py 2026-06-27_official-v1 off
"""
import json
import os
import re
import statistics
import sys
import random

random.seed(0)
N_PERM = 20000

GPT = "openai-gpt-4o"
CLA = "anthropic-claude-sonnet-4"

KNOW_CELLS = [
    ("baseline", "axis-baseline__val-default__n30"),
    ("k=0", "axis-knowability_min__val-0__n30"),
    ("k=60", "axis-knowability_min__val-60__n30"),
    ("k=120", "axis-knowability_min__val-120__n30"),
]
VOL_LABELS = [("0.5", "0.5"), ("1.0", "1"), ("1.5", "1.5"), ("2.0", "2")]


def load(session, model, mem, cell):
    path = os.path.join(session, "%s__mem-%s__%s/runs.jsonl" % (model, mem, cell))
    if not os.path.isfile(path):
        return None
    return [json.loads(l)["episode"] for l in open(path)]


# ---- per-run recompute of top_pick, IDENTICAL to metrics.compute_discovery ----
def top_pick(ep):
    pos = list(ep["closing_position"])
    w = ep["winning_outcome"]
    total = sum(abs(x) for x in pos)
    if total <= 1e-9 or not (0 <= w < len(pos)):
        return None  # undefined: no position taken (no opinion expressed)
    n_better = sum(1 for x in pos if x > pos[w] + 1e-12)
    return 1.0 if n_better == 0 else 0.0


def observed_prices(ep):
    return [float(x) for x in re.findall(r"min: ([0-9.]+)", ep["turns"][-1]["prompt"])]


def close_price(ep):
    pr = observed_prices(ep)
    return pr[-1] if pr else None


def exec_sizes(ep):
    return [t["parsed"]["size"] for t in ep["turns"]
            if t["parsed"]["action"] in ("buy", "sell") and t["executed"]]


def std(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def band_of(price, edges):
    """edges = sorted interior thresholds; returns band index."""
    b = 0
    for e in edges:
        if price >= e:
            b += 1
    return b


def parse_band_edges(ep):
    """Interior thresholds from the OUTCOMES block, model/seed-independent."""
    txt = ep["turns"][0]["prompt"]
    return sorted(set(float(x) for x in re.findall(r"< ([0-9.]+)\)", txt)))


# ============================ CHECK 1 ============================
def check_scorability(session, mem):
    print("=" * 78)
    print("(1) TOP_PICK SCORABILITY  -- is the 120 drop survivorship?")
    print("    n_scorable = runs that took a position; undefined = runs with no")
    print("    opinion. If n falls at k=120, the drop is partly a subpopulation shift.")
    print("=" * 78)
    hdr = "  %-8s %-22s %-22s" % ("cell", GPT, CLA)
    print(hdr)
    print("  %-8s %-22s %-22s" % ("", "n / undef  mean med", "n / undef  mean med"))
    for label, cell in KNOW_CELLS:
        row = "  %-8s" % label
        for model in (GPT, CLA):
            eps = load(session, model, mem, cell)
            if not eps:
                row += " %-22s" % "(missing)"
                continue
            tps = [top_pick(e) for e in eps]
            sc = [t for t in tps if t is not None]
            undef = sum(1 for t in tps if t is None)
            mean = statistics.mean(sc) if sc else float("nan")
            med = statistics.median(sc) if sc else float("nan")
            row += " %-22s" % ("%2d / %-2d   %.2f %.1f" % (len(sc), undef, mean, med))
        print(row)
    print()


# ============================ CHECK 2 ============================
def check_knowability(session, mem):
    print("=" * 78)
    print("(2) OUTCOME KNOWABILITY (model-independent: same seeds/paths both models)")
    print("    Path runs THROUGH the dead window -> settlement = price at close+k.")
    print("    'close-band hit' = fraction of runs where the band of the LAST OBSERVED")
    print("    price already equals the winning band (could anyone know from data?).")
    print("=" * 78)
    print("  %-8s %-18s %-14s %-16s" % ("cell", "winning bands", "mean|settle-close|",
                                        "close-band hit"))
    for label, cell in KNOW_CELLS:
        eps = load(session, GPT, mem, cell)  # model-independent quantities
        if not eps:
            print("  %-8s (missing)" % label)
            continue
        edges = parse_band_edges(eps[0])
        nb = len(edges) + 1
        wins = [e["winning_outcome"] for e in eps]
        dist = [wins.count(b) for b in range(nb)]
        moves = [abs(e["settlement_price"] - close_price(e)) for e in eps]
        hits = []
        for e in eps:
            cb = band_of(close_price(e), edges)
            hits.append(1.0 if cb == e["winning_outcome"] else 0.0)
        diststr = "/".join(str(d) for d in dist)
        print("  %-8s %-18s %-14.2f %-16s"
              % (label, diststr, statistics.mean(moves),
                 "%.0f%% (%d/%d)" % (100 * statistics.mean(hits),
                                     int(sum(hits)), len(hits))))
    print("  (winning bands listed as band0/band1/band2...; outer bands = wandered far)")
    print()


# ============================ CHECK 3 ============================
def perm_gap(a, b):
    """One-sided permutation p for (mean(b)-mean(a)) being <= observed (a drop)."""
    obs = statistics.mean(b) - statistics.mean(a)
    pool = list(a) + list(b)
    nb = len(b)
    idx = list(range(len(pool)))
    hit = 0
    for _ in range(N_PERM):
        random.shuffle(idx)
        gb = statistics.mean(pool[i] for i in idx[:nb])
        ga = statistics.mean(pool[i] for i in idx[nb:])
        if (gb - ga) <= obs:
            hit += 1
    return obs, hit / N_PERM


def check_permfloor(session, mem):
    print("=" * 78)
    print("(3) PERMUTATION FLOOR on the top_pick drop (SCORABLE runs only)")
    print("    gap = mean_top_pick(hi cell) - mean(lo cell); p = P(perm gap <= obs).")
    print("    baseline-vs-k0 is the NULL sanity (same config) -- expect p high.")
    print("=" * 78)
    for model in (GPT, CLA):
        print("  -- %s --" % model)
        cells = {}
        for label, cell in KNOW_CELLS:
            eps = load(session, model, mem, cell)
            if eps:
                cells[label] = [t for t in (top_pick(e) for e in eps) if t is not None]
        comparisons = [
            ("baseline", "k=0", "NULL sanity (same config)"),
            ("k=0", "k=60", ""),
            ("k=0", "k=120", "the headline drop"),
            ("k=60", "k=120", ""),
        ]
        # pooled 0+60 vs 120
        for lo, hi, note in comparisons:
            if lo in cells and hi in cells:
                obs, p = perm_gap(cells[lo], cells[hi])
                flag = "  <== clears (p<0.05)" if p < 0.05 else ""
                print("     %-6s -> %-6s  gap=%+5.2f  p=%.3f%s  %s"
                      % (lo, hi, obs, p, flag, note))
        if "k=0" in cells and "k=60" in cells and "k=120" in cells:
            pooled = cells["k=0"] + cells["k=60"]
            obs, p = perm_gap(pooled, cells["k=120"])
            flag = "  <== clears (p<0.05)" if p < 0.05 else ""
            print("     %-6s -> %-6s  gap=%+5.2f  p=%.3f%s  pooled low vs longest"
                  % ("0+60", "k=120", obs, p, flag))
        print()


# ============================ CHECK 4 ============================
def within_label(eps):
    ranges = [(max(observed_prices(e)) - min(observed_prices(e))) if observed_prices(e)
              else 0.0 for e in eps]
    sizes = [exec_sizes(e) for e in eps]
    tc = [len(s) for s in sizes]
    med = statistics.median(ranges)
    lo = [i for i, r in enumerate(ranges) if r <= med]
    hi = [i for i, r in enumerate(ranges) if r > med]

    def disp(h, l):
        return std([s for i in h for s in sizes[i]]) - std([s for i in l for s in sizes[i]])

    def tcg(h, l):
        return (statistics.mean(tc[i] for i in h) if h else 0) - \
               (statistics.mean(tc[i] for i in l) if l else 0)

    od, ot = disp(hi, lo), tcg(hi, lo)
    n, k = len(eps), len(hi)
    idx = list(range(n))
    pd = pt = 0
    for _ in range(N_PERM):
        random.shuffle(idx)
        if disp(idx[:k], idx[k:]) >= od:
            pd += 1
        if tcg(idx[:k], idx[k:]) >= ot:
            pt += 1
    return od, pd / N_PERM, ot, pt / N_PERM, len(lo), len(hi)


# ---- end-state-INDEPENDENT discovery metrics (the survivorship fix) ----
def _rank_hit(pos, w):
    """1.0 if winner is a top (no strictly-greater) holding, else 0.0; None if flat."""
    if sum(abs(x) for x in pos) <= 1e-9 or not (0 <= w < len(pos)):
        return None
    return 1.0 if sum(1 for x in pos if x > pos[w] + 1e-12) == 0 else 0.0


def position_states(ep):
    """Full held-position trajectory: state entering each turn + the close."""
    states = [list(t["position_before"]) for t in ep["turns"]]
    states.append(list(ep["closing_position"]))
    return states


def peak_top_pick(ep):
    """Winner == top holding at the turn of MAX gross exposure (peak conviction).

    Defined for any run that ever held a position -- does NOT condition on the
    end state, so it does not vanish when the model exits flat by the bell.
    """
    w = ep["winning_outcome"]
    best_gross, best_pos = -1.0, None
    for pos in position_states(ep):
        g = sum(abs(x) for x in pos)
        if g > best_gross + 1e-12:
            best_gross, best_pos = g, pos
    if best_pos is None or best_gross <= 1e-9:
        return None
    return _rank_hit(best_pos, w)


def buy_flow_on_winner(ep):
    """Fraction of executed BUY notional aimed at the winning band.

    Pure revealed-action flow; defined for any run that bought anything;
    unaffected by whether positions were later unwound. Chance = 1/n_outcomes.
    """
    w = ep["winning_outcome"]
    flow, total = {}, 0.0
    for t in ep["turns"]:
        if t["parsed"]["action"] == "buy" and t["executed"]:
            o = t["planned_order"].get("outcome", t["parsed"]["outcome"])
            amt = t["notional"]
            flow[o] = flow.get(o, 0.0) + amt
            total += amt
    if total <= 1e-9:
        return None
    return flow.get(w, 0.0) / total


def _pooled(session, model, mem, fn):
    """Pool a per-run metric across all core mem cells (defined runs only)."""
    cells = [c for _, c in KNOW_CELLS[1:]] + \
            ["axis-volatility__val-%s__n30" % v for _, v in VOL_LABELS] + \
            ["axis-temperature__val-%s__n30" % v for v in ("0", "0.7", "1.2")] + \
            ["axis-band_width__val-%s__n30" % v for v in ("5", "20")] + \
            ["axis-baseline__val-default__n30"]
    vals = []
    for cell in cells:
        eps = load(session, model, mem, cell)
        if not eps:
            continue
        vals += [v for v in (fn(e) for e in eps) if v is not None]
    return vals


def _perm_two_sample(a, b):
    """Two-sided permutation p for mean(b)-mean(a) (model gap)."""
    obs = statistics.mean(b) - statistics.mean(a)
    pool = a + b
    nb = len(b)
    idx = list(range(len(pool)))
    hit = 0
    for _ in range(N_PERM):
        random.shuffle(idx)
        g = statistics.mean(pool[i] for i in idx[:nb]) - statistics.mean(pool[i] for i in idx[nb:])
        if abs(g) >= abs(obs) - 1e-12:
            hit += 1
    return obs, hit / N_PERM


def check_discovery_robustness(session, mem):
    print("=" * 78)
    print("(5) DISCOVERY ROBUSTNESS -- is 'claude discovers better' a top_pick")
    print("    survivorship artifact? top_pick conditions on END state (vanishes when")
    print("    claude exits flat). peak_top_pick (max-conviction turn) and buy_flow")
    print("    (where buy capital went) do NOT. Pooled over all core cells; chance=0.33.")
    print("    If claude's edge survives on the end-state-independent metrics, it's real.")
    print("=" * 78)
    print("  %-18s %-22s %-22s %-14s" % ("metric", GPT, CLA, "gap [perm p]"))
    for name, fn in [("top_pick(end)", top_pick),
                     ("peak_top_pick", peak_top_pick),
                     ("buy_flow_winner", buy_flow_on_winner)]:
        g = _pooled(session, GPT, mem, fn)
        c = _pooled(session, CLA, mem, fn)
        if not g or not c:
            print("  %-18s (missing)" % name)
            continue
        obs, p = _perm_two_sample(g, c)
        flag = "  <== sig" if p < 0.05 else ""
        print("  %-18s n=%-3d mean=%.3f      n=%-3d mean=%.3f      %+.3f [p=%.3f]%s"
              % (name, len(g), statistics.mean(g), len(c), statistics.mean(c),
                 obs, p, flag))
    # baseline-only, for the exact 0.82-vs-0.46 number cited across turns
    print("\n  baseline cell only (the cited 'top_pick 0.82 vs 0.46'):")
    bcell = "axis-baseline__val-default__n30"
    for name, fn in [("top_pick(end)", top_pick), ("peak_top_pick", peak_top_pick),
                     ("buy_flow_winner", buy_flow_on_winner)]:
        ge = load(session, GPT, mem, bcell)
        ce = load(session, CLA, mem, bcell)
        if not ge or not ce:
            continue
        g = [v for v in (fn(e) for e in ge) if v is not None]
        c = [v for v in (fn(e) for e in ce) if v is not None]
        print("    %-18s gpt n=%-2d %.2f   claude n=%-2d %.2f"
              % (name, len(g), statistics.mean(g), len(c), statistics.mean(c)))
    print()


def check_temperature_floor(session, mem):
    print("=" * 78)
    print("(6) TEMPERATURE FLOOR -- does activity actually rise with temp, or is the")
    print("    mean-level read within noise? perm floor on temp 0.0 -> 1.2.")
    print("=" * 78)
    tcells = [("0.0", "axis-temperature__val-0__n30"),
              ("0.7", "axis-temperature__val-0.7__n30"),
              ("1.2", "axis-temperature__val-1.2__n30")]

    def tc(ep):
        return float(ep["footprint"]["trade_count"])

    def vol(ep):
        return float(ep["footprint"]["total_volume"])

    for model in (GPT, CLA):
        print("  -- %s --" % model)
        for mlabel, fn in [("trade_count", tc), ("total_volume", vol)]:
            series = {}
            for t, cell in tcells:
                eps = load(session, model, mem, cell)
                if eps:
                    series[t] = [fn(e) for e in eps]
            if "0.0" in series and "1.2" in series:
                # two-sided perm p so it's correct whether activity rises or falls
                obs, p = _perm_two_sample(series["0.0"], series["1.2"])  # gap = 1.2-0.0
                means = "  ".join("%s=%.2f" % (t, statistics.mean(series[t]))
                                  for t in ("0.0", "0.7", "1.2") if t in series)
                flag = "  <== clears (p<0.05)" if p < 0.05 else ""
                print("     %-13s %s   0->1.2 gap=%+.2f p=%.3f%s"
                      % (mlabel, means, obs, p, flag))
        print()


def check_within_label(session, mem, model=CLA):
    print("=" * 78)
    print("(4) WITHIN-LABEL volatility dispersion -- %s" % model)
    print("    Adjudicates the volatility null: does the model respond to REALIZED")
    print("    range within a fixed label? (flat label sweep != null.)")
    print("=" * 78)
    print("  %-6s %-9s %-26s %-26s" % ("vol", "n(lo/hi)", "size-disp gap [perm p]",
                                       "trade_count gap [perm p]"))
    for vol, v in VOL_LABELS:
        eps = load(session, model, mem, "axis-volatility__val-%s__n30" % v)
        if not eps:
            print("  %-6s (missing)" % vol)
            continue
        d, pd, t, pt, nlo, nhi = within_label(eps)
        flag = "  <== clears (p<0.05)" if pd < 0.05 else ""
        print("  %-6s %-9s %+7.1f [p=%.3f]%-10s %+7.2f [p=%.3f]"
              % (vol, "%d/%d" % (nlo, nhi), d, pd, flag, t, pt))
    print()


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "2026-06-27_official-v1"
    mem = sys.argv[2] if len(sys.argv) > 2 else "off"
    here = os.path.dirname(os.path.abspath(__file__))
    session = os.path.join(here, session)
    print("\nKNOB DIAGNOSTICS  session=%s  mem-%s  N_PERM=%d\n"
          % (os.path.basename(session), mem, N_PERM))
    check_scorability(session, mem)
    check_knowability(session, mem)
    check_permfloor(session, mem)
    check_discovery_robustness(session, mem)
    check_temperature_floor(session, mem)
    check_within_label(session, mem, CLA)


if __name__ == "__main__":
    main()
