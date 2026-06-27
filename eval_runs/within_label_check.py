"""Within-label path-response check (read-only analysis of a battery session).

Isolates "model responds to the realized path it drew" from "vol cells differ for
vol reasons" by binning episodes of a SINGLE volatility label by the realized
range they happened to draw (calm vs wild), nothing else varying. Reports the
size-dispersion and trade-count gaps against a PERMUTATION noise floor (shuffle
the calm/wild assignment within the label many times), so "clearest channel"
must clear its own bar before it is load-bearing.

Usage:
    python3 within_label_check.py <session_dir> <model_slug_sanitized> [mem]
    e.g. python3 within_label_check.py 2026-06-27_official-v1 openai-gpt-4o off
         python3 within_label_check.py 2026-06-27_official-v1 anthropic-claude-sonnet-4 off
"""
import json
import os
import re
import statistics
import sys
import random

random.seed(0)

VOL_LABELS = {"0.5": "0.5", "1.0": "1", "1.5": "1.5", "2.0": "2"}
N_PERM = 5000


def episodes(session, model, mem, cell):
    path = os.path.join(session, "%s__mem-%s__%s/runs.jsonl" % (model, mem, cell))
    if not os.path.isfile(path):
        return None
    eps = []
    for line in open(path):
        eps.append(json.loads(line)["episode"])
    return eps


def realized_range(ep):
    p = ep["turns"][-1]["prompt"]
    pr = [float(x) for x in re.findall(r"min: ([0-9.]+)", p)]
    return max(pr) - min(pr) if pr else 0.0


def exec_sizes(ep):
    return [t["parsed"]["size"] for t in ep["turns"]
            if t["parsed"]["action"] in ("buy", "sell") and t["executed"]]


def std(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def within_label(eps):
    """Return (observed dispersion gap, perm p) and trade-count gap + perm p."""
    ranges = [realized_range(e) for e in eps]
    sizes = [exec_sizes(e) for e in eps]
    tc = [len(s) for s in sizes]
    med = statistics.median(ranges)
    lo_idx = [i for i, r in enumerate(ranges) if r <= med]
    hi_idx = [i for i, r in enumerate(ranges) if r > med]

    def disp_gap(hi, lo):
        sh = [s for i in hi for s in sizes[i]]
        sl = [s for i in lo for s in sizes[i]]
        return std(sh) - std(sl)

    def tc_gap(hi, lo):
        return (statistics.mean(tc[i] for i in hi) if hi else 0) - \
               (statistics.mean(tc[i] for i in lo) if lo else 0)

    obs_disp = disp_gap(hi_idx, lo_idx)
    obs_tc = tc_gap(hi_idx, lo_idx)

    n = len(eps)
    k = len(hi_idx)
    idx = list(range(n))
    pd = pt = 0
    for _ in range(N_PERM):
        random.shuffle(idx)
        h, l = idx[:k], idx[k:]
        if disp_gap(h, l) >= obs_disp:
            pd += 1
        if tc_gap(h, l) >= obs_tc:
            pt += 1
    return obs_disp, pd / N_PERM, obs_tc, pt / N_PERM, len(lo_idx), len(hi_idx)


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "2026-06-27_official-v1"
    model = sys.argv[2] if len(sys.argv) > 2 else "openai-gpt-4o"
    mem = sys.argv[3] if len(sys.argv) > 3 else "off"
    here = os.path.dirname(os.path.abspath(__file__))
    session = os.path.join(here, session)

    print("MODEL=%s  mem-%s  session=%s\n" % (model, mem, os.path.basename(session)))

    # Identical-config anchor: baseline == vol=1.0 (same cell, independent sampling).
    base = episodes(session, model, mem, "axis-baseline__val-default__n30")
    v1 = episodes(session, model, mem, "axis-volatility__val-1__n30")
    if base and v1:
        sb = [s for e in base for s in exec_sizes(e)]
        s1 = [s for e in v1 for s in exec_sizes(e)]
        print("IDENTICAL-CONFIG ANCHOR (baseline vs vol=1.0, same config):")
        print("  size std: baseline=%.1f  vol1.0=%.1f  -> |gap|=%.1f is the noise floor for 'size std'\n"
              % (std(sb), std(s1), abs(std(sb) - std(s1))))

    print("WITHIN-LABEL calm-vs-wild (bin one label's episodes by realized range):")
    print("  %-6s %-8s %-28s %-28s" % ("vol", "n(lo/hi)", "size-disp gap (hi-lo) [perm p]",
                                       "trade_count gap (hi-lo) [perm p]"))
    for vol, cell_v in VOL_LABELS.items():
        eps = episodes(session, model, mem, "axis-volatility__val-%s__n30" % cell_v)
        if not eps:
            print("  %-6s (not complete yet)" % vol)
            continue
        d, pd, t, pt, nlo, nhi = within_label(eps)
        flag_d = "  <== clears noise (p<0.05)" if pd < 0.05 else ""
        print("  %-6s %-8s %+7.1f [p=%.3f]%-12s %+7.2f [p=%.3f]"
              % (vol, "%d/%d" % (nlo, nhi), d, pd, flag_d, t, pt))


if __name__ == "__main__":
    main()
