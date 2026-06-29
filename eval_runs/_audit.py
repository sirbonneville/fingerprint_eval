"""INDEPENDENT audit reducer (Phase 1) + cross-battery inputs (Phase 2).

Deliberately does NOT import master_analysis / knob_diagnostics / board_reaction /
allin_check. Every metric is recomputed here from raw runs.jsonl so a shared bug in
those scripts cannot launder into agreement. Read-only.
"""
import glob, json, os, re, statistics, random, sys

random.seed(0)
NP = 20000
HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, "battery_WITHOUT_SYNTH_TRADING/2026-06-27_official-v1")
V2 = os.path.join(HERE, "battery_WITH_SYNTH_TRADING/2026-06-29_official-v2")
GPT, CLA = "openai-gpt-4o", "anthropic-claude-sonnet-4"
MEMS = ["off", "on"]
BASE = "axis-baseline__val-default__n30"
AXES = {
    "volatility": ["0.5", "1", "1.5", "2"],
    "temperature": ["0", "0.7", "1.2"],
    "knowability_min": ["0", "60", "120"],
    "band_width": ["5", "20"],
}
SAME = [BASE, "axis-volatility__val-1__n30", "axis-knowability_min__val-0__n30",
        "axis-temperature__val-0.7__n30"]
KNOW = [("baseline", BASE), ("k=0", "axis-knowability_min__val-0__n30"),
        ("k=60", "axis-knowability_min__val-60__n30"),
        ("k=120", "axis-knowability_min__val-120__n30")]


def core_cells():
    cells = [BASE]
    for ax, vs in AXES.items():
        for v in vs:
            cells.append("axis-%s__val-%s__n30" % (ax, v))
    return cells


def load(session, model, mem, cell):
    p = os.path.join(session, "%s__mem-%s__%s/runs.jsonl" % (model, mem, cell))
    if not os.path.isfile(p):
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def fp(rec):
    return rec["episode"]["footprint"]


def gross(pos):
    return sum(abs(x) for x in pos)


# ---------- independent metric defs ----------
def m_trade_count(r): return float(fp(r)["trade_count"])
def m_hold_rate(r): return fp(r)["n_holds"] / fp(r)["n_turns"]
def m_volume(r): return float(fp(r)["total_volume"])
def m_med_size(r):
    v = fp(r)["median_trade_size"]
    return None if v is None else float(v)
def m_dir_sw(r): return float(fp(r)["direction_switches"])
def m_outcomes(r): return float(fp(r)["outcomes_traded"])
def m_first_frac(r): return float(fp(r)["first_trade_frac"])
def m_win(r): return 1.0 if fp(r)["win"] else 0.0
def m_exit_flat(r): return 0.0 if gross(r["episode"]["closing_position"]) > 1e-9 else 1.0
def m_held(r): return 1.0 - m_exit_flat(r)


def m_top_pick(r):
    e = r["episode"]; pos = e["closing_position"]; w = e["winning_outcome"]
    if gross(pos) <= 1e-9 or not (0 <= w < len(pos)):
        return None
    return 1.0 if sum(1 for x in pos if x > pos[w] + 1e-12) == 0 else 0.0


def m_peak(r):
    e = r["episode"]; w = e["winning_outcome"]
    states = [t["position_before"] for t in e["turns"]] + [e["closing_position"]]
    bg, bp = -1.0, None
    for pos in states:
        g = gross(pos)
        if g > bg + 1e-12:
            bg, bp = g, pos
    if bp is None or bg <= 1e-9 or not (0 <= w < len(bp)):
        return None
    return 1.0 if sum(1 for x in bp if x > bp[w] + 1e-12) == 0 else 0.0


def m_buy_flow(r):
    e = r["episode"]; w = e["winning_outcome"]; flow = 0.0; tot = 0.0
    for t in e["turns"]:
        if t["parsed"].get("action") == "buy" and t.get("executed"):
            o = (t.get("planned_order") or {}).get("outcome", t["parsed"].get("outcome"))
            amt = t.get("notional") or 0.0
            tot += amt
            if o == w:
                flow += amt
    return None if tot <= 1e-9 else flow / tot


BEH = [("trade_count", m_trade_count), ("hold_rate", m_hold_rate),
       ("total_volume", m_volume), ("median_trade_size", m_med_size),
       ("direction_switches", m_dir_sw), ("outcomes_traded", m_outcomes),
       ("exit_flat_rate", m_exit_flat)]


def vals(recs, fn):
    return [v for v in (fn(r) for r in recs) if v is not None]


def pooled(session, model, mem, fn, cells=None):
    out = []
    for c in (cells or core_cells()):
        recs = load(session, model, mem, c)
        if recs:
            out += vals(recs, fn)
    return out


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def cv(xs):
    m = mean(xs)
    return statistics.pstdev(xs) / m if xs and m else float("nan")


def perm_two(a, b, n=NP):
    if not a or not b:
        return float("nan"), float("nan")
    obs = mean(b) - mean(a); pool = a + b; nb = len(b)
    idx = list(range(len(pool))); hit = 0
    for _ in range(n):
        random.shuffle(idx)
        g = mean([pool[i] for i in idx[:nb]]) - mean([pool[i] for i in idx[nb:]])
        if abs(g) >= abs(obs) - 1e-12:
            hit += 1
    return obs, hit / n


def perm_drop(a, b, n=NP):
    """one-sided P(perm gap <= obs); for a discovery DROP a->b."""
    if not a or not b:
        return float("nan"), float("nan")
    obs = mean(b) - mean(a); pool = a + b; nb = len(b)
    idx = list(range(len(pool))); hit = 0
    for _ in range(n):
        random.shuffle(idx)
        g = mean([pool[i] for i in idx[:nb]]) - mean([pool[i] for i in idx[nb:]])
        if g <= obs:
            hit += 1
    return obs, hit / n


# ---------- prompt parsing (independent) ----------
def price_series(prompt):
    return [float(x) for x in re.findall(r"min: ([0-9.]+)", prompt)]


def band_edges(prompt):
    return sorted(set(float(x) for x in re.findall(r"< ([0-9.]+)\)", prompt)))


def band_of(price, edges):
    b = 0
    for e in edges:
        if price >= e:
            b += 1
    return b


def banner(s):
    print("\n" + "=" * 78); print(s); print("=" * 78)


def fast(session, name):
    banner("FAST  [%s]" % name)
    # personality
    print("-- A. personality pooled means (13 core cells) --")
    print("  %-20s %8s %8s %8s %8s" % ("metric", "gpt off", "gpt on", "cla off", "cla on"))
    for label, fn in BEH + [("win", m_win)]:
        row = "  %-20s" % label
        for model in (GPT, CLA):
            for mem in MEMS:
                row += " %8.2f" % mean(pooled(session, model, mem, fn))
        print(row)
    # noise floor
    print("\n-- B. noise floor: spread across 4 identical-config draws --")
    for model in (GPT, CLA):
        for mem in MEMS:
            tc = [mean(vals(load(session, model, mem, c), m_trade_count)) for c in SAME]
            sz = [mean(vals(load(session, model, mem, c), m_med_size)) for c in SAME]
            vo = [mean(vals(load(session, model, mem, c), m_volume)) for c in SAME]
            print("  %-26s mem-%s  tc_spread=%.2f  size_spread=%.1f  vol_spread=%.0f"
                  % (model, mem, max(tc) - min(tc), max(sz) - min(sz), max(vo) - min(vo)))
    # per-cell constancy
    print("\n-- A2. per-cell constancy (median_trade_size / trade_count by cell, mem-off) --")
    for model in (GPT, CLA):
        szs = []; tcs = []
        for c in core_cells():
            recs = load(session, model, "off", c)
            szs.append(mean(vals(recs, m_med_size))); tcs.append(mean(vals(recs, m_trade_count)))
        print("  %-26s med_size range=[%.1f, %.1f] band=%.1f   trade_count range=[%.2f, %.2f]"
              % (model, min(szs), max(szs), max(szs) - min(szs), min(tcs), max(tcs)))
    # sizing fingerprint
    print("\n-- F2. sizing fingerprint (executed BUYS) --")
    for model in (GPT, CLA):
        for mem in MEMS:
            sizes = []; vols = []
            for c in core_cells():
                for r in (load(session, model, mem, c) or []):
                    vols.append(m_volume(r))
                    for t in r["episode"]["turns"]:
                        if t["parsed"].get("action") == "buy" and t.get("executed"):
                            sizes.append(t["parsed"].get("size"))
            sizes = [s for s in sizes if s is not None]
            f333 = sum(1 for s in sizes if abs(s - 333.33) <= 0.5) / len(sizes)
            f50 = sum(1 for s in sizes if abs(s - round(s / 50) * 50) < 1e-6) / len(sizes)
            print("  %-26s mem-%s  n_buys=%d  @333.33=%.0f%%  $50-mult=%.0f%%  size_CV=%.2f  vol_CV=%.2f"
                  % (model, mem, len(sizes), 100 * f333, 100 * f50, cv(sizes), cv(vols)))
    # discovery means + n
    print("\n-- D/F8 discovery means (chance=0.33) + n / undefined --")
    for mem in MEMS:
        for nm, fn in [("top_pick", m_top_pick), ("peak", m_peak), ("buy_flow", m_buy_flow)]:
            row = "  mem-%s %-9s" % (mem, nm)
            for model in (GPT, CLA):
                allrecs = []
                for c in core_cells():
                    allrecs += (load(session, model, mem, c) or [])
                v = [fn(r) for r in allrecs]
                defi = [x for x in v if x is not None]
                row += "  %s: %.3f (n=%d, undef=%d)" % (
                    "gpt" if model == GPT else "cla", mean(defi), len(defi), len(v) - len(defi))
            print(row)
    # win decomposition
    print("\n-- F4/F6 win decomposition: held_rate / win / P(win|held) --")
    print("  %-22s %8s %8s %8s %8s" % ("", "gpt off", "gpt on", "cla off", "cla on"))
    for label, fn in [("held_rate", m_held), ("win", m_win)]:
        row = "  %-22s" % label
        for model in (GPT, CLA):
            for mem in MEMS:
                row += " %8.3f" % mean(pooled(session, model, mem, fn))
        print(row)
    row = "  %-22s" % "P(win|held)"
    for model in (GPT, CLA):
        for mem in MEMS:
            cond = []
            for c in core_cells():
                for r in (load(session, model, mem, c) or []):
                    if m_held(r) == 1.0:
                        cond.append(m_win(r))
            row += " %8.3f" % mean(cond)
    print(row)
    # knowability oracle (model-independent; compute from gpt mem-off)
    print("\n-- F7/F8 knowability oracle (model-independent; gpt mem-off paths) --")
    for label, cell in KNOW:
        recs = load(session, GPT, "off", cell)
        edges = band_edges(recs[0]["episode"]["turns"][0]["prompt"])
        hits = []; moves = []
        for r in recs:
            e = r["episode"]; ps = price_series(e["turns"][-1]["prompt"])
            close = ps[-1] if ps else None
            if close is None:
                continue
            hits.append(1.0 if band_of(close, edges) == e["winning_outcome"] else 0.0)
            moves.append(abs(e["settlement_price"] - close))
        print("  %-9s close-band-hit=%.0f%% (%d/%d)  mean|settle-close|=%.2f"
              % (label, 100 * mean(hits), int(sum(hits)), len(hits), mean(moves)))
    sys.stdout.flush()


def allin_counts(session, model, mem):
    n_allin = n_buys = n_turns = 0; eps_hit = 0; total = 0
    for c in core_cells():
        for r in (load(session, model, mem, c) or []):
            total += 1; had = False
            for t in r["episode"]["turns"]:
                n_turns += 1
                if t["parsed"].get("action") == "buy":
                    n_buys += 1
                    size = t["parsed"].get("size"); cash = t.get("cash_before") or 0.0
                    if size is not None and cash > 0 and size >= 0.9 * cash:
                        n_allin += 1; had = True
            if had:
                eps_hit += 1
    return n_allin, n_buys, n_turns, eps_hit, total


def board_reaction(session, model, mem):
    """Return (n, fav%, drift%, split_n, follow_drift%, follow_leader%,
              clean_n, clean_follow_drift%, drift==priceband% in split)."""
    rows = []  # (target, favorite, drift, prev_was_hold, priceband)
    for c in core_cells():
        for r in (load(session, model, mem, c) or []):
            turns = r["episode"]["turns"]
            edges = band_edges(turns[0]["prompt"])
            for i, t in enumerate(turns):
                if i < 1 or t["parsed"].get("action") != "buy":
                    continue
                tgt = t["parsed"].get("outcome"); pb = t["probabilities_before"]
                if tgt is None or not (0 <= tgt < len(pb)):
                    continue
                prev = turns[i - 1]["probabilities_before"]
                drift_vec = [pb[j] - prev[j] for j in range(len(pb))]
                if max(abs(x) for x in drift_vec) < 1e-9:
                    continue
                fav = max(range(len(pb)), key=lambda j: pb[j])
                drift = max(range(len(drift_vec)), key=lambda j: drift_vec[j])
                prev_hold = turns[i - 1]["parsed"].get("action") == "hold" or not turns[i - 1].get("executed")
                ps = price_series(t["prompt"])
                pband = band_of(ps[-1], edges) if ps else -1
                rows.append((tgt, fav, drift, prev_hold, pband))
    n = len(rows)
    favp = 100 * sum(1 for r in rows if r[0] == r[1]) / n if n else float("nan")
    driftp = 100 * sum(1 for r in rows if r[0] == r[2]) / n if n else float("nan")
    split = [r for r in rows if r[1] != r[2]]
    sd = 100 * sum(1 for r in split if r[0] == r[2]) / len(split) if split else float("nan")
    sl = 100 * sum(1 for r in split if r[0] == r[1]) / len(split) if split else float("nan")
    clean = [r for r in split if r[3]]  # drift!=fav AND prev turn was a hold => pure synth
    cd = 100 * sum(1 for r in clean if r[0] == r[2]) / len(clean) if clean else float("nan")
    # price confound: in split, how often does drift band == current price band
    dpb = 100 * sum(1 for r in split if r[2] == r[4]) / len(split) if split else float("nan")
    return n, favp, driftp, len(split), sd, sl, len(clean), cd, dpb


def slow(session, name, is_v2):
    banner("PERM  [%s]" % name)
    print("-- C. memory gaps (on-off) pooled + perm p (20k) --")
    for model in (GPT, CLA):
        print("  -- %s --" % model)
        for label, fn in BEH + [("top_pick", m_top_pick), ("peak", m_peak), ("buy_flow", m_buy_flow), ("win", m_win)]:
            off = pooled(session, model, "off", fn); on = pooled(session, model, "on", fn)
            obs, p = perm_two(off, on)
            print("     %-18s gap=%+8.2f [p=%.3f]%s" % (label, obs, p, "  *" if p < 0.05 else ""))
        sys.stdout.flush()
    print("\n-- E. discovery cross-model gap (cla-gpt) + perm p --")
    for mem in MEMS:
        for nm, fn in [("top_pick", m_top_pick), ("peak", m_peak), ("buy_flow", m_buy_flow)]:
            g = pooled(session, GPT, mem, fn); c = pooled(session, CLA, mem, fn)
            obs, p = perm_two(g, c)
            print("  mem-%s %-9s gpt=%.3f cla=%.3f gap=%+.3f [p=%.3f]%s"
                  % (mem, nm, mean(g), mean(c), obs, p, "  *" if p < 0.05 else ""))
    sys.stdout.flush()
    print("\n-- D. axis endpoint perm p (first vs last); key metrics --")
    for ax, vs in AXES.items():
        for model in (GPT, CLA):
            for mem in MEMS:
                cells = ["axis-%s__val-%s__n30" % (ax, v) for v in vs]
                series = [load(session, model, mem, c) for c in cells]
                if any(s is None for s in series):
                    continue
                tags = []
                for nm, fn in [("tc", m_trade_count), ("hold", m_hold_rate), ("vol", m_volume),
                               ("size", m_med_size), ("dsw", m_dir_sw), ("out", m_outcomes)]:
                    a = vals(series[0], fn); b = vals(series[-1], fn)
                    _, p = perm_two(a, b)
                    if p < 0.05:
                        tags.append("%s p=%.3f" % (nm, p))
                print("  %-13s %-26s mem-%s: %s" % (ax, model, mem, ", ".join(tags) if tags else "(none clear)"))
        sys.stdout.flush()
    print("\n-- F8 knowability top_pick floor (scorable; k0->k120 one-sided drop) --")
    for model in (GPT, CLA):
        for mem in MEMS:
            k0 = vals(load(session, model, mem, "axis-knowability_min__val-0__n30"), m_top_pick)
            k120 = vals(load(session, model, mem, "axis-knowability_min__val-120__n30"), m_top_pick)
            obs, p = perm_drop(k0, k120)
            print("  %-26s mem-%s  k0->k120 gap=%+.2f p=%.3f%s"
                  % (model, mem, obs, p, "  CLEARS" if p < 0.05 else ""))
    sys.stdout.flush()
    if is_v2:
        print("\n-- F3 board reaction (v2): drift as computed + CLEAN (prev=hold) + price confound --")
        for model in (GPT, CLA):
            for mem in MEMS:
                n, favp, driftp, sn, sd, sl, cn, cd, dpb = board_reaction(session, model, mem)
                print("  %-26s mem-%s n=%d fav=%.0f%% drift=%.0f%% | split(n=%d) followDrift=%.0f%% followLeader=%.0f%%"
                      % (model, mem, n, favp, driftp, sn, sd, sl))
                print("        CLEAN(prev=hold,drift!=fav n=%d) followDrift=%.0f%% | in-split drift==priceband=%.0f%%"
                      % (cn, cd, dpb))
        sys.stdout.flush()
    print("\n-- F9 all-in attempts --")
    for model in (GPT, CLA):
        for mem in MEMS:
            a, b, tt, eh, tot = allin_counts(session, model, mem)
            print("  %-26s mem-%s  allin=%d  buys=%d  %.1f%% of buys  eps=%d/%d"
                  % (model, mem, a, b, 100 * a / b if b else float("nan"), eh, tot))
    sys.stdout.flush()


if __name__ == "__main__":
    for sess, nm, v2 in [(V1, "V1 no-synth", False), (V2, "V2 synth", True)]:
        fast(sess, nm)
    for sess, nm, v2 in [(V1, "V1 no-synth", False), (V2, "V2 synth", True)]:
        slow(sess, nm, v2)
    print("\nAUDIT COMPLETE")
