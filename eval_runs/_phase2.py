"""Phase 2: matched-seed (cell,seed) v1<->v2 contrasts with paired sign-flip floors."""
import json, os, random, statistics
random.seed(1)
NP = 20000
HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(HERE, "battery_WITHOUT_SYNTH_TRADING/2026-06-27_official-v1")
V2 = os.path.join(HERE, "battery_WITH_SYNTH_TRADING/2026-06-29_official-v2")
GPT, CLA = "openai-gpt-4o", "anthropic-claude-sonnet-4"
BASE = "axis-baseline__val-default__n30"
AX = {"volatility": ["0.5", "1", "1.5", "2"], "temperature": ["0", "0.7", "1.2"],
      "knowability_min": ["0", "60", "120"], "band_width": ["5", "20"]}


def cells():
    c = [BASE]
    for a, vs in AX.items():
        for v in vs:
            c.append("axis-%s__val-%s__n30" % (a, v))
    return c


def load_map(session, model, mem, cell):
    p = os.path.join(session, "%s__mem-%s__%s/runs.jsonl" % (model, mem, cell))
    if not os.path.isfile(p):
        return {}
    out = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            out[(cell, r["seed"])] = r
    return out


def gross(pos):
    return sum(abs(x) for x in pos)


def peak(r):
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


def buy_flow(r):
    e = r["episode"]; w = e["winning_outcome"]; flow = tot = 0.0
    for t in e["turns"]:
        if t["parsed"].get("action") == "buy" and t.get("executed"):
            o = (t.get("planned_order") or {}).get("outcome", t["parsed"].get("outcome"))
            amt = t.get("notional") or 0.0; tot += amt
            if o == w:
                flow += amt
    return None if tot <= 1e-9 else flow / tot


def win(r):
    return 1.0 if r["episode"]["footprint"]["win"] else 0.0


def tc(r):
    return float(r["episode"]["footprint"]["trade_count"])


def hold(r):
    return r["episode"]["footprint"]["n_holds"] / r["episode"]["footprint"]["n_turns"]


def paired(model, mem, fn):
    """matched (cell,seed) pairs where metric defined in BOTH batteries."""
    d1 = {}; d2 = {}
    for c in cells():
        d1.update(load_map(V1, model, mem, c)); d2.update(load_map(V2, model, mem, c))
    a = []; b = []
    for k in d1:
        if k in d2:
            x = fn(d1[k]); y = fn(d2[k])
            if x is not None and y is not None:
                a.append(x); b.append(y)
    return a, b


def signflip_p(a, b, n=NP):
    """paired: obs = mean(b-a); null flips each pair's sign."""
    d = [b[i] - a[i] for i in range(len(a))]
    obs = statistics.mean(d)
    hit = 0
    for _ in range(n):
        s = sum(x if random.random() < 0.5 else -x for x in d)
        if abs(s / len(d)) >= abs(obs) - 1e-12:
            hit += 1
    return obs, hit / n


print("=" * 78)
print("PHASE 2 matched-seed v1->v2 (paired sign-flip floor, 20k)  chance peak/flow=0.333")
print("=" * 78)
for nm, fn in [("peak", peak), ("buy_flow", buy_flow), ("win", win), ("trade_count", tc), ("hold_rate", hold)]:
    print("\n-- %s --" % nm)
    for model in (GPT, CLA):
        for mem in ("off", "on"):
            a, b = paired(model, mem, fn)
            obs, p = signflip_p(a, b)
            tag = model.split("-")[0]
            print("  %-9s mem-%-3s n_pairs=%4d  v1=%.3f  v2=%.3f  delta=%+.3f  [p=%.3f]%s"
                  % (tag, mem, len(a), statistics.mean(a), statistics.mean(b), obs, p,
                     "  *" if p < 0.05 else ""))
print("\nPHASE2 COMPLETE")
