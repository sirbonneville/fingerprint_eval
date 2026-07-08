"""Review check: is 1/3 the right chance floor for 'discovery'?

Computes a NAIVE PRICE-FOLLOWER counterfactual: at every turn where the model
actually executed a buy, what buy_flow would it have earned if it had put the
same notional on the band the price was in at that moment? Also an unconditional
per-turn version (equal weight on all 5 decision points).

Read-only. Paths are regenerated from (cell config, seed) and validated against
the stored settlement price of every episode.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from fingerprint_eval.price_path import generate_path  # noqa: E402

V1 = os.path.join(HERE, "battery_WITHOUT_SYNTH_TRADING/2026-06-27_official-v1")
V2 = os.path.join(HERE, "battery_WITH_SYNTH_TRADING/2026-06-29_official-v2")
GPT, CLA = "openai-gpt-4o", "anthropic-claude-sonnet-4"

CORE = (["axis-baseline__val-default__n30"]
        + ["axis-volatility__val-%s__n30" % v for v in ("0.5", "1", "1.5", "2")]
        + ["axis-temperature__val-%s__n30" % v for v in ("0", "0.7", "1.2")]
        + ["axis-knowability_min__val-%s__n30" % v for v in ("0", "60", "120")]
        + ["axis-band_width__val-%s__n30" % v for v in ("5", "20")])


def band_of(price, edges):
    if price < edges[0]:
        return 0
    if price < edges[1]:
        return 1
    return 2


def cell_paths(session, model, mem, cell):
    folder = os.path.join(session, "%s__mem-%s__%s" % (model, mem, cell))
    cfg = json.load(open(os.path.join(folder, "config.json")))["cell_config"]
    w = float(cfg["band_width"])
    edges = [cfg["anchor"] - w / 2.0, cfg["anchor"] + w / 2.0]
    paths = {}
    for seed in range(int(cfg.get("n", 30) or 30)):
        p = generate_path(
            anchor=cfg["anchor"], hours=cfg["hours"], interval_min=cfg["interval_min"],
            volatility=cfg["volatility"], drift=cfg["drift"], band_width=w,
            post_close_min=cfg["knowability_min"], seed=seed, model=cfg["price_model"],
        )
        paths[seed] = {t: price for t, price in p.points}
    return folder, edges, paths


def main():
    mismatch = 0
    checked = 0
    for arm_name, arm in (("v1", V1), ("v2", V2)):
        print("=" * 78)
        print("%s  naive price-follower baseline vs actual buy_flow" % arm_name)
        print("=" * 78)
        for model in (GPT, CLA):
            for mem in ("off", "on"):
                act_num = act_den = 0.0
                naive_num = naive_den = 0.0
                turn_hits = turn_tot = 0
                for cell in CORE:
                    folder, edges, paths = cell_paths(arm, model, mem, cell)
                    rp = os.path.join(folder, "runs.jsonl")
                    if not os.path.isfile(rp):
                        continue
                    for line in open(rp):
                        ep = json.loads(line)["episode"]
                        seed, w = ep["seed"], ep["winning_outcome"]
                        pts = paths[seed]
                        # validation: regenerated settlement price == stored
                        settle = pts[max(pts)]
                        checked += 1
                        if abs(settle - ep["settlement_price"]) > 1e-9:
                            mismatch += 1
                        for t in sorted(pts):
                            if t <= ep["trading_close_min"]:
                                turn_tot += 1
                                if band_of(pts[t], edges) == w:
                                    turn_hits += 1
                        for turn in ep["turns"]:
                            if turn["parsed"]["action"] == "buy" and turn["executed"]:
                                o = turn["planned_order"].get(
                                    "outcome", turn["parsed"]["outcome"])
                                notion = turn["notional"]
                                act_den += notion
                                if o == w:
                                    act_num += notion
                                naive_den += notion
                                if band_of(pts[turn["t"]], edges) == w:
                                    naive_num += notion
                print("  %-26s mem-%-3s  actual buy_flow=%.3f   "
                      "naive(current-band, same buys)=%.3f   "
                      "naive(all 5 turns, unweighted)=%.3f"
                      % (model, mem,
                         act_num / act_den if act_den else float("nan"),
                         naive_num / naive_den if naive_den else float("nan"),
                         turn_hits / turn_tot if turn_tot else float("nan")))
        print()
    print("path reconstruction check: %d episodes, %d settlement mismatches"
          % (checked, mismatch))


if __name__ == "__main__":
    main()
