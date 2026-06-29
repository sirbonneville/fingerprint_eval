"""Does either model READ THE LIVE BOARD as signal? (v2-only analysis)

Read-only. v2's board moves on its own between the model's turns (synthetic
flow), which v1's frozen board never did. Question: when the model buys, does its
target band track where the board has just been PUSHED?

Three alignments per buy (turn i>=1), pooled per model x memory over all v2 core
cells, each vs the 1/3 chance baseline and a permutation floor (shuffle the
drift/favorite labels across buys N times):

  favorite     -- target == argmax(probabilities_before): buys the current LEADER.
  synth_drift  -- target == argmax(prob_before - left_board[i-1]): buys the band the
                  SYNTHETIC FLOW pushed up since the model last acted (pure board
                  move; left_board is the board AFTER the model's own prior trade).

THE CLEAN TEST (disentangles level from movement): restrict to buys where the
drift band != the favorite band (the board moved AWAY from the leader). Among
those, does the model follow the DRIFT or the LEADER? Following drift here is the
only alignment that isn't explained by "just buy the leader".

CONFOUND, stated up front: synthetic flow itself targets "which band the price is
in now", so synth_drift correlates with the underlying price the model can ALSO
read directly from the price line. Alignment with drift is therefore an UPPER
bound on board-reading; it cannot by itself separate "reads the board move" from
"reads the same price the board read". The split test narrows this but does not
fully kill it. Flagged, not hidden.

Usage:  python3 eval_runs/board_reaction.py [session_dir]
"""

import glob
import json
import os
import random
import sys

random.seed(0)
N_PERM = 20000
GPT = "openai-gpt-4o"
CLA = "anthropic-claude-sonnet-4"


def load_cells(session, model, mem):
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


def left_board(turns, i):
    t = turns[i]
    tr = t.get("trade_result")
    if t.get("executed") and tr and tr.get("probabilities_after"):
        return tr["probabilities_after"]
    return t["probabilities_before"]


def collect(cells):
    """Per buy (turn i>=1 with a measurable drift): (target, favorite, drift)."""
    rows = []
    for eps in cells.values():
        for e in eps:
            turns = e["episode"]["turns"]
            for i, t in enumerate(turns):
                if i < 1 or t["parsed"].get("action") != "buy":
                    continue
                tgt = t["parsed"].get("outcome")
                pb = t["probabilities_before"]
                if tgt is None or not (0 <= tgt < len(pb)):
                    continue
                lb = left_board(turns, i - 1)
                drift_vec = [pb[j] - lb[j] for j in range(len(pb))]
                if max(abs(x) for x in drift_vec) < 1e-9:
                    continue  # board did not move since the model last acted
                rows.append((tgt, argmax(pb), argmax(drift_vec)))
    return rows


def rate(rows, key):
    if not rows:
        return float("nan"), 0
    hit = sum(1 for r in rows if r[0] == r[key])
    return hit / len(rows), len(rows)


def perm_floor(rows, key):
    """P(shuffled match-rate >= observed). Shuffle the favorite/drift labels across
    buys, breaking any target<->label association while preserving marginals."""
    if not rows:
        return float("nan")
    labels = [r[key] for r in rows]
    targets = [r[0] for r in rows]
    obs = sum(1 for t, l in zip(targets, labels) if t == l)
    idx = list(range(len(labels)))
    hit = 0
    for _ in range(N_PERM):
        random.shuffle(idx)
        s = sum(1 for t, j in zip(targets, idx) if t == labels[j])
        if s >= obs:
            hit += 1
    return hit / N_PERM


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "battery_WITH_SYNTH_TRADING/2026-06-29_official-v2"
    here = os.path.dirname(os.path.abspath(__file__))
    session = os.path.join(here, session)
    print("\nBOARD REACTION  session=%s  N_PERM=%d" % (os.path.basename(session), N_PERM))
    print("chance baseline (3 bands) = 33.3%%; perm p = P(shuffled match >= observed)\n")

    for model in (GPT, CLA):
        for mem in ("off", "on"):
            rows = collect(load_cells(session, model, mem))
            fav, n = rate(rows, 1)
            dft, _ = rate(rows, 2)
            pf = perm_floor(rows, 1)
            pd = perm_floor(rows, 2)
            print("== %s mem-%s ==  buys-with-drift n=%d" % (model, mem, n))
            print("   target == favorite (level)  : %.1f%%  [perm p=%.3f]"
                  % (100 * fav, pf))
            print("   target == synth-drift (move) : %.1f%%  [perm p=%.3f]"
                  % (100 * dft, pd))
            # clean split: board moved AWAY from the leader
            split = [r for r in rows if r[1] != r[2]]
            if split:
                fl = sum(1 for r in split if r[0] == r[1]) / len(split)
                dl = sum(1 for r in split if r[0] == r[2]) / len(split)
                nei = sum(1 for r in split if r[0] != r[1] and r[0] != r[2]) / len(split)
                print("   SPLIT (drift != favorite, n=%d): follows leader %.1f%%  /  "
                      "follows drift %.1f%%  /  neither %.1f%%"
                      % (len(split), 100 * fl, 100 * dl, 100 * nei))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
