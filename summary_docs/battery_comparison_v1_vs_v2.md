# Battery comparison — v1 (frozen board) vs v2 (live board)

*Independent audit of both findings documents against their own data, followed by a
matched-seed cross-battery comparison. Every number below was recomputed from the raw
`runs.jsonl` with a standalone reducer (`eval_runs/_audit.py`, `eval_runs/_phase2.py`)
that does **not** import the existing analysis scripts, so a shared bug in those
scripts cannot launder into agreement. Read-only: neither battery directory was
modified. Permutation floors use ≥20,000 shuffles; cross-battery deltas are paired on
the shared `(cell, seed)` key — confirmed identical price paths, settlement prices,
and winning bands across the two batteries.*

---

## Plain-language summary

Both findings documents reproduce. Every load-bearing number was re-derived from scratch
and matched, with two small denominator-definition exceptions ($333.33 share and
$50-multiple share depend on buys-only vs all executed trades). The headline of the
comparison is one clean causal effect, one clean dissolution, and a third category that
is easy to mislabel.

> **Note (2026-07-08):** a later independent review found that the discovery results
> below are measured against the wrong null (chance = 1/3 instead of a naive
> price-follower baseline) and that the v2 "discovery collapse" is largely an
> **entry-timing** effect triggered by a pre-turn-0 board tilt. The p-values and
> numbers below all stand; their interpretation is revised. See the
> **Addendum** at the end of this document before quoting the discovery claims.

### Three buckets — how to read v1↔v2

**Model-level** (survives memory flip *and* environment flip): claude discovers the
winning band better than gpt on unbiased metrics (every cell directionally; peak clears
the floor in both batteries; v1 mem-off buy_flow is marginal at p=.056); memory's win
path runs through commitment not accuracy on the win decomposition; top_pick overstates
skill; knowability is a measurement ceiling; two stable personalities by rank (gpt
low-activity/high-hold; claude high-activity/low-hold — the budget/3 *anchor* is not
part of this bucket; it vanishes in v2).

**Environment-dependent capability** (real where observed, contingent on venue — *not*
an artifact, *not* model-level): v1 gpt unbiased discovery was **genuinely above chance**
(peak/buy_flow ≈ 0.47–0.52, ~5 SE over 0.33; the audit ruled out over-credit) and
**collapsed to chance on the live board** (≈ 0.33–0.36; matched-seed p ≤ 0.003, both
memory conditions). This is the scientific heart of Deliverable B: gpt *can* discover,
but only when the board is frozen — competence depends on the venue, not just the model.

**Frozen-board artifact** (never a robust trait): gpt's $333.33 anchor (59–74% → 0%);
v1's orthogonality 2×2; the band_width↔volatility axis hierarchy. These were properties
of the toy, not the models.

### Headline results

**The clean dissolution:** gpt's budget/3 anchor goes to **0%** in v2 — a no-signal
fallback, not a hard prior.

**The clean causal effect:** the live board **degraded gpt's real discovery** (not
v1 over-credit) while claude's peak discovery stayed essentially unchanged (Δ −0.05 /
−0.02, n.s.; buy_flow mem-on is one exception: 0.54→0.47, p=.000). Win **tracks** that
contrast on matched seeds (gpt 0.50→0.36 / 0.52→0.33, p=.000; claude mem-off
0.40→0.64, p=.000 — the load-bearing win shift; mem-on 0.56→0.60 is directional only,
p=.086) — correlational triangulation, not proven causation.

**Orthogonality dissolved:** on a live board gpt gains memory sensitivity (activity
only) and claude gains market responsiveness — but v2 "market" sensitivity (volatility)
is **entangled with synthetic-flow-driven board busyness**, not separable from
"responds to a noisier board"; do not read it as volatility-the-concept.

**Memory tension (named, not buried):** the commitment-not-accuracy throughline holds on
the win decomposition, but v2 shows memory lifting claude's peak_top_pick +0.08
(p≈.021) — one marginal hit against >100 tests; I would not promote it, but it is a
real seam suggesting memory may touch claude's *discovery* on a live board in a way
frozen-board data never showed.

The one big confound throughout v2: synthetic flow pushes the board toward the band the
price is already in (drift-band == price-band 54–85%), and F3's "synth-drift" metric
failed to isolate pure synthetic flow (see Phase 1 audit). Where that confound matters,
it is flagged each time.

### Audit-grade one-liner per model

**gpt-4o:** On a frozen board, a *genuine but non-robust* discoverer alongside
frozen-board-only behaviors (budget/3 anchor, memory-inertness) that were pure
artifacts; on a live board it loses the anchor, gains memory-driven busyness (activity
only, not skill), and discovery collapses to chance.

**claude-sonnet-4:** A stable, superior discoverer (robust on peak; one significant
buy_flow mem-on drop) whose memory buys commitment-not-accuracy for *win*, with the
load-bearing outcome shift on the live board being mem-off win 0.40→0.64 (p=.000) for
reasons not yet pinned down.

---

## 1. Phase 1 audit — did each document reproduce against its own battery?

**Verdict: yes.** Every load-bearing claim in both documents reproduced under
independent recomputation. The only deviations are two sizing percentages whose value
depends on the trade denominator (buys-only vs all executed trades); the qualitative
claims they support hold either way. I also record one *labeling* issue in the v2
board-reaction metric (the rates are correct, but the variable the doc calls
"synth-drift" is not isolating synthetic flow — detail below). I did not edit either
doc; this is a report.

### v1 — `findings_without_synthetic_trading.md`

| Claim (doc) | Doc value | My recompute | Status | Note |
|---|---|---|---|---|
| F1 personality table (7 metrics × gpt/claude × mem) | full table | identical to 2 d.p. | **MATCH** | tc 1.49/1.58/3.01/3.15; hold .70/.68/.40/.37; vol 511/534/771/864; dsw .32/.33/1.09/.91; outcomes 1.14/1.12/1.18/1.33; exit-flat .08/.14/.45/.18 |
| Noise floor (4 identical draws) | gpt tc .50 / vol 114 / size 62; claude .13/36/18 | gpt .50/114/62; claude .13/36/18 | **MATCH** | |
| gpt median stake "$6 band, $326.73–$333.33, every cell" | $6 band | per-cell **median-of-medians = $333.33 in all 13 cells (band $0.00)** | **MATCH** (stronger) | doc's $6 band is conservative; the robust statistic is dead-on $333.33 |
| "63% of gpt trades are exactly $333.33" | 63% | 59% (all executed trades) / 74% (buys only) | **MISMATCH (minor, definitional)** | anchor is robust and if anything stronger; 63% sits between the two natural denominators, but I cannot reproduce 63% exactly with either |
| "claude hits $333.33 only 11%" | 11% | 8% (all trades) / 22%–8% (buys, off/on) | **APPROX** | denominator-sensitive; claim direction holds |
| F8 discovery: gpt/claude top_pick, peak, buy_flow + n | table | peak-off .472(n388)/.546(n390); buy-off .460/.524; peak-on .522(n381)/.595(n390); top_pick-off .528(n360)/.726(n215) | **MATCH** | undefined counts also match (gpt top_pick undef=30; claude undef=175) |
| F8 cross-model gaps | peak +0.075[.044] / +0.073[.050]; buy_flow +0.064[.055] | +0.075[p=.047] / +0.073[p=.047]; +0.064[p=.056] | **MATCH** | p's within perm noise; both marginal, as stated |
| "both models genuinely above chance" | gpt peak 0.47 > 0.33 | gpt peak 0.472, ≈5 SE above chance | **MATCH** | **gating input for Deliverable B** |
| F4 memory: gpt inert / claude broad | gpt only exit_flat moves; claude 8 metrics p<0.02 | gpt: only exit_flat clears (p=.010); claude: tc/hold/vol/size/dsw/outcomes/exit/win all clear | **MATCH** | |
| F5 axis: band_width > temp > volatility≈knowability(null) | band_width replicates; volatility null | band_width clears gpt both mems (tc/hold/size/dsw, +vol/out mem-on); volatility only isolated single-metric hits; temp gpt mem-off (tc/hold/vol); knowability behavioral null | **MATCH** | |
| Withdrawn: knowability→discovery | withdrawn (floor/ceiling) | k0→k120 top_pick drop **fails floor** in 3/4 (gpt-off p=.078, gpt-on .432, claude-off .170); only claude-on clears (p=.031, isolated). Oracle close-band-hit **100%@k0 → 60%@k≥60** | **MATCH** | withdrawal justified |

### v2 — `findings_with_synth_trading.md`

| Claim (doc) | Doc value | My recompute | Status | Note |
|---|---|---|---|---|
| F1 personality table | full table | identical to 2 d.p. | **MATCH** | tc 2.01/2.18/3.82/3.88; hold .59/.56/.24/.22; vol 655/689/980/1009; exit-flat .25/.36/.15/.08 |
| Noise floor | gpt tc .37/vol 102/size 12; claude .13/52/10 | gpt .37/102/12; claude .13/52/10 | **MATCH** | |
| "0% of gpt trades on $333.33" | 0% | 0% (both denominators, both mems) | **MATCH** | |
| "~80% gpt / ~95% claude trades are $50-multiples" | 80% / 95% | gpt 65% (all)/72% (buys); claude 86% (all)/100%–98% (buys) | **MISMATCH (minor)** for gpt; **APPROX** for claude | doc's gpt ~80% is on the high side; correct figure ≈ 65–72% depending on denominator |
| F4 discovery: gpt at chance / claude discovers | gpt peak 0.36/0.36, buy_flow 0.35/0.35; claude peak 0.50/0.58, buy_flow 0.48/0.47 | gpt peak .364/.356, buy_flow .354/.347; claude peak .495/.579, buy_flow .479/.468 | **MATCH** | |
| F4 cross-model gaps | peak +0.131[.001]/+0.223[.000]; buy_flow +0.125[.000]/+0.120[.000] | identical, all p≤.001 | **MATCH** | |
| F3 board-reaction table (n, favorite%, drift%, split) | n 100/106/434/666; fav 54/58/68/59; drift 71/74/84/74; split-follows-drift 74/68/71/65; follows-leader 24/24/15/21 | **identical, every cell** | **MATCH** | rates reproduce exactly — see labeling caveat below |
| F3 confound ("synth-drift correlates with price line; upper bound") | qualitative | quantified: in the split set, drift-band == current price-band **54%–85%** of the time | **MATCH (confirmed & quantified)** | confound is real and large |
| F6 memory: gpt gains activity sensitivity; claude peak +0.08 | gpt tc/hold/size/exit; claude peak +0.08[.023] | gpt tc(.027)/hold(.015)/size(.029)/exit(.001) clear; claude peak +0.08[p=.021] | **MATCH** | |
| F5 axis: volatility ≫ temp≈band_width≈knowability | volatility replicates (gpt both mems); band_width non-replicating | volatility clears gpt both mems on 5 metrics; band_width clears only 1 metric in 1 mem; temp essentially null | **MATCH** | |
| Withdrawn: all-in instability | rare; most common claude mem-on; follows board | gpt 3.8%/3.8% of buys; claude 1.2%/**7.3%** (mem-on highest) | **MATCH** | withdrawal justified |
| Oracle close-band-hit ceiling | 100%@k0 → 60%@k≥60 | identical (and identical to v1, matched seeds) | **MATCH** | |

### One labeling issue worth recording (does not change any reported rate)

`trade_result` records contain no `probabilities_after` field. The v2 board-reaction
and all-in scripts try to read it and silently fall back to the prior turn's *pre-trade*
board. The net effect is that what the doc labels **"synth-drift"** is actually the
**total board move since the model's last decision point** — which includes the model's
*own* previous trade, not just synthetic flow. (Because the board only moves on trades,
the move between turns is `own_prev_trade + synth_flow`; it is pure synthetic flow only
on turns where the model *held* the previous turn.) The published F3 rates are still
correct as "follows the move since its last decision," but they do not isolate
synthetic flow. Restricting to the clean subset (model held previous turn ⇒ pure synth
flow, and drift ≠ favorite) leaves gpt still following drift at 69% (mem-off, n=16) and
83% (mem-on, n=24) — directionally consistent but tiny-n — while claude almost never
idles, so its clean subset is empty/near-empty (n=0 / n=6). So board-tracking survives
weakly for gpt under the stricter definition and is **untestable for claude** with this
data. This compounds the price confound the doc already states.

---

## PHASE 1 GATE

- **Did everything load-bearing reproduce?** Yes. Personality (every cell, both
  batteries), noise floors, the full discovery panel **including the gating gpt
  unbiased numbers**, memory effects and their p-values, the win decomposition, the
  axis sweeps (and the band_width↔volatility inversion), the knowability ceiling, the
  board-reaction rates, and the all-in/oracle withdrawals — all reproduced.
- **Every mismatch:** (1) v1 "63% @ $333.33" → my 59% (all trades) / 74% (buys);
  (2) v2 "gpt ~80% on $50-multiples" → my 65% / 72%; (3) v1 "claude 11% @ $333.33" → 8%
  (approx). All three are trade-denominator definitions, not computational errors, and
  none touch a Phase 2 input. (4) The F3 "synth-drift" labeling caveat above —
  qualitative refinement, rates unchanged.
- **Does any mismatch feed Phase 2?** No discovery mismatch — the gating Deliverable B
  numbers reproduced exactly, so Phase 2 uses my recomputed discovery (which equals the
  docs'). Deliverable C uses the $333.33 counts; I use my own (59–74% v1 → 0% v2). The
  F3 confound is carried into Phase 2 as an explicit caveat, not as a load-bearing
  cause.

Phase 1 is complete and logged. Proceeding to Phase 2.

---

## 2. The four named deliverables

All v1↔v2 deltas below are paired on the shared `(cell, seed)` key, over the 13 core
cells (baseline + 4 volatility + 3 temperature + 3 knowability + 2 band_width), with a
paired sign-flip permutation floor (20k). Discovery chance = 0.333.

### Deliverable A — Was the v1 "orthogonality" environment-dependent? **Yes, mostly.**

v1's headline 2×2 was: *gpt responds to the market and is inert to memory; claude
responds to memory and is inert to the market.* The audit shows two of those four
quadrants break on a live board.

**gpt's memory sensitivity (memory on−off gap, # metrics clearing p<0.05 of 11):**

| | v1 (frozen) | v2 (live) |
|---|---|---|
| metrics that clear | **1 / 11** (only exit_flat, p=.010) | **4 / 11** (trade_count .027, hold_rate .015, median_size .029, exit_flat .001) |
| discovery/win memory gaps | null (top_pick .079, peak .170, buy_flow .177, win .566) | **still null** (top_pick .172, peak .878, buy_flow .833, win .453) |

So gpt is **no longer memory-inert** on a live board — but the new sensitivity is in
*activity* (it trades more, holds less, and sizes differently when it can see its prior
rationales), **not in discovery or winning**. Honest framing: memory changed how much
gpt does, not how well it does it.

**claude's market sensitivity (volatility axis, # metrics clearing):** v1 ≈ 1 isolated,
non-replicating hit; v2 claude **mem-off clears 4 metrics** (volume, size, dir-switches,
outcomes-traded). claude is **no longer market-inert** on a live board either.

**gpt's market axis itself moved:** v1 = band_width (both mems, 4 metrics); v2 =
volatility (both mems, 5 metrics). gpt stays market-sensitive in both — but to a
*different* axis.

**Verdict.** The "each model is *sensitive* to one thing" half survives; the "and
*inert* to the other thing" half does not. Roughly **2 of the 4 orthogonality quadrants
survive**, and the surviving two are not orthogonal (both models end up responding to
both memory and market on a live board). The clean orthogonality of v1 was a property
of the frozen board, not a trait of the models. *Caveat:* gpt's memory pickup is
activity-only, and it co-occurs with the live board adding both churn and memory-able
content, so the memory/board effects are entangled (see synthesis note).

### Deliverable B (GATING) — Did the live board degrade gpt's discovery, or did v1 over-credit it? **The board degraded it. (Case 2.)**

This was the decisive question, so it is settled on *unbiased* metrics
(peak_top_pick, buy_flow), matched by seed, with floors — never on the
survivorship-biased top_pick.

| metric | model · mem | v1 | v2 | Δ (matched) | paired p |
|---|---|---|---|---|---|
| peak_top_pick | gpt · off | 0.472 | 0.361 | **−0.111** | **0.003** |
| peak_top_pick | gpt · on | 0.522 | 0.341 | **−0.181** | **0.000** |
| buy_flow | gpt · off | 0.460 | 0.351 | **−0.110** | **0.001** |
| buy_flow | gpt · on | 0.506 | 0.332 | **−0.174** | **0.000** |
| peak_top_pick | claude · off | 0.546 | 0.495 | −0.051 | 0.096 |
| peak_top_pick | claude · on | 0.595 | 0.579 | −0.015 | 0.597 |
| buy_flow | claude · off | 0.524 | 0.479 | −0.046 | 0.056 |
| buy_flow | claude · on | 0.543 | 0.468 | −0.076 | 0.000 |

v1 gpt was **clearly above chance** (peak 0.47–0.52, ≈5 SE over 0.333) — so v1 did *not*
over-credit it; the above-chance reading was **real but environment-dependent**, not
model-level (see three-bucket framework above). On the live board gpt drops to
**0.33–0.36 (chance)**, a significant matched-seed degradation in both memory
conditions. claude's discovery is **essentially unchanged on peak** (Δ −0.05 / −0.02,
not significant); **buy_flow mem-on** drops 0.543→0.468 (p=.000) — one significant
exception. The contrast is corroborated by **win** on the same seeds: gpt
0.500→0.356 (p=.000) and 0.523→0.328 (p=.000); claude mem-off *rises* 0.400→0.636
(p=.000 — the load-bearing win shift); mem-on 0.559→0.603 (p=.086, directional only).
Win **tracks** the discovery contrast on matched seeds — correlational triangulation,
not proven that gpt "wins less because it discovers less." The live board hurt the
weaker discoverer and helped the stronger one on the mem-off win axis.

This claim is supported on all four available axes of trust: it **clears the floor**
(paired p ≤ .003), **replicates across the memory condition** (both mems drop),
**correlated metrics move together** (peak + buy_flow + win all fall for gpt), and it is
**a clean cross-model contrast** (claude unaffected).

**Mechanism — stated with the confound.** The drop is *consistent* with board-chasing
(F3) replacing price-forecasting: a live board is noisier than the underlying price, and
if gpt follows the board it inherits that noise. But board-following is confounded with
price-following (the synthetic flow pushes the board toward the band the price is
already in; drift-band == price-band 54–85% of the time), so I cannot attribute the
degradation specifically to "reading the board" versus other live-board effects. What is
established is the *what* (gpt's unbiased discovery went from real to chance because the
board went live); the *why* is one decorrelated-flow experiment away.

### Deliverable C — Did gpt's $333.33 anchor dissolve on a live board? **Completely.**

Same model, prompt, and budget; only the board differs.

| | v1 (frozen) | v2 (live) |
|---|---|---|
| gpt trades at exactly $333.33 | **59%** (all trades) / 74% (buys); per-cell median = $333.33 in **every** cell | **0%** (both denominators, both mems) |
| gpt sizing in v2 | — | reactive **$50-multiples** (65% all / 72% buys), size CV 0.33–0.34 |

The budget/3 anchor was a **no-signal fallback**, not a hard prior: when there is a live
board to react to, gpt abandons the fixed third-of-budget bet entirely and sizes in
round increments off the board. This is the single cleanest illustration that a "v1
trait" was partly a v1-*environment* artifact. The counts are exact, so the
illustration holds.

### Deliverable D — Intersection (true model-level) vs single-battery (environment-dependent)

**Confirmed-robust in BOTH batteries → model-level results** (these survived an
environment change, a stronger test than memory replication):

1. **claude discovers the winning band better than gpt on unbiased metrics.** v1 peak
   gap +0.075 (p=.047) mem-off, +0.073 (p=.047) mem-on; v2 +0.131 (p=.001) / +0.223
   (p=.000). Same direction in every cell; v1 mem-off buy_flow gap +0.064 (p=.056) is
   marginal. Larger v2 gap only because gpt collapsed, not because claude improved
   (claude's own peak is 0.55/0.60 in v1, 0.50/0.58 in v2).
2. **A memory-driven win gain is commitment, not accuracy** — on the win decomposition.
   v1: held 0.55→0.82, win 0.40→0.56, but P(win|held) 0.73→0.68 (flat/down). v2: held
   0.85→0.92, win 0.64→0.60, P(win|held) 0.73→0.65. Winning moves through holding,
   not through better picks — both environments. **Tension:** v2 memory also lifts
   claude peak_top_pick +0.08 (p≈.021) — one marginal metric against >100 tests; named
   here, not promoted; does not overturn the cleaner win-decomposition story.
3. **End-state conditioning inflates discovery (the top_pick lesson).** In both, gpt's
   top_pick sits above its unbiased peak (v1 0.53/0.60 vs peak 0.47/0.52; v2 0.46/0.52
   vs peak 0.36/0.36), and the inflation is largest where the undefined/exit-flat count
   is highest. The survivorship correction matters identically in both batteries.
4. **knowability→discovery is withdrawn for the same ceiling reason in both.** The
   k0→k120 top_pick drop fails the floor in 3/4 v1 cells and 4/4 v2 cells, and the
   oracle close-band-hit ceiling is *identical* (100%@k=0 → 60%@k≥60) because it is a
   property of the shared price paths, not the model. Same structural cause both times.
5. **Two stable personalities, directionally.** gpt = low-activity, high-hold,
   round-sized stakes; claude = high-activity, low-hold, round sizing. The ordering
   holds in both (gpt tc 1.49/2.01 < claude 3.01/3.82; gpt hold 0.70/0.59 > claude
   0.40/0.24). Magnitudes shift on a live board; rank order never flips. The budget/3
   *anchor* is not in this bucket — it is a frozen-board artifact (Deliverable C).

**Environment-dependent capability** (real in v1, destroyed or altered in v2 — *not*
an artifact, *not* model-level):

- **gpt frozen-board discovery.** Unbiased peak/buy_flow ≈ 0.47–0.52 in v1 (~5 SE above
  chance); chance-level in v2. Deliverable B's gating result: v1 did not over-credit;
  the live board degraded real discovery. Most actionable customer-1 lesson: a model's
  task competence can depend on the venue.

**Holds in ONE battery only → frozen-board artifacts** (never robust traits):

- **Orthogonality of sensitivity** (gpt memory-inert; claude market-inert): **v1 only.**
  Dissolves in v2 (Deliverable A).
- **gpt's $333.33 anchor:** **v1 only.** 0% in v2 (Deliverable C).
- **Market-axis hierarchy:** band_width is the dominant, replicating market lever in v1
  but null in v2; volatility is null in v1 but dominant in v2. An **inversion**, not a
  trait (below).

---

## 3. Market-axis inversion and the synthesis hypotheses (briefer)

**The band_width ↔ volatility inversion.** v1 ranks the market axes
*band_width > temperature > volatility ≈ knowability (null)*; v2 ranks them
*volatility ≫ temperature ≈ band_width ≈ knowability (null)*. The natural reading: which
market feature dominates depends on whether live flow exists. On a **frozen** board the
only structural lever that changes the model's static reasoning is the band *geometry*
(band_width), so it dominates. On a **live** board, the path's volatility drives the
*magnitude of synthetic flow* (a more volatile path → a busier, more mobile board), and
that flow-driven board movement swamps the static geometry. **Confound (stated):** the v2 volatility effect is therefore confounded with
synth-flow magnitude *and* with the price line the model reads. gpt's apparent
volatility sensitivity cannot be cleanly separated from responding to a *busier board*
driven by that flow — it is entangled with board busyness, not established as
"volatility-the-concept." The inversion is a real observation; its v2 half should not
be read as clean market perception.

**v1's self-inconsistency hypothesis** ("memory bites a model in proportion to how much
it would otherwise contradict itself," predicting memory would start to bite gpt once it
churns). v2 is *weak, confounded corroboration*: on matched seeds gpt churns more
(trade_count 1.49→2.01, p=.000) **and** picks up memory sensitivity (1→4 activity
metrics clearing). But the churn increase is caused by the live board, which is also
what supplies memory-able content, so the two are entangled; and gpt's memory pickup is
activity-only (discovery/win memory gaps stay null in both batteries). Suggestive, not
proven.

**v2's "board pulls toward the move" hypothesis** is likewise untested — one
conditioning analysis (pure-synth turns only) away from a real test. Neither hypothesis
is promoted to a finding here.

---

## 4. True model-level findings (highest confidence)

These are the Deliverable-D intersection — claims that held under both a memory flip
**and** an environment flip, which is the strongest robustness evidence the two
batteries can jointly provide:

- **claude finds the winning band better than gpt** on survivorship-free metrics
  (peak_top_pick, buy_flow), in both batteries and both memory conditions
  (directionally every cell; v1 mem-off buy_flow marginal at p=.056).
- **Memory's effect on winning is mediated by commitment (holding longer), not by
  improved conditional accuracy** — true for claude in both batteries on the win
  decomposition. **Tension:** v2 memory lifts claude peak_top_pick +0.08 (p≈.021) —
  one marginal hit; named, not promoted alongside the cleaner win story.
- **End-state-conditioned discovery (top_pick) overstates skill**; the
  peak/buy_flow correction is necessary and gives lower, honest numbers in both
  batteries.
- **knowability does not create or destroy discovery** — the apparent drop is a
  measurement ceiling (close-band-hit 100%→60%) baked into the shared price paths;
  withdrawn for the same reason in both.
- **Two distinct, stable trading personalities** (gpt cautious/high-hold/round-sized;
  claude active/low-hold/mobile) in directional rank, robust to the environment. The
  budget/3 anchor is v1-only (artifact).

## 5. Environment-dependent findings

### 5a. Environment-dependent capabilities (real, contingent on venue)

- **gpt frozen-board discovery.** Genuinely above chance in v1; collapsed to chance in
  v2 (Deliverable B). Not an artifact (audit ruled out over-credit); not model-level
  (does not survive the environment flip). Competence can depend on the venue.

### 5b. Frozen-board artifacts (never robust traits)

These looked like model traits in v1 but turned out to be properties of the frozen
board:

- **The orthogonality of sensitivity.** gpt is memory-inert and claude is market-inert
  *only* on a frozen board. With live flow, gpt gains memory sensitivity (in activity)
  and claude gains market (volatility) sensitivity.
- **The $333.33 budget/3 anchor.** A no-signal fallback that vanishes (→0%) the moment
  the board moves.
- **The market-axis hierarchy.** band_width dominates a frozen board; volatility-driven
  flow dominates a live one. The ranking inverts with the environment.

## 6. Open questions and what would settle them

1. **Price vs. board (the one confound F3 cannot resolve).** Synthetic flow pushes the
   board toward the band the price is already in (drift-band == price-band 54–85% of the
   time), and the model reads both. The clean test (turns where the model held the
   previous turn ⇒ the board move is pure synthetic flow) has too little data
   (gpt n=16/24; claude n≈0) to decide. **Settle it** with a purpose-built run where
   synthetic flow is *decorrelated* from price — flow that pushes a band the price is
   *not* in. If the model still follows the flow, it is genuinely reading the board; if
   it follows the price, the "board-tracking" finding is really price-following. This
   same experiment would also clarify whether gpt's discovery degradation (Deliverable
   B) is board-chasing specifically or just the live board adding noise.
2. **The self-inconsistency hypothesis (v1).** Does memory bite gpt *because* it churns?
   **Settle it** by regressing the per-model memory effect on the model's own
   turn-to-turn contradiction rate; v2's churn↑ + memory↑ co-occurrence is consistent
   but confounded.
3. **The "board pulls toward the move" hypothesis (v2).** **Settle it** by conditioning
   board-follow on pure-synth turns only (needs the decorrelated-flow run for adequate
   n, per #1).
4. **claude's win jump on the live board** (0.40→0.64 mem-off, matched p=.000). Is it
   better discovery or just that a busier board plus claude's holding lands it on
   winners more often? **Settle it** by conditioning the win gain on peak_top_pick vs
   held_rate — i.e., does the extra winning come from picking the winner more, or from
   holding whatever it picked through to settlement?

---

## Addendum (2026-07-08) — the chance floor was the wrong null; the collapse is an entry-timing effect

*A second independent review, run after this document was finalized. Standalone
read-only reducer: `eval_runs/_review_naive_baseline.py`. It regenerates every price
path from `(cell config, seed)` and validates the reconstruction against the stored
settlement price of every episode — **0 mismatches across all 3,120 episodes** — then
computes counterfactual baselines against the raw `runs.jsonl`. Nothing in the Phase 1
audit is overturned: every number in this document still reproduces. What changes is
the interpretation of the discovery results, in one specific way.*

### A1. The 0.333 floor is not the right null for buy_flow / peak_top_pick

Every discovery claim above is tested against chance = 1/3. But the winning band is
the band the settlement price lands in, and the price path is *observable*: a
zero-skill strategy that buys whatever band the price is currently in is strongly
correlated with the winner, increasingly so as the episode progresses. Averaged over
all five decision points the current band matches the eventual winner **59.3%** of the
time (identical in both arms — a property of the shared paths, and consistent with the
knowability oracle ceiling of ~60% reported above).

The decisive comparison is the naive current-band follower evaluated **on the exact
same buys the model made** (same turns, same notional weights):

| arm | model · mem | actual buy_flow | naive price-follower, same buys | unconditional per-turn |
|---|---|---|---|---|
| v1 | gpt · off | 0.487 | **0.508** | 0.593 |
| v1 | gpt · on | 0.536 | **0.562** | 0.593 |
| v1 | claude · off | 0.570 | **0.557** | 0.593 |
| v1 | claude · on | 0.570 | **0.579** | 0.593 |
| v2 | gpt · off | 0.362 | **0.341** | 0.593 |
| v2 | gpt · on | 0.355 | **0.335** | 0.593 |
| v2 | claude · off | 0.491 | **0.480** | 0.593 |
| v2 | claude · on | 0.483 | **0.479** | 0.593 |

(These are notional-weighted pooled ratios, not the per-episode means used elsewhere
in this document, so the actual columns differ slightly from the tables above; the
actual-vs-naive comparison is apples-to-apples within each row.)

**Both models, both batteries, sit essentially at their timing-matched naive
baseline** (residuals −0.03 to +0.02; gpt v1 is slightly *below* it). Neither model
ever demonstrated band-picking skill beyond "buy the band the price is in now." The
v1 "genuinely above chance, ≈5 SE over 0.333" reading is arithmetically correct, but
0.333 is a null no price-reading trader would fail.

### A2. What actually collapsed in v2: entry timing, not band-picking

The naive baseline itself drops from ~0.51–0.56 (v1 gpt buys) to ~0.34 (v2 gpt buys).
On seed-identical paths that can only mean the buys moved to different turns. They
did — the share of buy notional placed at t=0:

| arm | gpt · off | gpt · on | claude · off | claude · on |
|---|---|---|---|---|
| v1 | 16% | 12% | 16% | 11% |
| v2 | **84%** | **82%** | **51%** | **44%** |

At t=0 the price sits at the anchor, dead center of the middle band; no strategy can
beat 1/3 there (empirically, t=0 buys hit the winner 23–31% in every arm × model ×
memory cell). gpt front-loads almost everything to t=0 on the live board; claude
front-loads half but keeps buying at t=60–90, where the current band is informative.
That *is* the "gpt collapsed, claude survived" contrast of Deliverable B — explained
almost entirely by timing.

**The proximate trigger is a design wrinkle: synthetic flow trades before the model's
first turn.** `opening_probabilities` are stored uniform, but the turn-0 prompt
already shows a tilted board (e.g. 25.8/25.8/48.4). At t=0 the flow has seen only the
anchor price, so that tilt is pure noise — yet gpt buys the tilted favorite at t=0
**72%** of the time (n=372 mem-off / 370 mem-on t=0 buys). claude buys the middle band
at t=0 100% of the time, ignoring the tilt, which preserves its late informative buys.
The model's very first observation in v2 is a spurious signal; gpt anchors on it and
commits immediately.

### A3. Revisions to the buckets (what changes, what does not)

- **Deliverable B (gating).** The *what* survives — the matched-seed drop is real and
  the p-values stand. Both halves are reframed: the v1 "capability" was
  timing-sensible price-following, not forecasting; and the degradation mechanism is
  now substantially identified *within existing data* — the t=0 noise tilt induces
  premature commitment at the one moment nothing is knowable. This partially answers
  open questions #1 and #3 without the v3 run (the v3 decorrelated-flow experiment
  remains the clean confirmation).
- **Model-level bucket, claim 1 ("claude discovers better than gpt").** Weakened as a
  *band-picking* claim: relative to timing-matched baselines the residual gap is small
  (claude ≈ +0.01, gpt ≈ −0.02). The robust model-level trait is better stated as:
  *claude distributes trades across the episode and resists spurious board tilts; gpt
  concentrates its entry and anchors on them.* A timing/attention trait, not a
  band-picking skill gap.
- **Unchanged:** memory = commitment-not-accuracy (the win decomposition does not use
  the chance floor); the two personalities; the $333.33 anchor dissolution; the
  top_pick survivorship lesson; the knowability measurement ceiling (independently
  corroborated by the 0.593 per-turn naive rate).
- **Discovery 5 framing (literature doc).** "More live information can degrade
  performance" becomes sharper and more specific: *a salient spurious signal at first
  decision induces premature commitment in gpt-4o but not claude-sonnet-4* — closer to
  the LLM anchoring-bias literature than to a generic information effect.

### A4. Provenance and between-arm confounds (recorded, judged second-order)

1. **v1's recorded git commit is wrong.** The manifest records `7a2182c`, but
   `prompt.py` at that commit lacks the settlement-timing disclosure that v1's stored
   prompts contain — v1 was run from a dirty working tree. The stored prompts are
   ground truth and confirm template-identity across arms (turn-0 prompts differ only
   in the board numbers), so no result changes; the provenance chain, however, is not
   trustworthy on its own.
2. **The arms differ by more than "board alive."** v1: no provider routing,
   concurrency 1. v2: `ignore azure` routing, concurrency 10. The same model slug
   served through different provider stacks can behave differently. Judged
   second-order next to the timing effect, but "identical with one change" is not
   strictly true.

### A5. Recommendations for v3

- Report every discovery metric as **skill-over-naive** (timing-matched
  price-follower), not skill-over-1/3.
- **Do not let synthetic flow act before turn 0** (or randomize its start), so the
  model's first observation is not a pure-noise tilt — unless first-impression
  anchoring is itself the object of study, in which case make it a deliberate axis.
- Require a **clean git tree** for official runs; pin provider routing and
  concurrency identically in both arms.

*Reproduce the addendum: `PYTHONPATH=src python3 eval_runs/_review_naive_baseline.py`
(standalone, read-only; validates path reconstruction against every stored settlement
price before computing baselines).*

---

*Reproduce: `PYTHONPATH=src python3 eval_runs/_audit.py` (per-battery audit, all
non-perm + permutation floors) and `PYTHONPATH=src python3 eval_runs/_phase2.py`
(matched-seed v1↔v2 paired contrasts). Both are standalone and read-only.*
