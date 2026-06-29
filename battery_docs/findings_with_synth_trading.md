# Master Findings — Battery WITH Synthetic Trading (v2)

The consolidated read of the two-model behavioral battery run in a **live-board**
environment: between the model's turns, synthetic ("noise") traders move the
implied-probability board on their own, so the odds the model sees are no longer
frozen between its decisions. This document is **self-contained** — it analyzes
this environment on its own terms and does not depend on any other battery.

**Battery:** `eval_runs/battery_WITH_SYNTH_TRADING/2026-06-29_official-v2/` — **52 cells, all complete.**
- **2 models:** `openai/gpt-4o`, `anthropic/claude-sonnet-4`
- **2 memory conditions:** mem-off (stateless) and mem-on (prior rationales replayed)
- **13 cells each:** baseline + 4 volatility + 3 temperature + 3 knowability + 2 band_width
- **N = 30 runs/cell** (1,560 episodes total), faithful prompt (`disclose_dead_window=True`)
- 3-band centered arithmetic random-walk market, anchor=100, 2h window, 5 decisions, budget=1000.

**The live board.** Synthetic traders place orders through the *same* DPM
`apply_trade` the model uses, under a separate address, so their flow moves the
shared share pool — and therefore the board everyone sees — **without** being
counted as the model's position. They read **only the revealed price path** (never
the settlement outcome), target "which band the price is in now" nudged by a small
momentum tilt, flip to a contrarian band with a per-episode probability drawn once
(~5–30%), and are capped per-trade / per-interval / per-episode (~$1000 total,
≈ the model's own budget, paced evenly across turns). Disclosure is **implicit**:
the moved board shows up only through the existing implied-probability line, so the
prompt template is byte-identical to a frozen-board run — only the probability
numbers differ. The board moving on its own is the *single* feature of this world.

All reads use the project discipline: **effect-vs-spread over significance**,
**identity (sizing-independent) over magnitude**, **permutation floors** under
every candidate effect (20k shuffles), and **replication across the memory
condition** as a second, independent robustness test. Money (pnl) is out of scope
and reported only as context.

---

## The one-paragraph result

On a live board, the two models remain **two stable, sharply different traders**:
gpt-4o is concentrated, larger-stake, moderately active and **barely discovers the
winner** (chance-level on unbiased metrics); claude-sonnet-4 is active,
diversified, smaller-stake, deploys nearly its full budget, and **genuinely
discovers** the winning band well above both chance and gpt. The dominant
behavioral lever here is **volatility** — it moves gpt on five correlated metrics
in *both* memory conditions — but that read is **confounded with the synthetic
flow** (a more volatile path produces a busier board), so gpt's apparent volatility
sensitivity **cannot be cleanly separated from responding to a noisier board** — it
is entangled with flow-driven board busyness, not established as volatility-the-concept.
Board-movement tracking was billed as this environment's signature, but **[see the F3
audit correction: the metric failed to isolate synthetic flow from the model's own
prior trade; on the clean definition board-tracking is only weakly supported for gpt
and is currently *unestablished* for claude.]** **Memory** is a model-specific
modifier: it nudges gpt toward more, smaller, churnier trading (and more exiting
flat), and pushes claude further toward committed accumulation (fewer reversals, less
exit-flat, more buckets). **Tension:** memory also lifts claude peak_top_pick +0.08
(p≈.021) — one marginal hit against >100 tests; named in F6, not promoted alongside
the cleaner commitment-not-accuracy win story. Several tempting leads
**did not survive**: a knowability→discovery drop (observability ceiling; fails the
floor in both memory conditions), a temperature→activity effect (directional but
does not clear the floor), band_width (non-replicating), and the "gpt goes all-in
out of instability" anecdote (all-in is rare, is actually *most* common in claude
mem-on, and its direction **follows the board** — signal-chasing, not sizing chaos).

---

## Methods that did the work

Beyond the standing "split a double-meaning statistic before trusting it"
discipline, six instruments carried this round:

1. **The built-in noise floor (4 identical-config draws).** Baseline is
   config-identical to three sweep midpoints — `volatility=1.0`, `knowability=0`,
   `temperature=0.7` are the *same cell*. That yields **four independent 30-run
   draws of one configuration** per model×mem, whose spread is the true
   within-config noise. Every axis effect is judged against *this measured floor*.
   (mem-off floor: gpt trade_count spread 0.37, volume 102, median-size 12; claude
   0.13 / 52 / 10.)
2. **End-state-independent discovery.** `top_pick` conditions on the *closing*
   position, so it is undefined whenever a model exits flat. We use `peak_top_pick`
   (winner == top holding at the turn of maximum gross exposure) and
   `buy_flow_on_winner` (fraction of buy notional aimed at the winner); both are
   defined for any run that traded, so the models are scored on comparable
   denominators (390 vs 390, not 294 vs 331).
3. **Permutation floors** under every candidate effect (20k shuffles), including a
   same-config null sanity (baseline vs knowability=0) that must *not* fire.
4. **Replication across memory.** mem-off and mem-on are two independent passes; a
   real effect should appear in both.
5. **Board-reaction split test (this environment only).** For every buy after the
   first turn, we measure whether the target band equals the current *favorite*
   (board level) versus the *synth-drift* band (`argmax(prob_before − the board the
   model left last turn)`, i.e. the pure flow move since it last acted). The clean
   test restricts to buys where **drift ≠ favorite** — the board moved *away* from
   the leader — and asks which the model follows. Confound stated up front:
   synth-drift correlates with the underlying price the model can also read
   directly, so drift-alignment is an **upper bound** on board-reading.
6. **All-in by intent, not rejection.** An "all-in attempt" is a BUY whose
   requested size ≥ 90% of available cash, counted whether it *filled or was
   rejected* (a rejection is just an all-in that clipped the 2% fee — counting only
   rejections measures a rounding artifact). Direction is then checked against the
   board move.

Scripts (read-only, no model calls): `eval_runs/master_analysis.py`,
`eval_runs/knob_diagnostics.py`, `eval_runs/within_label_check.py`,
`eval_runs/board_reaction.py`, `eval_runs/allin_check.py`. Each takes the v2
session directory as an argument.

---

## F1 — Two stable trader personalities (pooled over all 13 core cells)

| metric | gpt-4o off | gpt-4o on | claude off | claude on |
|---|---|---|---|---|
| trade_count | 2.01 | 2.18 | **3.82** | **3.88** |
| hold_rate | **0.59** | **0.56** | 0.24 | 0.22 |
| total_volume ($) | 655 | 689 | **980** | **1009** |
| median_trade_size ($) | **343** | 324 | 250 | 259 |
| direction_switches | 0.61 | 0.67 | **1.84** | **1.44** |
| outcomes_traded | 1.15 | 1.17 | **1.70** | **1.83** |
| exit_flat_rate | 0.25 | 0.36 | 0.15 | 0.08 |
| win *(context only)* | 0.36 | 0.33 | 0.64 | 0.60 |

**gpt-4o** = concentrated / larger-stake / moderately active: ~2 trades, holds
~57% of decisions, almost a single bucket (1.15), ~$340 stakes, rarely reverses.
**claude** = active / diversified / churning accumulator: ~3.8 trades, holds ~23%,
~1.7 buckets, smaller (~$250) stakes, reverses direction ~1.4–1.8×/run, deploys
~98% of budget.

These are **trait-level constants**, not draws. The noise floor (§Methods 1) shows
the spread across four identical-config draws is tiny (gpt trade_count 0.37, claude
0.13; median-size 12 and 10), while the cross-personality gaps are an order of
magnitude larger and hold in **every** cell (claude's per-cell median trade_count
sits at 3.6–4.2 throughout; gpt's at 1.5–2.6). That level of cross-config constancy
cannot be sampling.

## F2 — Sizing fingerprints

- **Both models size in round increments**, but at different scales: **~65–72%** of
  gpt trades and **~86–100%** of claude trades are exact multiples of $50 *(corrected
  — see note)*. gpt centers near **$350** with no single dominant value (it does
  **not** pin to a fixed fraction such as budget/3 — 0% of gpt trades land on $333.33
  here); claude centers smaller (~$200–300).

  > **Audit correction (2026-06-29):** the original text said "~80%" for gpt. Independent
  > recomputation gives gpt **65% over all executed trades, 72% over buys only** — the
  > "~80%" was high. claude is **86% (all trades) / ~98% (buys)**, so "~95%" is
  > reasonable. The qualitative point (claude sizes in round increments more rigidly
  > than gpt) holds; the **0% at $333.33** figure is exact and unchanged.
- **claude's total deployment is far more consistent** (run-level volume CV **0.21–0.26**
  vs gpt **0.40–0.41**), because it reliably makes ~3.8 trades while gpt's count
  swings and it occasionally dumps a large stake. Per-trade dispersion is similar
  (size CV ≈ 0.45 both), so gpt is not *more precise* per trade — its **total**
  footprint is the more erratic one run-to-run, the paradoxical cost of being the
  lower-activity trader.

## F3 — Board-movement tracking (the signature of the live board)

> **⚠ AUDIT CORRECTION (cross-battery audit, 2026-06-29) — this finding is demoted.**
> The records contain **no `probabilities_after` field**, so the board-reaction script
> silently fell back to the *prior turn's pre-trade board*. What is labeled
> **"synth-drift"** below therefore measures the **total board move since the model's
> last decision — including the model's *own* previous trade** — not pure synthetic
> flow. (The board only moves on trades, so the inter-turn move is
> `own_prev_trade + synth_flow`; it is pure synthetic flow *only* on turns where the
> model **held** the previous turn.) Consequences:
> - The rates in the table below are correct **as "follows the move since its last
>   decision,"** but they do **not** isolate reading the synthetic flow from following
>   the consequences of the model's own prior buy.
> - On the **clean** definition (model held the previous turn ⇒ move is pure synthetic
>   flow, restricted to drift ≠ favorite): **gpt** still follows the drift at **69%
>   (mem-off, n=16)** and **83% (mem-on, n=24)** — directionally consistent but tiny-n;
>   **claude is untestable — n ≈ 0 / 6** — because it almost never idles, so its clean
>   subset is empty.
> - This compounds the price confound already stated at the end of the section
>   (drift-band == current price-band **54–85%** of the time).
>
> **Revised claim:** live-board tracking is **weakly supported for gpt** (tiny-n clean
> subset, still confounded with price) and **currently *unestablished* for claude**.
> The headline that *both* models track the live board does **not** hold on the clean
> definition. Treat F3 as suggestive, not as the established "signature" it is billed
> as above and in the one-paragraph result. The decorrelated-flow v3 run
> (`v3_decorrelated_flow_spec.md`) is the experiment that would settle it.

This is the finding the frozen-board world could not produce. With the board
moving on its own between turns, **both models, when they buy, target the band the
board has just been pushed toward** — and they do so *even against the standing
favorite*. Pooled over all core cells, for buys after turn 1 where the board moved:

| | gpt off | gpt on | claude off | claude on |
|---|---|---|---|---|
| buys-with-drift (n) | 100 | 106 | 434 | 666 |
| target == favorite (level) | 54% | 58% | 68% | 59% |
| target == synth-drift (move) | **71%** | **74%** | **84%** | **74%** |
| **SPLIT: follows *drift* when drift ≠ favorite** | **74%** | **68%** | **71%** | **65%** |
| SPLIT: follows *leader* when drift ≠ favorite | 24% | 24% | 15% | 21% |

All alignments clear the permutation floor at p=0.000. The **split rows are the
load-bearing ones**: when the board moves *away* from the current favorite, both
models follow the **movement** (65–74%) and abandon the **leader** (15–24%) — a
~3:1 preference for the direction the board is going over its standing level, in
all four arms. So the models are not merely "buying the leader"; they are tracking
the *change*.

**Kept, but with the confound stated plainly.** The synthetic flow itself targets
"which band the price is in now," so the drift band is correlated with the
underlying price the model can read directly off the price line. This test cannot
fully separate **reading the board's movement** from **reading the same price the
board read**; drift-alignment is an upper bound on board-reading per se. The honest
claim is: *both models reliably aim new capital in the direction the market is
moving, not at its standing favorite.* (Note claude's much larger n reflects F1 —
it simply makes far more later-turn buys for the test to see.)

## F4 — Discovery: claude genuinely finds the winner; gpt is at chance

Scoring both models on **comparable, end-state-independent** denominators
(chance = 0.33):

| metric | gpt-4o | claude | gap [perm p] |
|---|---|---|---|
| `top_pick` (end-state, **biased**) mem-off | 0.46 (n=294) | 0.73 (n=331) | +0.266 [0.000] |
| `peak_top_pick` mem-off | **0.36** (n=390) | **0.50** (n=390) | +0.131 [0.001] |
| `buy_flow_on_winner` mem-off | **0.35** | **0.48** | +0.125 [0.000] |
| `peak_top_pick` mem-on | **0.36** (n=390) | **0.58** (n=390) | +0.223 [0.000] |
| `buy_flow_on_winner` mem-on | **0.35** | **0.47** | +0.120 [0.000] |

Two clean results:

1. **gpt-4o does not discover the winner.** On both end-state-independent metrics it
   sits at **0.35–0.36, essentially chance (0.33)** — in both memory conditions. Its
   above-chance `top_pick` (0.46–0.52) is **pure survivorship**: it only scores runs
   that held to the bell, and those are the runs it happened to get right.
2. **claude discovers, and the edge survives the survivorship fix.** peak_top_pick
   (0.50/0.58) and buy_flow (0.48/0.47) are both well above chance and **+0.12 to
   +0.22 above gpt**, clearing the floor at p≤0.001 and **replicating across the
   memory split** (two independent passes). This is corroborated by a third,
   structurally different instrument — conditional accuracy P(win|held) ≈ **0.69–0.73
   for claude vs 0.46–0.51 for gpt** (F6) — so position-state-at-peak, capital-flow,
   and held-accuracy all agree.

Unlike the discovery edge in a frozen-board world (which was marginal), here it is
**unambiguous on the unbiased metrics**, not threshold-dependent.

## F5 — Volatility is the dominant axis (gpt), but confounded with the flow

Ranked by floored, replicated strength, the market-axis hierarchy in this
environment is **volatility ≫ temperature ≈ band_width ≈ knowability (nulls)** —
the reverse of what a frozen board would suggest, and the reversal is the point.

1. **volatility — strongest and the only axis that replicates across memory (gpt).**
   Raising volatility moves gpt on **five correlated metrics in *both* memory
   conditions**: trade_count (0.5→2.0: 1.47→2.40 mem-off p=0.000; 1.73→2.47 mem-on
   p=0.010), hold_rate down (p=0.000 / 0.008), total_volume up (p=0.006 / 0.006),
   median_trade_size down (p=0.003 / 0.027), direction_switches up (p=0.001 / 0.004).
   The move is mostly a **low-volatility suppression** (vol=0.5 is the quiet cell;
   activity plateaus from vol≥1.0). claude is essentially flat on the volatility
   *label* (only scattered mem-off hits, none mem-on).
   - **Confound (load-bearing).** The synthetic flow reads the price path, so a more
     volatile path produces a **busier, more mobile board**. gpt's volatility
     response is therefore inseparable here from "gpt responds to a livelier board"
     (and F3 shows gpt does track board movement). This is a real, replicated effect
     on gpt's behavior, but its **mechanism is confounded by construction** and
     cannot be attributed to volatility perception alone.
2. **temperature — directional only, does NOT clear the floor.** gpt activity rises
   with temperature in the means (trade_count 1.80→2.33 mem-off, 2.07→2.57 mem-on;
   stake shrinks 366→302), but the endpoint permutation floor does **not** clear in
   either memory condition (0→1.2 trade_count p=0.090 mem-off, p=0.103 mem-on). On
   this battery's discipline, temperature is **not** a load-bearing behavioral lever
   here. claude is flat.
3. **band_width — non-replicating.** The only sub-0.05 hit is gpt mem-on total_volume
   (526→714, p=0.004); mem-off does not clear and no other metric moves. An effect in
   one memory condition only is treated as noise. claude flat.
4. **knowability — behavioral null** (and its discovery story is withdrawn, F7).

## F6 — Memory is a model-specific modifier (both models move, differently)

Memory (replaying the model's own prior rationales) shifts **both** models here, in
opposite characters (mem-on − mem-off, pooled, permutation p):

| metric | gpt-4o gap [p] | claude gap [p] |
|---|---|---|
| trade_count | **+0.17 [0.026]** | +0.06 [0.257] |
| hold_rate | **−0.04 [0.017]** | −0.02 [0.086] |
| median_trade_size | **−18.7 [0.029]** | **+9.3 [0.014]** |
| direction_switches | +0.06 [0.194] | **−0.40 [0.000]** |
| outcomes_traded | +0.02 [0.562] | **+0.13 [0.001]** |
| exit_flat_rate | **+0.12 [0.000]** | **−0.07 [0.002]** |
| peak_top_pick (discovery) | −0.01 [0.883] | **+0.08 [0.023]** |

- **gpt:** memory makes it **trade a bit more, in smaller stakes, and exit flat more
  often** (four metrics clear). The effects are small — trade_count's +0.17 is the
  same order as its mem-off noise floor (0.37) — so this is a *modest, coherent*
  nudge toward more fragmented activity, not a transformation. exit_flat (+0.12,
  p=0.000) is the most robust single gpt memory effect.
- **claude:** memory pushes the churner further toward a **committed accumulator** —
  **−0.40 direction_switches** (the strongest effect, p=0.000), less exit-flat
  (−0.07), bigger stakes (+$9), more buckets (+0.13), and a **marginal peak_top_pick
  lift (+0.08, p≈.021)**. It reverses itself less and holds onto more.

> **Memory→discovery tension (named, not buried):** the commitment-not-accuracy
> throughline below holds on the win decomposition, but peak_top_pick +0.08 is a real
> seam — memory may touch claude's *discovery* on a live board in a way frozen-board
> data never showed. Against >100 tests at p≈.021 it is exactly the kind of single-metric
> hit this project withdraws; I lean (b) do not promote it, but (a) environment-dependent
> memory→discovery remains an open reconciliation. See `battery_comparison_v1_vs_v2.md`.

**The win column is commitment, not accuracy (decomposed, not interpreted).** `win`
(held the winning band at the bell) is end-state-conditioned exactly like
`top_pick`, so split it `win = held_rate × P(win | held)`:

| | gpt off | gpt on | claude off | claude on |
|---|---|---|---|---|
| held_rate `P(held)` | 0.75 | 0.64 | 0.85 | **0.92** |
| `win` `P(win)` | 0.36 | 0.33 | 0.64 | 0.60 |
| **cond-accuracy `P(win\|held)`** | 0.46 | 0.51 | **0.73** | **0.65** |

claude's holding rises with memory (0.85→0.92) while its **conditional accuracy is
flat-to-down** (0.73→0.65): memory makes claude *commit*, not *pick better*. gpt
moves the other way — memory **lowers** its holding (0.75→0.64, the exit-flat rise),
so its win edges down. The cross-model accuracy gap (claude ~0.69 vs gpt ~0.49) is
the same discovery signal as F4, seen through the outcome. *Money (pnl) is out of
scope; claude is net-positive and gpt net-slightly-negative here, not significant.*

## F7 — Volatility, the realized-range version (claude, now memory-replicated)

The volatility *label* sweep is a lossy proxy for the *realized* price range a run
actually drew. Binning a single label's 30 episodes by realized range (calm vs
wild) and permutation-testing the stake-dispersion gap:

- **claude, mem-off:** all-but-one label shows a positive (wild → more dispersed
  sizing) gap; vol=2.0 clears (+8.9, p=0.044).
- **claude, mem-on:** **4/4 labels positive**, vol=2.0 clears (+18.6, p=0.032), and
  trade_count also clears at vol=0.5 and vol=2.0 (p≈0.046–0.049).
- **gpt:** no consistent direction, nothing clears.

So claude **modulates stake dispersion with the magnitude it actually sees** — a
sensitivity the flat label sweep hides — and here it appears in **both** memory
conditions (weak, but the direction is consistent and one label clears in each).
The reusable methodological point stands and now has live cross-model support:
**sweeping a design parameter understates sensitivity that only appears when you
condition on the realized path**, and gpt running the identical test and finding
nothing shows it is not a test artifact.

## F8 — Knowability: behavioral null + withdrawn discovery effect (observability ceiling)

A knowability→discovery drop (top_pick falling at the long dead window) is tempting
but **withdrawn** after three checks:

- **Mechanism (decisive).** The price path runs *through* the dead window, so
  settlement is set by *unseen* post-close diffusion. A model-independent oracle that
  reads the last observed price has a **close-band-hit of 100% at knowability=0 but
  only 60% at knowability ≥ 60** — the outcome literally leaves the observable
  window. A top_pick drop is therefore **the target becoming unknowable by
  construction, not discovery skill degrading.**
- **Floor.** The k=0→120 top_pick drop (scorable runs only) clears in **neither**
  model **nor** memory condition (gpt p=0.093 mem-off / 0.214 mem-on; claude p=0.092
  / 0.245).
- **Replication.** The behavioral knowability sweep is a clean null on every metric
  for both models mem-off; the only mem-on hits (gpt trade_count/hold/volume at
  k=120) do not replicate to mem-off and are treated as noise.

Knowability is thus a **behavioral null**, and its discovery axis is **unmeasurable
by construction** in this centered market — a stronger statement than "measured and
found zero."

## F9 — All-in behavior: rare, claude-led, and signal-chasing (anecdote withdrawn)

Counting all-in **attempts** (size ≥ 90% of cash, filled or rejected), pooled over
all 13 core cells per arm:

| arm | all-in attempts | % of buys | episodes with ≥1 all-in |
|---|---|---|---|
| gpt off | 18 | 3.8% | 17 / 390 |
| gpt on | 18 | 3.8% | 16 / 390 |
| claude off | 10 | 1.2% | 10 / 390 |
| claude on | **77** | **7.3%** | **77 / 390** |

Two corrections to the casual impression that "gpt goes all-in":

1. **It is rare and not a gpt signature.** All-in is 1–4% of buys for gpt and *most*
   common in **claude mem-on** (7.3%) — consistent with memory pushing claude toward
   commitment (F6), i.e. occasionally committing the whole stack.
2. **Its direction follows the board (signal-chasing, not instability).** In every
   arm, all-in buys align with the board-drift band far more than ordinary buys do
   (gpt-off 100% vs 65%; claude-on 82% vs 73%) and far above chance. So when a model
   goes all-in, it is **maximally backing the direction the board is moving** — the
   F3 movement-tracking behavior taken to its sizing extreme — rather than sizing
   chaotically. The "all-in = instability" reading is **withdrawn**; the evidence is
   Reading A (signal-chasing). (n is modest; treat as directional.)

## F10 — Sensitivity profiles in this environment

Tallying which interventions actually move each model (permutation-floored;
requiring correlated metrics and/or memory replication):

| | responds to **market** axes | responds to **memory** | tracks **board movement** |
|---|---|---|---|
| **gpt-4o** | **volatility only** (5 metrics, both mems) — *entangled with flow/busier board; not volatility-the-concept* | **yes, small** (activity up, stake down, exit-flat up) | **weak / suggestive only** (F3 audit correction; clean subset n≈16–24) |
| **claude** | **nearly no** label response (weak realized-range only, F7) | **yes** (churner→committer, 5 metrics) | **unestablished** (F3 audit correction; clean subset n≈0) |

The clean "gpt is market-driven, claude is memory-driven" split is **softened in
this environment**: gpt picks up a (small) memory sensitivity, claude stays
memory-leaning. gpt's apparent volatility sensitivity is **not** a clean read on a
different market lever — it is entangled with synthetic-flow-driven board busyness.
Board-movement tracking (F3) is **demoted** after the audit correction: rates in the
table are "follows the move since last decision," not pure synth-flow; claude is
untestable on the clean definition.

---

## Synthesis (hypothesis, not a finding): the live board pulls both models toward the move

F3, F5 and F9 point at one mechanism: **a board that moves on its own becomes a
salient signal both models chase.** New buys aim where the board is going (F3); the
one "market" axis that bites is the one that makes the board move more (F5,
volatility); and the most extreme sizing events back the board direction (F9). A
falsifiable prediction within this world: in episodes where the per-episode
contrarian fraction happened to be high (the synthetic flow pushed *against* the
price more often), movement-tracking buys should be *less* accurate at finding the
winner — if the models are chasing the board rather than the underlying price, their
discovery should track the board's correctness, not the price's. That test would
separate "reads the board" from "reads the price the board reads," the one confound
F3 cannot resolve. **Flagged as a hypothesis; one conditioning analysis away from a
test, not asserted here.**

---

## Scorecard

**Confirmed (robust):**
- Two stable, distinct trader personalities; trait-level, config-invariant (F1).
- Round-$50 sizing both models; claude far steadier total deployment (F2).
- claude genuinely discovers the winner (peak_top_pick, buy_flow, conditional
  accuracy all agree, replicated across memory); **gpt is at chance** (F4).
- Memory modifies both models: gpt → more/smaller/churnier + exit-flat; claude →
  committed accumulator (−reversals, −exit-flat) (F6). claude's win gain on the
  decomposition is **commitment, not accuracy** — with the peak +0.08 tension named
  in F6, not promoted.

**Confirmed (weak / scoped):**
- Board-movement tracking (F3): **demoted** after audit — rates are "follows move since
  last decision," not pure synth-flow; weakly suggestive for gpt (clean n≈16–24);
  **unestablished for claude** (clean n≈0). Price/board confound 54–85%.
- volatility → gpt engagement, replicated across memory, **entangled with synthetic
  flow and board busyness — not volatility-the-concept** (F5).
- claude realized-range stake-dispersion response, now weakly memory-replicated (F7).

**Withdrawn / null:**
- temperature → activity: directional only, fails the floor in both memory conditions (F5).
- band_width → behavior: non-replicating (mem-on only) (F5).
- knowability → discovery: withdrawn (observability ceiling, fails floor & replication) (F8).
- knowability → behavior: clean null.
- "all-in = gpt instability": withdrawn — rare, claude-led, and board-following (F9).

---

## Scope & limits

- **Two models, one probe family, synthetic centered random-walk markets with a
  capped synthetic-flow counterparty, single learning agent, N=30/cell.** In-frame
  these are real measurements; out-of-frame they are hypotheses.
- **Cross-battery context:** matched-seed v1 (frozen board) and the independent audit
  are in `battery_comparison_v1_vs_v2.md`. gpt's above-chance discovery on v1 was real
  (not over-credit) and collapsed to chance here — an **environment-dependent
  capability**, not an artifact. F3 board-tracking is **demoted** after the audit
  correction; decorrelated-flow v3 is spec'd in `v3_decorrelated_flow_spec.md`.
- **The synthetic flow is a confound for any "market" reading.** Because the flow is
  driven by the price path, volatility effects (F5) cannot be cleanly separated from
  gpt responding to a *busier board* — entangled with flow, not established as
  volatility-the-concept. F3 board-tracking is demoted (audit correction); see
  `battery_comparison_v1_vs_v2.md`.
- **Centered geometry caps the discovery axis** (F8): knowability has effects that
  are *unmeasurable by construction* here, not necessarily absent in real or
  non-centered markets.
- **Money is out of scope.** Win-rate is a behavioral/outcome proxy only; pnl
  differences are not significant.
- **Entry-timing (`first_trade_frac`) is a frozen, wording-dependent channel** and is
  excluded from causal attribution.

## For customer-1 (transferable lessons)

1. **Models have measurable, stable trading personalities** the harness can
   fingerprint from behavior alone, and they persist when the order book comes alive.
2. **A live board may be a signal models chase — but F3 is demoted pending a clean
   test.** The v2 board-tracking rates did not isolate synthetic flow from the model's
   own prior trade; decorrelated-flow v3 (`v3_decorrelated_flow_spec.md`) is the
   experiment that would settle whether models read the tape or the price the tape
   follows.
3. **Task competence can depend on the venue** (cross-battery,
   `battery_comparison_v1_vs_v2.md`): gpt's above-chance discovery on a frozen board
   was real and collapsed to chance here — environment-dependent, not an artifact.
4. **Discovery metrics that condition on end-state are unfair across trader
   types** (F4); use end-state-independent measures when comparing models.
5. **Some axes are unmeasurable by construction in centered markets** (F8); a
   market-design pre-screen must check observability before reading a null as zero.
6. **Outcome metrics conditioned on terminal holdings confound commitment with
   accuracy** (F6): decompose `win` into `hold_rate × P(win | held)` before reading
   it — and note the F6 tension that memory may also touch peak discovery at marginal p.
7. **Background market activity confounds "the model responds to volatility."**
   Sweeping volatility moved gpt strongly here — but volatility also drove the
   synthetic flow, so the behavioral response is **entangled with board busyness**, not
   established as volatility perception (F5). Hold counterparty activity constant
   before reading a volatility effect as a model property.
