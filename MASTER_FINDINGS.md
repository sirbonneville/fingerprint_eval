# Master Findings — Official Battery v1

The consolidated read of the full two-model behavioral battery. This is the
headline document; `FINDINGS.md` holds the older single-model (gpt-4o, N=10)
record, now reconciled against this battery at its head.

**Battery:** `eval_runs/2026-06-27_official-v1/` — **52 cells, all complete.**
- **2 models:** `openai/gpt-4o`, `anthropic/claude-sonnet-4`
- **2 memory conditions:** mem-off (stateless) and mem-on (prior rationales replayed)
- **13 cells each:** baseline + 4 volatility + 3 temperature + 3 knowability + 2 band_width
- **N = 30 runs/cell** (1,560 episodes total), faithful prompt (`disclose_dead_window=True`)
- 3-band centered arithmetic random-walk market, anchor=100, 2h window, 5 decisions, budget=1000.

All reads use the project discipline: **effect-vs-spread over significance**,
**identity (sizing-independent) over magnitude**, **permutation floors** under
every candidate effect, and — new this round — **replication across the memory
condition** as a second, independent robustness test. Money (pnl) is out of
scope and reported only as context.

---

## The one-paragraph result

Two models on identical markets are **two stable, sharply different traders**, and
the difference is *trait-level*, not config-driven: gpt-4o is a cautious,
concentrated, budget-thirds staker; claude-sonnet-4 is an active, diversified,
smaller-stake repositioner. The most interesting structural finding is an
**orthogonality of sensitivity**: **gpt-4o responds to the *market* (band geometry,
temperature) but ignores its own *memory*; claude responds to its *memory*
(transformatively) but is nearly inert to the market *label* (it is weakly
responsive to the realized *path*, F6).** Almost every "market effect" in the
battery is gpt-only; almost every "memory effect" is claude-only.
The market-axis effects, ranked by strength, are **band_width > temperature >
volatility ≈ knowability (null)** — i.e. *geometry moves behavior more than
volatility does*. Two earlier candidate effects (a knowability→discovery drop, a
volatility→engagement sweep) **did not survive** the floors and the memory
replication; one earlier null (volatility) was **rescued in a weak form** by
conditioning on realized path instead of the label. The discovery channel is
**weak for both models, marginally better for claude (~+0.07 on the unbiased
metric)**, and the headline "claude discovers better" was largely a **survivorship
artifact** of a metric (`top_pick`) that silently conditions on end-state.

---

## Methods that did the work

Beyond the standing "split a double-meaning statistic before trusting it"
discipline, four instruments carried this round:

1. **The built-in noise floor (4 identical-config draws).** Baseline is
   config-identical to three sweep midpoints — `volatility=1.0`, `knowability=0`,
   `temperature=0.7` are the *same cell*. That yields **four independent 30-run
   draws of one configuration** per model×mem, whose spread is the true
   within-config noise. Every axis effect is judged against *this measured floor*,
   not an assumed one. (mem-off floor: gpt trade_count spread 0.50, volume 114,
   median-size 62; claude 0.13 / 36 / 18.)
2. **End-state-independent discovery.** `top_pick` conditions on the *closing*
   position, so it is undefined whenever a model exits flat — which claude does
   ~45% of the time. We added `peak_top_pick` (winner == top holding at the
   turn of maximum gross exposure) and `buy_flow_on_winner` (fraction of buy
   notional aimed at the winner). Both are defined for any run that traded, so the
   two models are scored on comparable denominators (~390 vs ~390, not 360 vs 215).
3. **Permutation floors** under every candidate effect (20k shuffles), including a
   same-config null sanity (baseline vs knowability=0) that must *not* fire.
4. **Replication across memory.** mem-off and mem-on are two independent passes;
   a real effect should appear in both. This deflated three of the marginal leads.

A note on conservatism: the memory test (section C) pools all 13 core cells and
shuffles the memory label **unstratified**. Because the cells genuinely differ,
that injects cross-cell variance into the null and makes the test *conservative* —
so the 8 claude metrics clearing at p<0.02 do so against a slightly inflated null
(more trustworthy than nominal, not less). A within-cell-blocked permutation would
only sharpen results that already hold. (Data-hygiene trap, guarded in the engine:
the parser zeroes `parsed["outcome"]` on hold turns, so any future metric reading
that field on non-trade turns would silently see band 0; current metrics read
`planned_order["outcome"]` / positions and are unaffected.)

Scripts (read-only, no model calls): `eval_runs/master_analysis.py` (the
aggregate tables here), `eval_runs/knob_diagnostics.py` (the six targeted checks),
`eval_runs/within_label_check.py` (path-magnitude binning).

---

## F1 — Two stable trader personalities (pooled over all 13 core cells)

| metric | gpt-4o off | gpt-4o on | claude off | claude on |
|---|---|---|---|---|
| trade_count | 1.49 | 1.58 | **3.01** | **3.15** |
| hold_rate | **0.70** | **0.68** | 0.40 | 0.37 |
| total_volume ($) | 511 | 534 | **771** | **864** |
| median_trade_size ($) | **343** (≈budget/3) | 348 | 246 | 267 |
| direction_switches | 0.32 | 0.33 | **1.09** | 0.91 |
| outcomes_traded | 1.14 | 1.12 | 1.18 | **1.33** |
| exit_flat_rate | 0.08 | 0.14 | **0.45** | 0.18 |

**gpt-4o** = cautious / concentrated / decisive: ~1.5 trades, holds 70% of
decisions, one bucket, almost never reverses, rarely exits flat.
**claude** = active / diversified / churning: ~3 trades, holds 40%, more buckets,
reverses direction ~1×/run, deploys ~50% more capital.

These are **trait-level constants**, not draws: across all 26 cells (both memory
conditions), claude's median trade_count is **3.0 in nearly every cell** (a couple
at 3.5) and gpt's median stake sits in a **$6 band, $326.73–$333.33 (≈budget/3),
in every cell**. The cross-cell constancy *is* the evidence — that level of
repetition under varied configs cannot be a coincidence of sampling.

## F2 — The sizing fingerprints (verified)

- **gpt-4o anchors to budget/3.** **63%** of all gpt trades are *exactly* $333.33
  (one-third of the 1000 budget); claude hits that value only **11%** of the time.
  gpt treats "bet a third" as a default action; claude sizes off round numbers
  (100/200/250/300) it picks per-situation.
- **Per-trade dispersion is similar** (size CV ≈ 0.46 both) — claude is not
  *more precise* per trade. But **claude's total run deployment is more
  consistent** (volume CV **0.33** vs gpt **0.56**), because it reliably makes ~3
  trades while gpt's count swings 1–4 and it occasionally dumps a large stake
  (max single trade $1000). gpt's caution is, paradoxically, the more *erratic*
  total footprint run-to-run.

## F3 — Orthogonal sensitivity: gpt is market-driven, claude is memory-driven

The master cross-reference. Tally which interventions actually move each model
(permutation-floored, requiring correlated metrics and/or replication):

| | responds to **market** axes | responds to **memory** |
|---|---|---|
| **gpt-4o** | **yes** — band_width (4 metrics, both mems), temperature (3 metrics, mem-off) | **no** — only exit_flat (+0.06) and entry-timing (+0.02) move |
| **claude** | **nearly no** — flat on band_width/temp/knowability; only scattered, non-replicating volatility hits | **yes** — 8 metrics shift, all p<0.02 |

The two models have **complementary, almost non-overlapping sensitivity
profiles.** gpt rewrites its behavior when you change the *board* and shrugs at
its own replayed reasoning; claude shrugs at the board and rewrites its behavior
when you give it *memory*. This is the kind of qualitative policy difference the
instrument was built to expose, and it is clean.

## F4 — Memory is a model-specific amplifier (claude: churner → accumulator)

Memory (replaying the model's own prior rationales) is **inert for gpt-4o** but
**transformative for claude** (mem-on − mem-off, pooled, permutation p):

| metric | gpt-4o gap [p] | claude gap [p] |
|---|---|---|
| trade_count | +0.09 [0.116] | **+0.14 [0.017]** |
| hold_rate | −0.02 [0.101] | **−0.03 [0.007]** |
| total_volume | +23 [0.257] | **+94 [0.000]** |
| median_trade_size | +4 [0.515] | **+21 [0.000]** |
| direction_switches | +0.01 [0.812] | **−0.18 [0.000]** |
| outcomes_traded | −0.02 [0.639] | **+0.15 [0.000]** |
| exit_flat_rate | **+0.06 [0.010]** | **−0.27 [0.000]** |
| win (held winning band) | +0.02 [0.572] | **+0.16 [0.000]** |

**Memory converts claude from a churner into an accumulator.** With its prior
rationales in view, claude **reverses direction less** (−0.18), **stops bailing
to flat** (exit-flat 0.45→0.18, a 27-pt drop), **deploys more capital** (+$94) in
**bigger stakes** (+$21) across **more buckets** (+0.15). gpt, by contrast, does
not act on its own replayed rationale at all (its only memory response is exiting
flat slightly more). Memory's one clear effect on gpt is the opposite sign on the
one metric they share — a nice illustration that the *same intervention* lands
differently on different policies.

**The win-rate gain is commitment, not accuracy (decomposed, not interpreted).**
`win` ("held the winning band at the bell") is end-state-conditioned in exactly
the way `top_pick` was (F8), so it must be split into its two factors,
`win = held_rate × P(win | held)`:

| claude | mem-off | mem-on |
|---|---|---|
| held_rate `P(held)` | 0.55 | **0.82** |
| `win` `P(win)` | 0.40 | 0.56 |
| **conditional accuracy `P(win\|held)`** | **0.73** | **0.68** |

The entire 16-point win gain is the **holding rate** rising (0.55→0.82);
**conditional accuracy is flat — slightly *down*** (0.73→0.68). Memory does not
make claude *pick* better; it makes claude *commit* — convert an existing,
unchanged, modest edge into terminal positions instead of round-tripping to flat.
In a market where claude is right ~70% of the time *when it holds anything*,
commitment mechanically yields more wins. This is **required** by F8 (memory
leaves discovery flat: peak_top_pick +0.073 mem-on ≈ +0.075 mem-off); the win
gain and the flat discovery only reconcile if the win is pure commitment and zero
selection — which is exactly what the decomposition shows. (Separately, claude's
conditional accuracy ~0.70 vs gpt's ~0.57 is a third, independent corroboration
of the small F8 discovery edge.) *Money (pnl) is out of scope and not significant.*

## F5 — The market-axis hierarchy (what actually moves behavior)

Ranked by floored, replicated strength:

1. **band_width — strongest and most robust.** Wider bands (5→20) move gpt-4o on
   **four correlated metrics in *both* memory conditions**: trade_count
   1.23→1.80 (mem-off, p=0.012; mem-on 1.47→2.13, p=0.015), hold_rate down,
   median stake down, outcomes_traded up. Wider bands → gpt trades more, holds
   less, sizes smaller, spreads across more buckets. This is the **only market
   axis that replicates across memory**, making it the most trustworthy
   behavioral effect in the battery. **claude is flat on band_width.**
2. **temperature — real but memory-fragile (gpt only).** gpt activity rises with
   temperature mem-off (trade_count 1.20→1.80 p=0.006, hold_rate p=0.005,
   volume p=0.041) but **attenuates under memory** (mem-on same direction,
   p=0.13, not clearing). Notably this **moves the activity *center*** (trade
   frequency), refining the older "temperature is a pure spread knob" read — under
   the faithful prompt it shifts how often gpt trades, while the *sizing* center
   stays pinned (~$340). claude flat both mems.
3. **volatility (by label) — null.** Neither model's footprint tracks the
   volatility *label*; the only sub-0.05 hits are isolated single metrics that do
   not replicate across memory (noise under multiple comparisons). See F6 for the
   rescued within-label version.
4. **knowability — total behavioral null.** Flat on every metric, both models,
   both memory conditions. (Its *discovery* story is F7, and it was withdrawn.)

The ordering is itself a finding: **market geometry (band_width) moves gpt more
than market volatility does.** A naive expectation that "more volatile market →
more trading" is simply false here; "wider decision bands → more trading" is the
real lever, and only for gpt.

## F6 — Volatility: label-null, weak realized-range response (claude, mem-off only)

The lossy-proxy lesson, confirmed on live data. Sweeping the volatility *label*
shows nothing, but the label is a noisy proxy for the *realized* price range a run
actually drew. Binning a single label's 30 episodes by realized range (calm vs
wild) and permutation-testing the size-dispersion gap:

- **claude, mem-off:** all four labels show a **positive** gap (wild → more
  dispersed sizing); one clears (vol=1.0, p=0.018). 4/4 same direction
  (sign-test p≈0.06) + one significant = a **weak-but-consistent realized-range
  response.**
- **claude, mem-on:** does **not** replicate (3/4 positive, none clear).
- **gpt, mem-off:** nothing clears, no consistent direction.

So the honest claim is narrow: **claude (stateless) modulates stake dispersion
with the magnitude it actually sees, a sensitivity the label sweep hides; it does
not survive into the memory condition, and gpt shows it nowhere.** The
*methodological* point is the durable one — **sweeping a design parameter
understates sensitivity that only appears when you condition on the realized
path** — and it now has live cross-model support (gpt ran the identical test and
got nothing, so it is not a test artifact).

## F7 — Knowability: behavioral null + the withdrawn discovery effect (observability ceiling)

An apparent knowability→discovery drop (top_pick falling at the 120-min dead
window) was the round's most tempting "first market effect." It was **withdrawn**
after three checks:

- **Mechanism (the decisive one).** The price path runs *through* the dead window,
  so the settlement bucket is set by *unseen* post-close diffusion. A
  model-independent oracle that reads the last observed price scores **100% at
  knowability=0 but only 60% at knowability≥60** — the outcome literally leaves
  the observable window. A top_pick drop is therefore **the target becoming
  unknowable by construction, not the model's discovery skill degrading.** And the
  ceiling collapses at 0→60 while the model dip is at 60→120, so the mechanical
  story doesn't even align with the dip.
- **Floor.** The gpt headline 0→120 drop is p=0.08 (does not clear); claude clears
  nothing (p≈0.15).
- **Replication.** Across memory the effect **flips which model shows it**
  (mem-off marginal in gpt; mem-on it is claude that clears, gpt flat). An effect
  that swaps carriers between conditions is noise.

This is the **second time the centered-market geometry has eaten a candidate
finding through a *different* mechanism** (first: no-signal-to-discover in a
martingale; now: target-leaves-the-window). The reusable lesson: in this market
design several "nulls" are **unmeasurable-by-construction**, a stronger and more
interesting claim than "measured and found zero." Behaviorally, knowability is
also a flat null.

## F8 — Discovery: weak for both, claude marginally better, top_pick is biased

Scoring both models on **comparable, end-state-independent** denominators
(chance = 0.33):

| metric | gpt-4o | claude | gap [p] |
|---|---|---|---|
| `top_pick` (end-state, **biased**) mem-off | 0.53 (n=360) | 0.73 (n=215) | +0.198 [0.000] |
| `peak_top_pick` mem-off | 0.47 (n=388) | 0.55 (n=390) | **+0.075 [0.044]** |
| `buy_flow_on_winner` mem-off | 0.46 | 0.52 | +0.064 [0.055] |
| `peak_top_pick` mem-on | 0.52 (n=381) | 0.60 (n=390) | **+0.073 [0.050]** |

The cited "claude discovers better, 0.82 vs 0.46" was **largely a survivorship
artifact**: `top_pick` only scores runs that held a position to the bell, and
claude's *holding to the bell is itself correlated with being right* (it exits
flat unless confident). End-conditioning inflates claude by +0.18 but gpt by only
+0.06. On metrics that don't condition on end-state, the gap **collapses to a
small ~+0.07.**

**What carries this edge is NOT the p-values — say so plainly.** The
peak (p=0.044 / 0.050) and buy_flow (p=0.055) hits are single-metric, sit on the
0.05 threshold, and this battery runs >100 permutation tests; none would survive a
multiple-comparison correction on its own. The ~+0.07 edge is trustworthy for two
*other* reasons: **(a) replication across the memory split** — +0.075 (mem-off)
and +0.073 (mem-on) are two independent passes landing on the same value — and
**(b) triangulation across structurally different metrics** — peak_top_pick
(position-state at peak exposure), buy_flow (capital flow, no position-state at
all), and the F4 conditional accuracy P(win|held) (~0.70 claude vs ~0.57 gpt) all
agree. Three different instruments + two independent passes is the evidence; the
lone threshold p-values are the weakest part of it.

Verdict: both models are genuinely above chance (real, if weak, discovery),
**claude is marginally better (~+0.07)**, and `top_pick` is an unfair instrument
for comparing trader types whose holding behavior differs.

## F9 — Replication-across-memory as a robustness axis

Running both memory passes turned the memory condition into a free second test.
What it did to each candidate:

| candidate | mem-off | mem-on | verdict |
|---|---|---|---|
| two personalities | strong | strong | **holds** |
| budget/3 anchor, claude churn | strong | strong | **holds** |
| band_width → gpt engagement | clears | clears | **holds (robust)** |
| temperature → gpt activity | clears | attenuated | real but memory-fragile |
| claude discovery edge (peak) | +0.075 | +0.073 | **holds (small)** |
| volatility realized-range (claude) | weak-positive | fails | fragile |
| knowability → discovery | gpt-marginal | claude-marginal (flips) | **withdrawn** |
| memory → claude transformation | — | — | **the new headline** |

The discipline paid off exactly as intended: the load-bearing results replicated,
and every *marginal* lead either firmed up or deflated under the second pass.

---

## Synthesis (hypothesis, not yet a finding): memory bites in proportion to self-inconsistency

F1, F3, and F4 are three descriptions of one likely mechanism: **memory affects a
model in proportion to how much it would otherwise contradict its own past
actions.**

- **gpt-4o one-shots and holds** — it is already self-consistent turn to turn, so
  replaying its rationale gives memory *nothing to act on*. Inert (F4), exactly as
  observed.
- **claude churns and round-trips** — it is self-*inconsistent* turn to turn, so
  its own prior reasoning is a large surface for memory to grab, and the effect is
  to **suppress the contradiction**: direction_switches −0.18, exit_flat −0.27 →
  commitment (F4).

This upgrades "memory is a model-specific amplifier" (a description) to a
mechanism (a *why*), and it makes a **falsifiable prediction in data we can
generate**: since temperature makes gpt churnier (F5), a **memory × temperature
interaction** should make memory *start to bite gpt at high temperature*. If it
does, the mechanism holds; if gpt stays memory-inert even when churning, the cause
is something more intrinsic and the mechanism is wrong. Either outcome is a sharper
customer-1 lesson than "amplifier." **Flagged as a hypothesis; one clean cell-cross
away from a test, not asserted here.**

---

## Scorecard

**Confirmed (robust):**
- Two stable, distinct trader personalities; trait-level, config-invariant (F1).
- gpt budget/3 sizing anchor (63%); claude free sizing + steadier total deployment (F2).
- Orthogonal sensitivity: gpt market-driven, claude memory-driven (F3).
- Memory transforms claude (churner→accumulator); inert for gpt (F4). The win-rate
  gain is **commitment-mechanical** (held-rate rises, conditional accuracy flat),
  not improved accuracy.
- band_width → gpt engagement, replicated across memory (F5).
- Instrument validation: claude's rich dynamics prove gpt's flatness is policy.

**Confirmed (weak / scoped):**
- temperature → gpt activity center, mem-off clean, mem-on attenuated (F5).
- claude realized-range sizing-dispersion response, mem-off only (F6).
- claude marginal discovery edge ~+0.07 on unbiased metrics (F8).

**Withdrawn / null:**
- knowability → discovery drop: withdrawn (observability ceiling, fails floor & replication) (F7).
- knowability → behavior: clean null.
- volatility → behavior by label: null (rescued only as F6).
- "claude discovers better, everywhere": downgraded; mostly top_pick survivorship (F8).

---

## Scope & limits

- **Two models, one probe family, synthetic centered random-walk markets, single
  agent, N=30/cell.** In-frame these are real measurements; out-of-frame they are
  hypotheses.
- **Centered geometry caps the discovery axis** (F7): knowability and (in the
  martingale) drift have effects that are *unmeasurable by construction* here, not
  necessarily absent in real or non-centered markets.
- **Money is out of scope.** Win-rate is reported as a behavioral/outcome proxy
  only; pnl differences are not significant.
- **Entry-timing (`first_trade_frac`) is a frozen, wording-dependent channel**
  (`FINDINGS.md` #2) and is excluded from causal attribution.

## For customer-1 (transferable lessons)

1. **Models have measurable, stable trading personalities** the harness can
   fingerprint from behavior alone — and they differ on *what they respond to*
   (market vs memory), not just how much they trade.
2. **Sweeping a design parameter understates sensitivity** that conditioning on
   the realized path reveals (F6). Pre-screening a market design on label sweeps
   alone will miss path-dependent behavior.
3. **Discovery metrics that condition on end-state are unfair across trader
   types** (F8); use end-state-independent measures when comparing models.
4. **Some axes are unmeasurable by construction in centered markets** (F7); a
   market-design pre-screen must check observability before reading a null as zero.
5. **Outcome metrics conditioned on terminal holdings confound commitment with
   accuracy** (F4): decompose `win` into `hold_rate × P(win | held)` before reading
   it. A model that merely *commits* more (holds positions to settlement) will post
   more wins without picking any better — the win gain can be entirely mechanical.
   This is the same end-state-conditioning trap as the discovery-metric lesson (#3),
   and it bites outcome metrics just as hard.
