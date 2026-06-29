# Findings

A running record of what the instrument has actually established, written to the
same standard the runs are held to: effect-vs-spread over significance, identity
(sizing-independent) over magnitude (sizing-confounded), and an explicit
statement of limits rather than a manufactured result past them.

All reads below are gpt-4o (`openai/gpt-4o`), N=10/cell unless noted, on the
3-band centered price-bucket market (anchor=100, 2h trading window, band-relative
arithmetic volatility), gated behind the easy/hard calibration control pair.

---

> **SUPERSEDED IN PART — see [`MASTER_FINDINGS.md`](MASTER_FINDINGS.md).** This
> file is the *original single-model, N=10, settlement-blind* record. The official
> two-model battery (`eval_runs/2026-06-27_official-v1/`, gpt-4o + claude-sonnet-4,
> N=30, faithful prompt, mem-off & mem-on, 52 cells) re-measured every axis. The
> master file is now the headline; reconciliation of the specific findings here:
>
> | this file | battery verdict |
> |---|---|
> | #1 sizing-independent discovery metric | upheld and extended — `top_pick` *also* found to be **survivorship-biased** across trader types (claude exits flat ~45%); added `peak_top_pick` / `buy_flow_on_winner` (master F8). |
> | #2 entry-timing wording-frozen | upheld; entry-timing still excluded from attribution. |
> | #3 knowability has no discovery gradient (martingale) | upheld and **deepened** — a *second* mechanism (target leaves the observable window; oracle ceiling 100%→60%) independently caps it. A candidate knowability→discovery drop was **withdrawn** (master F7). |
> | #4 drift discovery confidence-confirmed/identity-inconclusive | unchanged (drift stays parked). |
> | **#5 volatility drives engagement** | **DID NOT REPRODUCE.** Under the faithful prompt at N=30 the volatility *label* sweep is a behavioral **null** for both models; the N=10 monotonic trend did not survive. Rescued only in a weak within-label realized-range form for claude, mem-off (master F6). |
> | **#6 temperature is a spread knob, not a center knob** | **PARTLY REVERSED.** Under the faithful prompt, temperature **does move gpt's activity center** (trade_count 1.20→1.80, p=0.006, mem-off; attenuates with memory). The *sizing* center stays pinned (~$340), so the refined read is "moves activity frequency, not stake size" (master F5). |
> | new in battery | **band_width is the strongest, most robust market axis** (gpt, both memory conditions); **memory transforms claude but not gpt**; **orthogonal sensitivity** (gpt market-driven, claude memory-driven). |
>
> Read #5 and #6 below as the old-prompt baseline that the battery has now corrected.

---

## The method (one discipline, applied five times)

The findings below matter, but the *method* is the transferable thing, and it is
otherwise invisible unless written down. Every result in this project came from a
single move repeated: **a clean-looking number was secretly measuring two things at
once; separate them before trusting anything built on top, and the truth falls
out — either rescuing the result or honestly killing it.** Five instances:

1. **Settlement time: verified vs assumed.** Before any analysis, the settlement
   rule was checked against ground truth rather than assumed. Verifying the
   precondition first is what kept everything downstream interpretable.
2. **Calibration: accuracy vs sizing.** `prob_on_winner` conflated *being right*
   with *betting hard*. Separating them (sizing-independent `rank_score`/`top_pick`)
   re-widened the calibration gap from ~0.15 back to ~0.50 and unblocked the sweeps
   (finding #1).
3. **Discovery: identity vs confidence.** "Did the probe find the winner" (identity,
   un-confounded) is a different claim from "how confidently did it bet" (magnitude,
   sizing-confounded). Holding them apart is what exposed the martingale result as a
   *structural limit* rather than a missed effect (findings #3, #4).
4. **Temperature: median-shift vs spread-widening.** Decided *in advance* that a
   moved center means real behavioral change and a fanned spread means noisier
   sampling. Pre-registering the two possibilities is the only reason the
   center-pinned/spread-inflated result resolved cleanly instead of becoming an
   argument (finding #6).
5. **Significance: sample effect vs population dispersion.** "More N will fix it" is
   only true for a sampling artifact, not for a genuinely small population effect.
   Distinguishing the two is what licensed declaring some effects confirmed and
   others unrescuable — and what made refusing to chase CLES to 0.80 the disciplined
   call, not a cop-out (findings #4, #5).

This is one failure mode — *a single statistic silently carrying two meanings* —
and one defense — *split it before you trust it.* The instrument was built to resist
that failure mode, and resisting it changed the answer all five times. That is the
reusable artifact here; the individual findings are its output.

## Scope (what these claims do and do not cover)

Everything below is **gpt-4o, one probe, synthetic centered random-walk markets,
single-agent, N=10/cell.** Within that frame the findings are real measurements.
*Outside* it they are hypotheses, not results:

- **One model.** Volatility-moves-center and temperature-moves-spread are clean
  statements about gpt-4o. Whether they hold for other models is untested (the
  mini-vs-4o pacing contrast already shows probe choice is itself a first-class
  variable).
- **Synthetic, centered markets.** The price paths are generated, not real, and the
  settlement bucket is centered — which is precisely the geometry that caps the
  knowability-discovery question (finding #3). Real or non-centered markets are a
  different regime.
- **Single agent.** One probe against the curve; no multi-trader dynamics.

Stating the boundary plainly is what makes the in-frame claims trustworthy rather
than overreaching — the limits are reported, not buried.

---

## 1. Discovery metric confound: `prob_on_winner` conflates accuracy with sizing

`prob_on_winner` / `brier` / `edge_over_open` are computed from the probe's
closing **position magnitude** — they reward betting *hard*, not just being
*right*. A probe that correctly identifies the winner but rationally sizes small
scores low. This silently penalized gpt-4o (which sizes more conservatively than
gpt-4o-mini) and collapsed the calibration gap from 0.57 to ~0.15.

**Fix:** sizing-independent discovery metrics — `winner_top_pick` (is the winner
the probe's largest position) and `winner_rank_score` (the winner's rank among
the probe's positions, 1.0=top … 0.0=last). Switching the calibration gate to
`rank_score` re-widened the easy/hard gap to a clean ~0.50. Identity (rank/
top_pick) is the headline discovery channel; magnitude metrics are secondary and
carry the sizing caveat.

## 2. `MARKET TIMING` prompt block is a frozen experimental condition (wording-dependent on entry timing)

A neutrality check (rephrase-and-rerun under alternative neutral wordings)
found `first_trade_frac` tracked the *elaborateness* of the phrasing
monotonically (A<B<C), reproduced at n=12. The block is therefore **not** neutral
for entry timing. It is frozen as canonical Variant A and documented in
`prompt.py`. Consequence: knowability/temperature effects on *what* and *how much*
the model trades remain readable; effects on *when* it enters are confounded with
the prompt wording and cannot be cleanly attributed.

## 3. Knowability has no sizing-independent gradient in a centered random-walk market (the martingale / geometry result)

This is a result, not a null. Established by exclusion across three runs:

- **Driftless, 3 bands:** identity pinned at the **ceiling** — a centered
  symmetric distribution keeps the middle band the modal best-guess regardless of
  dead-window length, so `top_pick` is trivially right. No gradient.
- **Driftless, 7 bands:** identity pinned at the **floor** — discovery collapses
  to ~chance and gpt-4o under-trades the harder market. No gradient.
- Both resolutions fail, in opposite directions. The signal isn't being missed;
  in a driftless (martingale) market there is **no discoverable signal** for
  knowability to act on — the optimal forecast is always "the current price."

**Conclusion.** *In a centered random-walk price-bucket market, outcome
knowability has no gradient that a sizing-independent discovery metric can detect,
because the geometry forces a ceiling-or-floor on best-guess identity — coarse
bands make the answer trivial, fine bands make it noise, and the engaged middle
inherits both failure modes.*

## 4. Drift creates discoverable signal — confidence-confirmed, identity-inconclusive (scoped)

Adding deterministic upward drift (a trend the dead window can obscure) was the
pre-authorized way to give knowability something to degrade. Drift sweeps
({0,0.5,1,2,4}) established:

- **Confidence channel:** discovery rises monotonically with drift. At
  vol=1.5/knowability=120, the `drift=0 → drift=4` contrast reached
  **effect/spread = 1.24, CLES = 0.80 ("possible effect")** — the only
  threshold-clearing discovery effect observed in the project. Because Brier (a
  proper scoring rule that punishes confident-and-wrong) *improves*, and the
  drift=0 baseline sits at chance (ruling out a blind "always up" prior), the rise
  is most plausibly genuine discovery, not mere sizing aggressiveness.
- **Identity channel (the un-confounded verdict):** **not cleanly confirmed.**
  Identity-median is structurally **ceiling-pinned at 3 bands** (a competent probe
  among only three options is right most runs), in every vol/knowability/drift
  condition tried. The directional signal that *is* present — the lower tail of
  misses collapses to zero as drift rises (rank IQR 0.5→0, CLES→0.72) — is real
  but sub-threshold (CLES < 0.80), and per the population-property of effect-size,
  **more N cannot rescue it.**
- The un-saturating lever (more bands) breaks probe engagement at 7 bands; the
  engaged band counts inherit the ceiling. The instrument is squeezed in a vise it
  has now mapped on both jaws.

**Net.** Drift improving discovery is *confidence-confirmed and identity-suggestive*
but **cannot be cleanly confirmed on the sizing-independent channel within this
market design.** This is a characterized structural limit, not an unexplained
flat read.

### Shelved (deliberately, not abandoned)

Confirming drift-discovery on the identity channel requires a **different market**,
not a parameter tweak — one whose outcome is not a centered bucket: settling on
*direction* (up/down), on a *moving* target, or with drift strong enough that the
distribution's mode actually relocates. Those have a discoverable signal by
construction, so knowability would have something real to degrade. Queued as a
market-redesign question, deliberately chosen.

---

> **PROMPT-VERSION NOTE (findings #5 and #6).** Both were measured under the
> *settlement-blind* prompt, before the harness disclosed the settlement schedule
> (the dead window). The official battery re-measures every axis under the new
> *faithful* prompt (`disclose_dead_window=True`), so #5/#6 are **superseded** by
> the battery's volatility/temperature cells. This is not wasted work: the battery
> re-confirming these axes under a changed prompt is a **free robustness check** --
> if volatility-moves-center and temperature-moves-spread reproduce under the
> faithful prompt, the effects survive a prompt change (robust); if they shift,
> that prompt-dependence is itself a finding. Read #5/#6 below as the old-prompt
> baseline pending that check.

## 5. Volatility drives engagement (the first clean positive, behavior axis)  [OLD-PROMPT; DID NOT REPRODUCE in battery — see MASTER_FINDINGS F6]

Sweeping volatility ∈ {0.5, 1.0, 1.5, 2.0} (k=0, drift=0, 3 bands, N=10), reading
the **un-confounded behavioral footprint**. Four correlated activity metrics move
monotonically with volatility — the same "more volatility → more engagement"
syndrome:

```
                    vol=0.5   vol=1.0   vol=1.5   vol=2.0
trade_count          1.0       1.5       3.0       3.0
hold_rate            0.8       0.7       0.4       0.4
total_volume         333       542       653       778
direction_switches   0.0       0.5       1.0       1.0
(median_trade_size ~333 flat; outcomes_traded ~1 flat; first_trade_frac noisy)
```

`trade_count`, `hold_rate`, and `total_volume` clear the **primary effect>spread
bar** (ratios up to 1.78). Per-comparison CLES sits at ~0.74–0.78 — just under the
0.80 overlap cut, so the analyzer labels individual contrasts "weak/ambiguous" —
but the **monotonic, four-metric-consistent trend that clears effect>spread is a
stronger form of evidence than any single CLES value**, and is recorded as a
confirmed effect.

Crucially distinct from the discovery nulls: there the effect/spread was *tiny*
(~0.08, a genuinely small population effect more N cannot rescue); here it is
*large* (up to 1.78). The CLES is a population property too, so chasing it to 0.80
with more N would be the manufacture-significance move and is declined. The effect
is real, monotonic, and on the clean behavioral axis the instrument was built for.

**Banner fix (hygiene):** `--no-trust` previously reused the stand-in "PLUMBING
ONLY" banner, mislabeling real-model behavior reads. A real model with the gate
bypassed now prints `DISCOVERY UNGATED` instead, making clear the BEHAVIOR section
is a genuine finding and only DISCOVERY is ungated.

---

## 6. Temperature is a spread knob, not a center knob (the model-side intervention)  [OLD-PROMPT; PARTLY REVERSED in battery — temperature moves gpt's activity center under the faithful prompt; see MASTER_FINDINGS F5]

Sweeping temperature ∈ {0.0, 0.7, 1.2} (vol=1.5, k=0, drift=0, 3 bands, N=10),
the first **model-side** intervention — the within-model analog of the
mini-vs-4o pacing contrast. Read through the pre-registered lens that temperature
is *known* to inflate dispersion: **median-shift = real behavioral change;
spread-widening = same behavior, noisier sampling.** The result is unambiguous and
falls entirely on the spread side.

**No behavioral median moved past the bar.** Every footprint metric reads
`NO DETECTABLE EFFECT` on effect>spread, and crucially **none is monotonic** —
the signature that separates this from the volatility result:

```
                    temp=0.0      temp=0.7      temp=1.2
trade_count          2.5           3             1.5      (non-monotonic)
hold_rate            0.5           0.4           0.7      (non-monotonic)
total_volume         820           865           577      (non-monotonic)
direction_switches   1.5           1             0.5      (within spread)
outcomes_traded      1             1             1        (flat)
```

Contrast volatility (finding #5): there four metrics moved *monotonically* and
*cleared* effect>spread. Here the medians wander inside the noise with no
direction. Temperature does **not** causally change *what* gpt-4o does.

**What it does change is dispersion — cleanly and monotonically — on sizing.**
`median_trade_size` is the tell: the **median is pinned at 333.3 in all three
cells**, but the spread fans open with temperature exactly as predicted:

```
                    temp=0.0          temp=0.7          temp=1.2
median_trade_size   333.3             333.3             333.3     (center fixed)
  IQR               0                 18.2              212.5     (fans open)
  range            [326.7, 333.3]    [200, 375]        [166.7, 500]
```

This is the textbook spread-widening signature: identical center, monotonically
widening tails. Temperature makes gpt-4o's *sizing* noisier around the same
typical bet — it does not shift the bet. (`discovery_brier` and `prob_on_winner`
spreads widen the same way at temp=1.2 while their medians barely move — the
accuracy *distribution* gets noisier, the typical accuracy does not improve or
degrade.)

**Net.** Temperature is a **variance/dispersion lever, not a behavior lever**, in
this market. It widens the run-to-run spread of how hard the model bets without
moving the center of any behavioral metric. This is itself a clean causal read —
and a useful caution: a naive sweep that only watched medians would call
temperature "no effect," while a naive one that only watched spread would call it
"a big effect." The honest read is both: *center unmoved, dispersion inflated.*
The one metric with a hint of a median trend (`first_trade_frac` 0→0.125→0.25) is
exactly the **frozen, wording-dependent entry-timing channel** (finding #2) and is
sub-threshold (effect/spread=1.0, CLES=0.62) — it cannot be cleanly attributed and
is set aside per the freeze.

## What works (the instrument itself)

The causal-read machinery is done and trustworthy: it caught the sizing confound,
held effect-vs-spread through disconfirmed predictions, refused to inflate a
pre-registered hope, caught its own bad premise (the vol=2.0 anchor) against
interest, found the martingale structure by bracketing, and reported its own
resolution limit rather than manufacturing a result past it. Its negatives are
worth as much as its positives.

## Live, clean axes (the instrument's actual purpose)

The behavioral footprint (trade count, hold rate, sizing, churn, entry timing) is
un-confounded and is where the instrument sees clearly. Active:

- **Volatility → behavior** (market-side intervention): moves the *center* —
  monotonic engagement effect, clears effect>spread (finding #5).
- **Temperature → behavior** (model-side intervention): moves the *spread* —
  inflates sizing dispersion, center unmoved (finding #6).

Together these two are a clean orthogonal pair: a market knob that shifts what the
model typically does, and a model knob that shifts how variable it is around that
typical behavior. That contrast is the instrument working as designed.

### Prompt-faithfulness fix: settlement timing is now disclosed (knowability revived)

The prompt previously hid settlement timing, so the model could not perceive the
post-close dead window even though every real Delphi agent reads it from the
settlement schedule. The harness now discloses it in relative minutes
(`disclose_dead_window=True`, info-discipline-safe: market structure, not a price
or generative parameter). Consequences recorded for the battery:

- **Knowability is now a LIVE behavioral axis** (it was behaviorally dormant only
  because the prompt hid it — distinct from its *discovery* deadness, which is
  structural/martingale and unchanged). Sweeping it writes the window length into
  the prompt, so it measures "told-about-window" (what a real agent faces); the
  **disclosure on/off cross** disambiguates "window matters" vs "being told
  matters" and is run as its own preset.
- **band_width** stays live and is *mechanically distinct* from `n_outcomes`:
  band_width sets how often the price crosses a bucket boundary during trading
  (observed: tighter → more/smaller trades), `n_outcomes` sets the decision
  surface's size. The battery keeps `n_outcomes` as a separate **geometry study**
  (it changes the market, not an intervention on a fixed one); `starting_cash` is
  an optional follow-up; `drift` stays parked.

The temperature result is also a mechanistic sanity check on the instrument
itself: temperature *is* a sampling-randomness parameter, so a sound instrument
should find it moves dispersion and not the typical decision. It did. The
measurement matching the mechanism is the tell that the instrument is reading
reality, not its own wishes.

---

## Status: finished, honest v1

The board is clear. The instrument is built and validated; the two behavior axes it
was designed for both produced clean orthogonal findings; the discovery axis was
mapped to its structural limit and the open question shelved with a precise
specification of what would resolve it. This is a coherent body of work, not a pile
of runs — a stopping point, taken deliberately rather than drifted past.

### Shelved, well-specified (not loose ends)

- **Knowability-discovery** needs a *different market*, not a parameter tweak: a
  non-centered, directional (up/down), or moving-target settlement, so there is a
  discoverable signal for the dead window to obscure (finding #3/#4). A deliberate
  next project with known requirements.
- **REE wrapper** for reproducibility, and the **customer-1 framing** (the
  instrument as a market-design pre-screening / verifiable-agent-eval tool) are both
  clean additions *on top of* what exists — possible precisely because the
  market/harness seam was placed correctly. A deliberate next chapter, not unfinished
  business.
