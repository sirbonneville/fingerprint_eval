# v3 spec — the decorrelated-flow battery

*A short design spec for the third battery. v3 exists to break the one confound that
v1↔v2 could not: synthetic flow in v2 pushes the board toward the band the price is
already in, so "reads the board" and "reads the price line" are indistinguishable. v3
pushes the board toward a band the price is **not** in. One experiment resolves three
currently-confounded results at once.*

---

## Why this is the highest-value next run

Three v2 conclusions all lean on the same unresolved channel — board movement that is
correlated with the readable price (drift-band == price-band 54–85% of the time):

1. **F3 board-tracking** — is it genuine board-*reading*, or just price-following? (And
   it is currently *unestablished for claude*; see the F3 audit correction.)
2. **Deliverable B mechanism** — gpt's discovery dropped from above-chance (v1) to chance
   (v2). Is that *board-chasing* specifically, or just the live board adding noise?
3. **v2's synthesis hypothesis** — "the board pulls the model toward the move." Untested.

If synthetic flow is **decorrelated from price**, the model's target reveals which line
it follows. All three questions key off the same decisive measurement.

## The two fixes v3 makes

### Fix 1 — Decorrelate the synthetic target from the price band

In `synthetic.py`, `SyntheticFlow._inferred_band` currently returns *which band the
price is in now* (momentum-nudged). v3 adds a flow **mode** that, on a controlled
fraction of intervals, targets a band the price is **not** in:

- New `NoiseConfig` knob `decorrelation_fraction` (proposed default **0.5**): with this
  probability an interval's target is drawn **uniformly from the non-price bands**
  (reuse the existing `_other_band` helper, seeded by the noise RNG); otherwise it
  behaves exactly as v2. This yields a clean within-episode contrast (price-aligned vs
  price-divergent pushes) and keeps the board plausibly market-like.
- **Information discipline is unchanged and load-bearing:** the decorrelated target is a
  function of the *price band + noise RNG only* — it **never reads the winning band or
  settlement price**. It pushes *away from price*, not *toward the answer*. (Pushing
  toward the winner would make the flow an oracle and is forbidden.)
- All three calibration caps (`trade_max`, `per_interval_cap`, `episode_budget`) and the
  per-episode contrarian draw stay as calibrated in v2, so depth/realism are held fixed;
  only the *target band* changes.

### Fix 2 — Log the post-model-trade board (instrumentation)

The v2 records have **no `probabilities_after`**, which is why the v2 "synth-drift"
silently became "total board move since the model's last decision" (own trade + synth)
and why the clean subset collapsed to n≈0 for claude. v3 must log, per turn, the board
**after the model's own trade** (e.g. `trade_result.probabilities_after`, or an explicit
`board_after_model` field on the turn). Then for every turn:

```
synth_drift_band_t = argmax( board_model_sees_at_t  −  board_after_model_trade_{t-1} )
```

is pure synthetic flow **on every turn**, with no need to condition on "model held last
turn." This alone restores the statistical power the clean test needs for *both* models
— it is worth doing even independent of Fix 1.

## Arms and matching

- **Reuse the v1/v2 per-episode seeds**, so the price path, settlement price, and winning
  band are byte-identical to both prior batteries. v3 becomes a third matched arm:
  frozen (v1) → price-correlated flow (v2) → price-decorrelated flow (v3).
- Same 13 core cells × 30 episodes, both models (gpt-4o, claude-sonnet-4), mem on/off —
  identical to v1/v2 so every contrast is paired on `(cell, seed)`.
- Prompt template stays **byte-identical**; the only thing the model sees differently is
  the probability numbers on the board (now decorrelated from price on ~half of
  intervals).
- Optional: a small `decorrelation_fraction` sweep (e.g. 0.0 / 0.5 / 1.0) to confirm the
  effect scales with how hard the flow is pulled off price.

## The decisive measurement

Restrict to later-turn buys on **price-divergent intervals** (target band ≠ current
price band, computed with the Fix-2 clean drift). When the board has been pushed to a
band the price is *not* in, does the model's buy target:

- the **board-pushed band** → genuine board-reading; or
- the **current price band** → it was price-following all along (F3 was a price artifact).

Judge against the permutation floor (≥20k, shuffle the target labels) and report
separately per model and per memory condition. Because the push is decorrelated from
price by construction, this cleanly separates the two channels for the first time.

| Outcome | Reading |
|---|---|
| Both models follow the **pushed band** | board-reading is real; F3 (re)confirmed, claude included |
| Both follow the **price band** | F3 was price-following; demote board-tracking fully |
| gpt follows price, claude follows board (or vice-versa) | the channel is model-specific — a new finding |

And the mechanism check for Deliverable B: recompute gpt's unbiased discovery
(peak_top_pick, buy_flow) in v3. If gpt's discovery degradation tracks **price-divergent
flow specifically** (worse when the board lies about price), the drop is board-chasing;
if it is the same as v2 regardless of decorrelation, the live board degrades discovery by
adding noise generally, not by chasing per se.

## Gates (before trusting v3)

1. **Calibration gate** (as v2): decorrelated flow must still land ~10–25 net points per
   interval without pinning the cost-curve ceiling, across band_width variants.
2. **Decorrelation achieved:** verify drift-band == price-band drops to ~chance on the
   decorrelated intervals (the whole point); report the realized coincidence rate.
3. **Sufficient n on the clean test:** with Fix 2 the clean drift is defined every turn,
   so claude's split-set n should be in the hundreds, not ~0. Confirm before interpreting.
4. **Seed/path identity:** confirm v3 price paths and winners are identical to v1/v2 on
   matched seeds (same guarantee as v2).

## Guardrails

- The decorrelated target is **never** a function of the winner or settlement — only of
  the price band and the noise RNG. Pushing toward the answer would invalidate discovery.
- Keep the depth caps fixed at the v2-calibrated values so v3↔v2 differences are
  attributable to *targeting*, not to a busier/quieter board.
- Every v1/v2/v3 delta is paired on `(cell, seed)`; report any comparison that cannot be
  matched as weaker.
- This is a new battery (new model calls). It is not derivable from existing data — the
  instrumentation (Fix 2) and the decorrelated board (Fix 1) did not exist in v1/v2.
