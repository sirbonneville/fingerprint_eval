# Prediction-Market Agent-Evaluation Harness — Build Spec (v1)

## What this is

A **controlled instrument** for studying how a language model behaves as a prediction-market trader. You drop a model into a simulated price-bucket market it cannot see the future of, record every trade and every stated rationale, and — the actual point — **change one market variable at a time and watch whether behavior shifts**. The insight comes from intervention, not introspection.

## The one thing to get right

**This is a causal instrument. Interventions are the product.** The value is in holding everything fixed, sweeping one variable (volatility, knowability, band width), and observing the behavioral delta. The model's written rationale is *observability color* — the story it tells — not the true cause of its decisions (models confabulate; stated reasons ≠ real reasons). Build around clean interventions; treat rationale as supporting data.

**Runs-per-cell is your new `n`.** Same model + same prompt run twice gives different trades (sampling). A single run per condition is noise. Every experimental cell must be run **N times** (start N=10) and you read the *distribution* of behavior, not one trace. This is the discipline that makes causal claims valid — design it in from line one; it is expensive to retrofit.

## The one architectural decision

A **generic harness** talks to a **market** through a thin fixed interface. There is **one concrete implementation**: the price-bucket market. The harness never knows what kind of market it's driving. This buys you future extensibility (a second market type = a second implementation, harness unchanged) **without paying for generality now** (you define the interface against a real example, not in the abstract).

Do **not**: build generic market support, integrate testnet, or add REE in v1. Each is a clean later addition *because* the seam is in the right place.

---

## The four pieces

### 1. DPM core (build and test FIRST, in isolation)

A faithful dynamic-parimutuel market — mirror the real Delphi curve, not a toy. This is the most reused component (identical in every later version) and the one where correctness silently matters most.

```
class PriceBucketMarket(Market):    # implements the thin Market interface
    pool_state                       # shares outstanding per outcome, pool balances, k
    config                           # bands, settlement rule text, fee, timing

    cost_to_buy(outcome, shares) -> cost      # integrate along the curve
    apply_trade(outcome, shares, side) -> fill_price   # mutate state, return fill
    current_prices() -> [p0..p3]              # implied odds per outcome from pool
    settle(winning_outcome) -> {addr: payout} # winners split pool, fee applied
```

Test in isolation against hand-computed inputs **before** wiring anything else: a known buy moves price by the known amount; settlement pays out exactly the pool minus fee; prices always sum to 1. If this is wrong, every downstream result is silently wrong.

### 2. The thin Market interface (the seam)

The harness only ever calls these. Everything market-type-specific lives behind them:

```
interface Market:
    outcomes() -> list                        # how many, what they mean
    current_state() -> {prices, time_to_close, pool, my_position}
    apply_trade(action) -> result
    settle(winning_outcome) -> payouts
    settlement_rule_text() -> str             # the prompt-facing description
```

Three things vary by market type and are isolated here: **settlement rule**, **outcome structure**, **payout math**. The harness is agnostic to all three.

### 3. Price generator (your intervention instrument)

A parametric synthetic price path — this is *how you set knowability deliberately*, the thing observation could never control.

```
generate_path(anchor, hours, interval_min, volatility, drift) -> [(t, price)]
```

- Random walk with **explicit volatility and drift knobs** — these are experimental variables, not constants.
- Generates the full path up front (creation → settlement reference), including the **post-trading-close segment** — the "dead window" the trader can't act in. Controlling how much the price moves there is how you set outcome knowability.
- Determinism: seed the RNG so a given path is reproducible (matters for both clean experiments and the eventual REE port).

### 4. The harness loop (market-agnostic)

```
for each timestep t in path up to trading_close:
    state = market.current_state()           # prices, time_to_close, my position, cash
    prompt = build_prompt(
        settlement_rule, outcomes, state,
        price_path[:t],     # CARDINAL RULE: never any price after t
        my_history,
    )
    response = model(prompt)                  # rigid action schema, see below
    action = parse(response)                  # {action, outcome, size, rationale}
    market.apply_trade(action)
    log(t, prompt, response, action, state)   # FULL input-output pair, line one
settle against price_path[settlement_reference]
compute model's footprint
```

**Cardinal validity rule — information discipline.** At timestep `t` the model must never see any price data after `t`. Not via the path, not via the token name, not via a date it can reason about. The easiest thing to get subtly wrong; if the model can peek at the future, your "trader" is reading the answer and the experiment is void. Be paranoid: use a fictional token with no inferable real-world price, strip any date the model could anchor to.

**Action schema (rigid format, free strategy).** The model chooses *what/when/how much* freely, but must emit a fixed structure or parsing breaks nondeterministically:

```json
{ "action": "buy" | "sell" | "hold",
  "outcome": 0-3,
  "size": <number>,
  "rationale": "<free text — logged as observed behavior, NOT trusted as cause>" }
```

**Logging from line one.** Persist the full prompt *and* response for every timestep, not just the parsed action. This is your whole dataset; it's also exactly what the REE-reproducibility step needs later (input→output pairs to replay). Build logging as if reproducibility matters even in v1.

---

### 5. Experiment runner (the actual product — now load-bearing)

This is what turns "a simulator" into "a causal instrument." It holds variables fixed, sweeps one, runs each cell N times, logs the distribution.

```
grid = {
  volatility:  [low, mid, high],
  knowability: [window 0, 60, 120 min],   # how much price can move after close
  band_width:  [tight, wide],
}
for each cell in chosen sweep (vary ONE axis, freeze the rest):
    for run in 1..N:                        # N≈10 — this is your 'n'
        path = generate_path(..., seed=run)
        result = harness(market(cell), model, path)
        record(cell, run, footprint, rationale_log)
report PER CELL: distribution (median + spread) of each behavior metric, not a single value
```

**Behavior metrics to capture per run** (so you can read interventions): entry timing, trade count, per-trade size distribution, buy/sell churn, bucket spread, win/loss, volume-by-time-to-close. Same descriptive vocabulary throughout, so cells are comparable.

**Reading it:** a behavior that shifts consistently across a swept axis *while everything else is frozen and across N runs* is a **causal** finding ("higher volatility causes this model to hedge"). A shift smaller than the run-to-run spread is noise. That comparison — between-cell delta vs within-cell spread — is the entire analysis.

---

## Build order (each step independently testable, kills the project early & cheap)

1. **DPM core** + isolation tests. Nothing else until a known buy moves price correctly and settlement pays out exactly.
2. **Market interface** wrapping the DPM. Confirm the harness can read state and apply a trade through the seam.
3. **Price generator** with vol/drift knobs + seeding. Eyeball a few paths.
4. **Harness loop** against ONE model, ONE hardcoded market, ONE path. Run once, read the trades — are they even sane? (Information-discipline audit here.)
5. **Experiment runner** — N runs per cell, one swept axis. First real experiment: sweep volatility, freeze the rest.
6. **Analysis** — between-cell deltas vs within-cell spread.

Later, clean additions (not rewrites): real price paths (swap the generator), testnet (second market backend + live settlement source), REE (swap the inference backend; logging already shaped for it), a second market type (second Market implementation).

## v1 question this answers

*How does a model trade a price-bucket market, and how does its behavior shift as I vary volatility / outcome-knowability / band width?* — fully answerable with a hardcoded market and synthetic paths. Genuinely novel as an agent-behavior eval, needs no real market and no REE to be interesting, and every later ambition is additive.

## What it can and can't tell you

- **Can (strong):** causal behavioral claims from intervention — "varying X changes how the model trades, holding all else fixed, across N runs." This is the prize.
- **Can (medium):** a behavioral fingerprint of a given model as a market trader, comparable across models.
- **Can't:** the *true* internal cause of any single decision. The rationale is the model's stated story (observed behavior), not verified cause. Treat it as color; let the intervention deltas carry the causal weight.