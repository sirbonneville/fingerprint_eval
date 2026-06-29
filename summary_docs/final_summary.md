# **fingerprint-eval: Isolating Model-Driven Behavior in a Controlled Prediction-Market Instrument**

*A controlled instrument for measuring how an LLM actually behaves when it trades inside a specific kind of Delphi prediction market and what changes that behavior.*

---

## What this is

This project builds a controlled, repeatable instrument: take a trading model, run it
through a synthetic prediction market many times, change exactly one variable at a time,
and read the *distribution* of its behavior rather than any single run. 

The point is not to play the market well but instead it is to measure how a model's trading behavior responds to the market, to its own memory, and to the presence of other traders, cleanly enough that the results can be trusted and re-checked.

## Why we built it

I create crypto **price-bucket prediction markets** on Delphi regularly: markets that ask "which price band will this token land in at settlement?" 

Seeing a lot of agentic trading activity on these CPM markets, I was curious how different models rationalize and trade in these markets. We already know two different models will trade differently. This is implied and is not the primary finding.

The question was *what* differs, at a granular level, and what actually moves that
behavior. So the model became a **probe**: a way to stress-test a market design before it
goes live.

## The environment

The synthetic market is built to mirror the real Delphi price-bucket market as closely as possible:

- A **fictional token** with a made-up price and no real dates or prices. This is deliberate data discipline: a real token would let the model pattern-match its way to the answer instead of trading. A fictional source name (e.g. "GrayCoinPrice") stands in for the data feed, so the model gets the *shape* of a real settlement disclosure without any real-world  
hook.
- **Three price bands** (below / inside / above a middle range). You hold the band you think
the price lands in when trading closes.
- A **2-hour trading window with five decision points** — a turn every 30 minutes
(`hours: 2.0, interval_min: 30.0` in the cell config). At each turn the model sees the
price revealed so far and chooses to buy, sell, or hold across the bands. Price updates and
synthetic-trader flow happen *between* turns, so the board can move many times within an
episode even though the model itself acts at five points.
- A **centered random walk** for the price path. Each step moves the price up or down by a
random amount with no built-in trend, so any single path wanders and can end well above or
below where it started, but the *average* over many paths stays relatively close to the starting price.
- The **same settlement disclosure a real Delphi agent receives**: how the market settles, the data source, the close time, the settlement time, and the finality of the closing price (the equivalent of CoinGecko's 8pm final print). The model knows the rules of the market the same way a live agent would.

## The pricing mechanism (DPM)

The market uses Delphi's **dynamic-parimutuel (DPM) cost-function** pricing. The math isn't published, so we recovered the cost function using the Delphi SDK and the Agentic Trading Toolkit, and verified it against the live on-chain contract before relying on any result. In short:

- **Cost / collateral:** `C(q) = k·√(Σqᵢ²)`
- **Spot price:** `pᵢ = k·qᵢ / √(Σqᵢ²)`
- **Implied probability:** `πᵢ = qᵢ² / Σqᵢ²` — probability is price squared (a 0.51 share
price is ~26% implied probability, not 51%).

The verification is reproducible from a script.

## What we ran

Each battery is the full cross-product below; we ran **two** batteries (see next section).


|                          |                                                                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **Models**               | gpt-4o, claude-sonnet-4                                                                                                     |
| **Memory**               | off (stateless) and on (the model's own prior rationales/trades replayed into its context on later turns)                   |
| **Axes (4)**             | volatility (of the generated price path), temperature, knowability (how early trading closes before settlement), band width |
| **Cells**                | 13 (baseline + the four axes' values) × 2 models × 2 memory conditions = 52                                                 |
| **Runs per cell**        | 30                                                                                                                          |
| **Episodes per battery** | 1,560                                                                                                                       |


"Memory" is concrete: the model's own previous reasoning and trades are logged and fed back
to it on later turns, so it can see what it already decided and why.

## The two batteries

### Battery 1: Without synthetic trading

In the first battery the model is the only trader, so the implied-probability board moves only when the model itself trades. Between its turns the odds hold steady. This is the control: it measures how a model trades when it acts purely on the price being revealed to it, with no other participants on the board.

A model trading this way never sees the board move on its own and never sees other participants take positions. That is precisely the gap the second battery is built to probe, which is to see whether, and how, a model's decisions change when there is live counterparty activity to read alongside the revealed price.

### Battery 2: With synthetic trading

The second battery is **identical to the first with one change**: synthetic "noise" traders
move the board between the model's turns. They are informed only by the *revealed* price path (never the settlement outcome so there's no peeking at the future), tilt toward the band the price is currently in with a small momentum nudge, and include a per-episode contrarian fraction (~5–30%) so the board stays a market rather than a mirror of the latest price. 

Their size is capped so they move the board about one trader's-worth per interval, they route through the **same** trade function the model uses (identical DPM math), and they are seeded off the **same episode seed** as battery 1, so for any given seed, both batteries have the *same price path and the same winning band*, and the only difference is whether the
board is alive. 

The model sees this only through the moved probability numbers; the prompt
template is byte-identical to battery 1.

Battery 1 is the control arm; battery 2 is the treatment; the contrast between them is the experiment that was supposed to reveal unique behaviors and limitations of the two models especially as configuration "knobs" (the axes) were programmatically adjusted.

---

## What we found

### The two personalities (across models)

Holding everything fixed, the two models are two stable, fingerprint-able traders:

- **gpt-4o**: Concentrated and patient: 
  - roughly one or two trades
  - a large stake
  - one band
  - rarely changes its mind. 
  - It bets *exactly* a third of its budget ($333.33) about 63% of the  
  time, like a default it falls back on.
- **claude-sonnet-4**: Active and diversified: 
  - more trades
  - smaller situational stakes
  - spreads across bands
  - reverses direction more often.

These held across configs *and* across the environment flip from battery 1 to battery 2. That stability is the evidence they are traits, not noise and the reproducibility of the data included in the dataset (cells, runs.JSONL, etc.) as well as the read-only scripts and adjacent tests (calibration tests or otherwise) were designed to make this evident. 

### What memory does

Memory turns claude from a churner into a committer. It stops flip-flopping and holds positions to the end. Its win rate rises, but that is **commitment, not accuracy**.

Decomposing `win = hold-rate × accuracy-when-holding`:


| claude                      | memory off | memory on |
| --------------------------- | ---------- | --------- |
| Holds a position to the end | 55%        | 82%       |
| Wins                        | 40%        | 56%       |
| **Accuracy when it holds**  | **73%**    | **68%**   |


The entire win-rate gain comes from holding more often; the chance the held position is the
winning one is flat (slightly down). Memory makes it commit, not pick better.

### What moves behavior, and what the two batteries showed

In battery 1 the apparent ranking of what moved behavior was band width > temperature >
volatility ≈ knowability. 

Comparing the two batteries refines that picture: several effects that looked like model traits on the frozen board turned out to depend on the board being frozen, and changed once it went live. Namely:

- The gpt-reacts-to-market / claude-reacts-to-memory split
- Gpt's exact-$333 anchor
- and the band-width-over-volatility ordering.

The two-personality finding held across both batteries; these did not. Turning the board live also moved gpt's unbiased discovery from above-chance toward chance (matched-seed, p ≤ 0.003) while leaving claude roughly unchanged. That result is still confounded, because the synthetic flow correlates with the readable price line: a decorrelated-flow run is what would separate the two (this is proposed as a v3 battery, see the `v3_decorrelated_flow_spec.md`).

A couple of single-battery readings are worth stating as outcomes, neutrally:

- Battery 1 suggested volatility drove engagement, but that appears to have been a feature of the frozen board rather than a response to volatility itself, since it did not carry over once the board was live. 
- The dead-window (knowability) axis looked at first like it reduced the model's discovery; on inspection, when trading closes early the price keeps moving afterward, so the winning band is set by data the model never sees, but even a perfect predictor caps around 60%. So that axis measures how answerable the question is under the market's geometry, not the model's ability.

---

## Discoveries and the published literature

Three of the findings here appear to have been reached independently in this work and also
corroborated, or previously arrived at from a different angle, in published benchmark
literature:

- **Stable, model-driven trading personalities**: Corroborated by *Agent Market Arena* and,
more closely, by *AlphaForgeBench*, which uses the same scaffold-fixed, model-varied design.
- **Win and return must be decomposed before they can be read as skill**: Corroborated by *KTD-Fin*, whose factor attribution finds most LLM headline return is passive exposure
rather than selection skill.
- **Turn-to-turn churn, and that supplying action-memory suppresses it**: Corroborated by *AlphaForgeBench*, which attributes action-flipping to the absence of persistent action memory, the same lever our memory condition manipulates.

## One open hypothesis

Memory may bite a model in proportion to how much it would otherwise contradict itself:  
gpt-4o decides once and holds, so replaying its reasoning gives memory nothing to act on;  
claude churns, so its own prior reasoning is a large surface for memory to grab and suppress.

This is a hypothesis, not a finding. The clean test is a memory × temperature run, since  
higher temperature makes gpt-4o churn more, so memory *should* start to bite it if the  
mechanism is right.

---

## Reproducibility

All of the data is reproducible using the read-only analysis scripts. The full run data is retained on disk for both batteries. This includes:

- every model
- every memory condition
- every cell
- every run
- and the full per-turn record within each episode (prompts, the model's actions and rationales, synthetic-trader flow where present, the price path, board state, and settlement).

The scripts take a battery directory as input and regenerate the tables and statistics from those raw logs, so the numbers in the findings documents can be recomputed and checked against the source data.

## Scope and limits

- Two models, one market family, synthetic centered markets, a single model trader, N=30 per
cell. In-frame these are real measurements; out-of-frame they are hypotheses.
- The centered-walk design limits some questions by construction--for a few axes a flat result means the question is unmeasurable under this market geometry rather than zero in the real world.
- Money is out of scope as a measurement. Profit-and-loss differences between conditions are within noise at N=30, so we make no money claims, and the model was never shown its running P&L across episodes. Each episode starts fresh at $1,000. Feeding P&L forward would be a future axis, not a result we have.

## Where to look next

- `findings_without_synth_trading.md`: The full results for battery 1 (frozen board).
- `findings_with_synth_trading.md`: The full results for battery 2 (live board).
- `battery_comparison_v1_vs_v2.md`: The side-by-side contrast that carries the what-dissolved-and-what-survived story.
- `discoveries_versus_verified_literature.md`: The discovery-by-discovery audit against the published benchmark literature.

