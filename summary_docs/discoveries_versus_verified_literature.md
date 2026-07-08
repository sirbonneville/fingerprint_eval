# Discoveries — and how they line up with the benchmark literature

*A companion to `battery_comparison_v1_vs_v2.md`. For each discovery this project
made, this records (a) the finding in one line, (b) whether and how an independent,
published benchmark arrived at a related place, and (c) the citation. Where outside
work converged on a similar result, that convergence is noted as corroboration; where
a result has no close match in the surveyed literature, that is noted as an absence in
the surveyed set rather than a claim of priority.*

*Citation status: every benchmark cited below — AMA, KTD-Fin, AlphaForgeBench, and
PredictionMarketBench — has been verified against full text. Two further papers read during
this review (TimeSeek and the blindfolded-LLM portfolio framework) were removed: on full
reading, neither shares a convergent finding with this project — the apparent overlaps were a
different information channel (TimeSeek's "information hurts" is external web retrieval, not a
live board) and a false-friend use of "survivorship bias" (the blindfolded paper's is the
classical delisting kind, not a metric-conditioning selection effect). By the rule that this
doc cites only genuinely convergent work, they are not listed. Dates and arXiv IDs are as
listed on arXiv.*

> **Revision (2026-07-08).** A later independent review (addendum in
> `battery_comparison_v1_vs_v2.md`; reproducible via
> `eval_runs/_review_naive_baseline.py`) revises **Discovery 3** and **Discovery 5**:
> the 1/3 chance floor used for the discovery metrics is the wrong null (a naive
> current-band price-follower scores ~0.59 per turn, and both models sit essentially
> at that baseline in both batteries), and the v2 collapse is largely an entry-timing
> effect triggered by a pure-noise board tilt the synthetic flow creates before the
> model's first turn. Dated notes appear in those two sections below. Discoveries 1,
> 2, 4, and 6 — including all three literature convergences — are unaffected.

---

## What actually converges (and what doesn't)

Three findings here have a genuine independent convergence in the surveyed literature — a
different group, on a different market, reaching a related conclusion by different machinery:

1. **Stable, model-driven trading personalities (Discovery 1)** — converges with *AMA*
   (distinct behavioral styles, seen across agent architectures) and, more closely, with
   *AlphaForgeBench*, which uses the same scaffold-fixed / model-varied design as this project
   and finds stable per-model risk personalities across six models.
2. **Win/return must be decomposed before reading skill (Discovery 2)** — converges with
   *KTD-Fin*, which reaches the same principle through a Barra market/style/selection-alpha
   attribution and finds most LLM headline return is passive factor exposure, not skill.
3. **Turn-to-turn churn, and that supplying action memory suppresses it (Discovery 4)** —
   converges with *AlphaForgeBench*, which documents action-flipping and attributes it to
   stateless architectures lacking persistent action memory: the exact lever the memory
   condition manipulates.

The remaining findings do **not** have a convergent match in the surveyed set, and are
labelled as such rather than presented as corroboration: the survivorship-free discovery
metric (Discovery 3 — a novel instrument; only the underlying "prove the skill signal is real"
discipline is shared with KTD-Fin), the live-board discovery collapse (Discovery 5 —
unconfirmed and confounded, awaiting the v3 experiment), and the frozen-board artifacts
(Discovery 6 — no analogue in the surveyed set). The two design elements with no close match
in the surveyed literature are the parimutuel (DPM) prediction-market microstructure used as a
controlled probe and the same-seed frozen-board versus live-board ablation that isolates
counterparty flow as a single variable. *PredictionMarketBench* appears once, in the summary
map, as the nearest-neighbour harness in the same genre — a contrast for scoping that novelty
honestly, not a convergence.

---

## Discovery 1 — Two stable, distinct trading "personalities," model-driven

**Finding.** Holding scaffold, prompt, market, and seeds fixed, gpt-4o and
claude-sonnet-4 are two stable, distinct traders (gpt concentrated / high-hold /
round-anchored; claude active / diversified / mobile), and the difference is trait-level
— it persisted across a memory flip and an environment flip.

**Related work.** The closest external result is the *Agent Market Arena* (AMA) live
benchmark, which ran two months on real BTC/ETH/TSLA/BMRN across four agent architectures
and five backbones including gpt-4o and claude-sonnet-4. It reports that agent frameworks
display markedly distinct behavioral patterns, spanning from aggressive risk-taking to
conservative decision-making, whereas model backbones contribute less to outcome
variation, and that changing the LLM within a fixed agent framework produced only modest
performance variations while switching agent architectures with the same LLM led to
substantial differences. Its behavioral analysis (RQ4) notes that although all agents
operate on identical market information, their trading strategies diverge from the first
day — e.g. one architecture stays persistently contrarian, another tends to hold — which
it attributes to each framework's inherent risk preferences.

A second benchmark, *AlphaForgeBench*, is the closer methodological match. It holds the
scaffold fixed (one prompt template across all models) and varies the backbone — the same
design logic used here — across six frontier models, and reports that different LLMs encode
distinct and stable risk personalities in their generated strategies, persisting across
runs, assets, and decoding temperatures. It groups the models into recurring archetypes
(aggressive-creative, balanced-stable, conservative-rigid), and notes that one model
(claude-sonnet-4.5) shows the most diversified factor usage, drawing on a broader set of
indicators rather than a few dominant signals. Its task is code generation (the model writes
a strategy that is then executed deterministically), not live trading, so the personality is
expressed in a different artifact than the one measured here.

**Relationship.** Two independent benchmarks bear on this from different angles. AMA varies
the *agent architecture* and finds architecture is the larger driver of profitability, with
the backbone the lesser factor; this project and AlphaForgeBench both hold the scaffold fixed
and vary the *backbone*, and both find that backbone identity alone produces a stable,
distinguishable behavioral signature. These are compatible: a backbone can move the
profitability needle less than the scaffold (AMA's claim) while still carrying a recognizable
behavioral fingerprint (the claim here and in AlphaForgeBench). AlphaForgeBench is the closer
design analogue because it shares the scaffold-fixed / model-varied setup; its diversification
observation for claude echoes the diversified/mobile profile measured here, though on
different model versions (claude-sonnet-4.5, not claude-sonnet-4) and a different task, so it
corroborates the existence of stable per-model personalities rather than the specific traits
assigned to any one model here. (Narrower note: AMA's backbone-versus-architecture comparison
is on profitability metrics and does not run the architecture-fixed, model-varied behavioral
isolation done here; AlphaForgeBench does run model-varied isolation but in code generation
rather than direct trading.)

*Sources: Qian et al., "When Agents Trade: Live Multi-Market Trading Benchmark for LLM
Agents," arXiv:2510.11695, Oct 2025. Zhang et al., "AlphaForgeBench," arXiv:2602.18481, Feb
2026 (KDD '26).*

---

## Discovery 2 — Win/return is a contaminated metric; decompose it (commitment vs. accuracy)

**Finding.** A memory-driven rise in claude's win rate is commitment, not accuracy:
`win = held_rate × P(win|held)`, and the entire gain came from holding longer while
conditional accuracy stayed flat-to-down. This generalizes to a rule: end-state-conditioned
outcome metrics confound holding behavior with skill.

**Related work.** The *KTD-Fin* benchmark makes a structurally identical argument at the
level of asset-pricing attribution. It states that a single return number lumps market
drift, persistent style tilts, and genuine stock-specific skill into one figure, and that
without decomposition a benchmark cannot distinguish "the agent made money" from "the agent
rode a factor any similarly-exposed portfolio would have ridden." It introduces a
Barra-style cross-sectional attribution that decomposes each agent's return into market,
style, and stock-selection-alpha components, and reports that LLM agents' cumulative
returns under leakage-controlled evaluation are largely explained by passive market and
style exposure, with limited evidence of persistent stock-selection alpha, concluding that
benchmarks should evaluate not only whether an agent makes money but also whether the
source of returns reflects transferable skill.

**Relationship.** This is close independent corroboration of the *decomposition discipline*
specifically. KTD-Fin decomposes return into market/style/residual via a Barra regression;
this project decomposes win into held_rate × conditional-accuracy via the harness.
Different machinery, the same underlying principle: the headline number contains a
non-skill component that must be factored out before reading skill, and once it is factored
out, much of the apparent performance is not skill. An independent group, on a different
market (CSI300 equities, ten models), reached the same methodological conclusion.

*Scope note on the term "memory."* KTD-Fin also controls "memory," but in a different sense
than this project's memory axis. KTD-Fin's control is *pretraining-memory leakage* — it
masks real tickers and dates so the model cannot draw on memorized real-world knowledge
(its "bright" vs. "blinded" conditions). This project's memory axis is *within-task
rationale replay* — whether the model sees its own prior reasoning from earlier turns. The
two are unrelated mechanisms that share a word; KTD-Fin corroborates the decomposition
point above, not this project's memory-condition findings. (Separately and only thematically
adjacent: KTD-Fin reports that, stripped of ticker identity under masking, a representative
model declined to trade in all 25 of 25 tested cells — an instance of information
availability changing trading activity, related in spirit to Discovery 5 but not the same
measurement.)

*Source: KTD-Fin — "From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading
Agents on Stock Markets," arXiv:2605.28359, May 2026.*

---

## Discovery 3 — A survivorship-free discovery-skill gap (and top_pick is biased)

**Finding.** On end-state-independent metrics (peak_top_pick, buy_flow), claude identifies
the winning band above chance and above gpt in both batteries; the biased top_pick metric
overstated skill because it silently conditions on holding to the bell.

> **Revision (2026-07-08).** "Above chance" here means above 1/3, and 1/3 is the
> wrong null: a zero-skill trader that buys the band the price currently sits in
> matches the winner ~59% of the time across the five decision points, and matched to
> the turns each model actually bought on, both models score essentially *at* that
> naive baseline in both batteries (residuals −0.03 to +0.02). The claude>gpt gap is
> therefore largely an entry-timing/attention difference (claude keeps buying at
> later, informative turns), not a band-picking skill gap. The survivorship lesson —
> top_pick overstates skill by conditioning on holding to the bell — **stands**, and
> the peak/buy_flow instrument remains the right correction; it just needs the naive
> price-follower as its null. See the addendum in `battery_comparison_v1_vs_v2.md`.

**Related work.** No surveyed benchmark uses this exact peak/buy_flow construction — the
instrument has no close match in the surveyed set. The principle it serves — that
end-state or contamination-sensitive metrics overstate skill, so a selection-robust measure is
needed — is shared with *KTD-Fin*, whose Barra attribution finds most LLM headline return is
passive factor exposure rather than selection skill. The convergence is the underlying
discipline (prove the skill signal is real, not a selection or contamination artifact), the
same discipline that underwrites Discovery 2; it is not a convergence on the specific metric.

**Relationship.** The instrument (peak-exposure top-pick and buy-flow) is novel within the
surveyed set; the discipline it serves converges with KTD-Fin. The survivorship issue corrected
here is a metric-level selection effect (top_pick silently conditions on episodes the model
held to the bell), which is a distinct mechanism from the contamination KTD-Fin controls — so
KTD-Fin corroborates the discipline, not the specific correction.

*Source: KTD-Fin (above).*

---

## Discovery 4 — Churn / direction-flipping is real, and memory suppresses it

**Finding.** claude is a churner (high direction-switch rate, frequent exit-to-flat); memory
converts it toward a committed accumulator (−0.40 direction_switches, exit-flat 0.45→0.18).
The associated synthesis hypothesis (not a confirmed finding): memory affects a model in
proportion to how much it would otherwise contradict its own past actions — a self-consistent
model gives memory little to act on, a self-inconsistent one gives it a large surface.

**Related work.** *AlphaForgeBench* documents the churn phenomenon and attributes it to the
same cause the synthesis hypothesis names (this is verified against the full text, not the
abstract alone). It shows that as direct trading agents, LLMs exhibit extreme run-to-run
variance, inconsistent action sequences even under deterministic decoding, and irrational
action flipping across adjacent time steps, and attributes these to stateless autoregressive
architectures lacking persistent memory of prior actions, together with sensitivity to
continuous-to-discrete action mappings in portfolio allocation. Its appendix runs a per-model
instability scan (including a claude and a gpt model) in which models differ in how stable
they are run-to-run.

**Relationship.** This cuts two ways, and both are worth stating.

*As corroboration:* the phenomenon (action flipping turn to turn) matches the direction-switch
churn measured here, and the mechanism AlphaForgeBench names — statelessness, no persistent
action memory — is exactly the lever the memory condition manipulates. Mem-on replays prior
actions and rationales, supplying the persistent action memory they identify as missing, and
the result is suppressed flipping (−0.40 switches). That is independent support for both
halves of the discovery.

*As a challenge:* AlphaForgeBench's larger thesis is that direct action-emitting trading
benchmarks are so dominated by run-to-run noise that single-run rankings are unreliable, which
is why it abandons direct trading for code generation. This project's harness *is* a
direct-trading benchmark, so that critique applies to it. The mitigations here are the ones
AlphaForgeBench's critique implies are necessary: N=30 episodes per cell rather than single
runs, fixed per-episode seeds shared across batteries, and matched-seed permutation tests
(the gpt discovery drop at p≤0.003) that test a contrast against precisely the run-to-run
variance AlphaForgeBench warns about. The honest reading is that AlphaForgeBench both supports
the churn result and raises the bar the harness has to clear to be trusted as a direct-trading
instrument.

*Bound on the per-model match:* AlphaForgeBench's per-model instability and personality
assignments are on different model versions (claude-sonnet-4.5, gpt-5.2) and a different task
(BTC daily long/cash), and its "churn" is primarily run-to-run and step-to-step action
consistency, whereas the churn measured here is within-episode direction switching. It
therefore corroborates that churn exists, that the mechanism is statelessness, and that supplying
action memory should suppress it — but not the specific direction of the gpt-versus-claude churn
contrast found here.

*Note carried from the comparison doc: the self-inconsistency synthesis remains a hypothesis;
the v2 "gpt churns more and gains memory sensitivity" co-occurrence is confounded with the
live board. AlphaForgeBench raises its plausibility (it provides an independent per-model
self-consistency ranking) but does not test memory, so it does not confirm the hypothesis
here. AlphaForgeBench also reports model-specific decoding-temperature sensitivity in its
direct-trading appendix, which bears on the planned memory×temperature experiment.*

*Source: Zhang et al., "AlphaForgeBench: Benchmarking End-to-End Trading Strategy Design with
Large Language Models," arXiv:2602.18481, Feb 2026 (KDD '26).*

---

## Discovery 5 — More live information can degrade performance (gpt discovery collapse)

**Finding.** Turning the board live dropped gpt's unbiased discovery from above-chance to
chance (matched-seed, p≤0.003, both memory conditions), while leaving claude unchanged or
slightly improved. v1 did not over-credit gpt; the live board degraded a real capability.

> **Revision (2026-07-08).** The matched-seed contrast stands, but both halves of the
> framing are revised. The v1 "capability" was price-following at informative turns,
> not forecasting (both models sit at the naive current-band baseline once it is
> matched to their buy timing). And the mechanism is now substantially identified
> within existing data: on the live board gpt front-loads ~84% of its buy notional to
> t=0 (vs 16% frozen) — where no strategy can beat chance — apparently anchoring on a
> pure-noise board tilt the synthetic flow creates *before the model's first turn*
> (gpt buys the tilted favorite at t=0 72% of the time; claude buys the middle band
> 100% of the time and keeps its late informative buys). The sharper statement of this
> discovery is: *a salient spurious signal at first decision induces premature
> commitment in gpt-4o but not claude-sonnet-4* — closer to the LLM anchoring-bias
> literature than to a generic "more information hurts" effect. The v3
> decorrelated-flow run remains the clean confirmation. See the addendum in
> `battery_comparison_v1_vs_v2.md`.

**Related work.** No surveyed benchmark converges with this finding. The broad "more
information need not help" theme appears in the literature, but the surveyed instances operate
through a different channel (external retrieval / tool use degrading forecast calibration)
than the one here (a live market board, i.e. endogenous microstructure), so none is a
convergent match and none is cited as corroboration.

**Relationship.** This is a finding of this project with no independent corroboration in the
surveyed set, and it is internally confounded: the *what* is established (gpt's unbiased
discovery drops from above-chance to chance when the board goes live, matched-seed, p≤0.003),
but the *why* (board-chasing versus generic live-board noise) is unresolved, because the
synthetic drift correlates with the readable price line. It is the result most dependent on the
v3 decorrelated-flow experiment to settle, and should be presented as a standalone, unconfirmed
result rather than as something the literature backs.

*No citation — no convergent match in the surveyed set.*

---

## Discovery 6 — Frozen-board artifacts: orthogonality, the $333 anchor, the axis hierarchy

**Finding.** Three things that looked like model traits in v1 turned out to be properties of
the frozen board: the gpt-market/claude-memory orthogonality, gpt's exact-budget/3 sizing
anchor (59–74% → 0%), and the band_width↔volatility axis inversion.

**Related work.** No surveyed benchmark runs a same-seed frozen-versus-live ablation, so the
specific result that named v1 traits dissolve under an environment change has no close match
in the surveyed set. The enabling concern — that environment or execution conditions
contaminate what looks like a model property — is the motivation behind both AlphaForgeBench
(execution-induced instability) and KTD-Fin (leakage and style contamination).

**Relationship.** The demonstration that specific behavioral traits dissolve under an
environment flip has no close analogue in the surveyed literature. It rests on N=30, two
models, and one synthetic-flow design, which bounds how far it generalizes.

*No direct citation; the closest motivating concerns are execution-induced instability
(AlphaForgeBench, above) and leakage/selection contamination (KTD-Fin, above).*

---

## Summary map

- **DPM/prediction-market microstructure as a probe (contrast, not a convergence).** A
  nearest-neighbour harness exists: *PredictionMarketBench* (Arora & Malpani, arXiv:2602.00133,
  Jan 2026) is also a prediction-market agent harness with binary contracts, maker/taker
  execution, fee modeling, and a stateless tool-calling LLM loop. It is a deterministic replay
  benchmark over real historical Kalshi *limit-order-book* data, built for execution-realistic
  backtesting — its headline result is that an aggressive LLM agent loses to fees and settlement
  while a fee-aware algorithmic strategy stays competitive — and its released results test a
  single small model (gpt-4.1-nano) as a baseline, making it an infrastructure paper rather than
  a multi-model behavioral study. The instrument here is generative, parametric, and
  fictional-token, built for controlled causal ablation (sweep one variable, hold seeds fixed),
  and its market is a synthetic DPM/parimutuel cost-function rather than a replayed limit-order
  book. This entry exists to scope novelty honestly — to show the genre is not new — not to
  claim a convergent finding.
- **Same-seed frozen-versus-live counterparty ablation.** No close match in the surveyed set.
- **Decomposition discipline.** Independently arrived at by KTD-Fin (Barra return attribution)
  — convergent corroboration. (The survivorship correction in Discovery 3 is a separate,
  metric-level selection effect with no convergent match in the surveyed set.)
- **Stable per-model personalities.** Corroborated by two benchmarks — AMA (across
  architectures) and AlphaForgeBench (across backbones, scaffold fixed, the closer design
  match). Convergent.
- **Churn-and-memory result.** Phenomenon and mechanism independently supported by
  AlphaForgeBench — which also, in its larger thesis, raises a methodological challenge to the
  direct-trading paradigm this harness uses (addressed here by N=30, fixed seeds, and
  matched-seed permutation tests).

The decomposition, personalities, and churn results have independent corroboration in the
surveyed literature. The remaining findings (the survivorship-free metric, the live-board
discovery collapse, the frozen-board artifacts) have no convergent match in the surveyed set
and are labelled as such; the ablation design and the synthetic-probe framing have no close
match either. PredictionMarketBench is the nearest neighbour on the prediction-market harness
genre and is a contrast, not a convergence. One near-homonym is flagged in-text to avoid
overclaiming: KTD-Fin's "memory" is pretraining-leakage masking, not this project's
rationale-replay memory.