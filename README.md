# agent_eval_harness

A **controlled instrument** for studying how LLMs behave as traders in synthetic
Delphi-style price-bucket prediction markets. The harness runs many episodes under
fixed prompts, changes one market variable at a time, and reads the *distribution* of
behavior — not any single trace. Interventions are the product; stated rationales are
observability color, not ground truth.

Built to stress-test market designs before they go live: same DPM math as on-chain
Delphi, strict information discipline (models never see settlement until the episode
ends), and reproducible seeded paths.

---

## What it measures

Each episode drops a model into a fictional token market with:

- **Three price bands** (below / inside / above a middle range)
- A **2-hour window, five decision points** (one turn every 30 minutes)
- A **centered random walk** price path (no built-in trend)
- **Dynamic parimutuel (DPM) pricing** recovered from the Delphi SDK and verified
  against the live contract
- Optional **memory**: the model's own prior rationales replayed on later turns
- Optional **synthetic traders** (v2 battery): background flow moves the implied-probability
  board between the model's turns

The battery sweeps four axes one at a time — **volatility**, **temperature**,
**knowability** (dead window before settlement), **band width** — across two models and
two memory conditions (52 cells × 30 episodes = 1,560 episodes per battery).

---

## Quick start

### Requirements

- Python ≥ 3.9
- An [OpenRouter](https://openrouter.ai/) API key for live model runs

### Install

```bash
git clone <repo-url>
cd agent_eval_harness
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY and optionally OPENROUTER_MODEL
```

### Smoke test (one API call)

```bash
PYTHONPATH=src python3 -m agent_eval_harness.cli --ping
```

### Single-axis sweep

```bash
PYTHONPATH=src python3 -m agent_eval_harness.cli \
  --model openai/gpt-4o \
  --axis volatility \
  --values 0.5,1.0,1.5,2.0 \
  --n 10
```

### Full battery (resumable)

```bash
PYTHONPATH=src python3 -m agent_eval_harness.battery \
  --models openai/gpt-4o,anthropic/claude-sonnet-4 \
  --label official-v1 \
  --memory both \
  --n 30 \
  --concurrency 4
```

**Live-board battery (v2)** — same seeds and prompts; synthetic traders move the board
between turns:

```bash
PYTHONPATH=src python3 -m agent_eval_harness.battery \
  --models openai/gpt-4o,anthropic/claude-sonnet-4 \
  --label official-v2 \
  --synthetic-traders \
  --ignore-providers azure \
  --memory both \
  --n 30 \
  --concurrency 4 \
  --out eval_runs/battery_WITH_SYNTH_TRADING
```

Run the **calibration gate** before a full v2 launch:

```bash
PYTHONPATH=src python3 eval_runs/calibration_gate.py
```

**Dry run** (no API calls; deterministic stand-in model):

```bash
PYTHONPATH=src python3 -m agent_eval_harness.battery \
  --dry-run --models offline-stand-in --label smoke --n 2
```

Resume a crashed battery into the same session folder:

```bash
PYTHONPATH=src python3 -m agent_eval_harness.battery \
  --resume eval_runs/battery_WITH_SYNTH_TRADING/2026-06-29_official-v2 \
  --models openai/gpt-4o,anthropic/claude-sonnet-4 \
  --synthetic-traders --memory both --n 30
```

---

## Analysis (read-only)

Analysis scripts live under `eval_runs/` and consume completed battery output. They do
not call models.

| Script | Purpose |
|---|---|
| `master_analysis.py` | Personality fingerprints, memory effects, axis sweeps, permutation floors |
| `knob_diagnostics.py` | Discovery, knowability ceiling, within-label checks |
| `within_label_check.py` | Response to realized price range within a fixed label |
| `board_reaction.py` | Board-movement tracking (v2 only) |
| `allin_check.py` | All-in attempt rates and direction |
| `calibration_gate.py` | Synthetic-flow calibration before v2 runs |
| `depth_table.py` | Market liquidity / conviction-unit tables |
| `_audit.py` / `_phase2.py` | Independent cross-battery audit reducers |

Example:

```bash
PYTHONPATH=src python3 eval_runs/master_analysis.py \
  eval_runs/battery_WITHOUT_SYNTH_TRADING/2026-06-27_official-v1
```

---

## Tests

```bash
PYTHONPATH=src python3 -m pytest
```

DPM math, harness loop, synthetic flow determinism, battery provider routing, and
analysis helpers all have unit tests under `tests/`.

---

## Repository layout

```
agent_eval_harness/
├── src/agent_eval_harness/     # Core package
│   ├── market.py               # DPM price-bucket market
│   ├── price_path.py           # Seeded random-walk generator
│   ├── harness.py              # Episode loop (turns, prompts, trades)
│   ├── experiment.py           # Cell runner, calibration, parallelism
│   ├── battery.py              # Full battery orchestrator (resumable)
│   ├── synthetic.py            # Synthetic trader flow (v2)
│   ├── model.py                # OpenRouter client (+ provider routing)
│   ├── prompt.py               # Prompt templates (information discipline)
│   ├── analysis.py             # Single-run report formatting
│   └── cli.py                  # CLI entrypoint
├── eval_runs/                  # Battery outputs + analysis scripts
│   ├── battery_WITHOUT_SYNTH_TRADING/   # v1 (frozen board)
│   └── battery_WITH_SYNTH_TRADING/      # v2 (live board)
├── battery_docs/               # Per-battery findings write-ups
├── summary_docs/               # Cross-battery comparison + executive summary
├── architectural_specs/        # Build spec, DPM spec, v3 proposal
├── auxiliary_docs/             # Earlier single-model notes
└── tests/
```

Each completed battery cell writes:

```
<model>__mem-<on|off>__axis-<name>__val-<value>__n30/
    config.json      # frozen CellConfig + git hash
    runs.jsonl       # one JSON line per episode (full turn log)
    metrics.json     # per-cell footprint/discovery distributions
    summary.txt      # human one-pager
```

`runs.jsonl` is gitignored; regenerate locally by re-running cells or restore from your
own archive. `metrics.json` and summaries may be present in the repo depending on what
was committed.

---

## Documentation map

Start here for the narrative; drill into specs and raw data as needed.

| Document | Contents |
|---|---|
| [`summary_docs/final_summary.md`](summary_docs/final_summary.md) | Executive overview: instrument, both batteries, headline findings |
| [`summary_docs/battery_comparison_v1_vs_v2.md`](summary_docs/battery_comparison_v1_vs_v2.md) | Independent audit + matched-seed v1↔v2 comparison (three-bucket framework) |
| [`battery_docs/findings_without_synthetic_trading.md`](battery_docs/findings_without_synthetic_trading.md) | v1 (frozen board) full findings |
| [`battery_docs/findings_with_synth_trading.md`](battery_docs/findings_with_synth_trading.md) | v2 (live board) full findings |
| [`architectural_specs/spec.md`](architectural_specs/spec.md) | Original build spec and design principles |
| [`architectural_specs/dpm_spec.md`](architectural_specs/dpm_spec.md) | DPM cost-function derivation and verification |
| [`architectural_specs/v3_decorrelated_flow_spec.md`](architectural_specs/v3_decorrelated_flow_spec.md) | Proposed v3 battery (decorrelated synthetic flow) |
| [`summary_docs/discoveries_versus_verified_literature.md`](summary_docs/discoveries_versus_verified_literature.md) | Overlap with published agent-trading benchmarks |

---

## Headline results (very short)

Two batteries, matched seeds, 1,560 episodes each:

- **Two stable personalities** — gpt-4o: low-activity, concentrated; claude-sonnet-4:
  active, diversified. Rank order survives both batteries.
- **claude discovers the winning band better than gpt** on unbiased metrics in both
  batteries; gpt is at chance on a live board.
- **Memory → commitment, not accuracy** for claude (win decomposition); gpt memory
  touches activity only.
- **Environment-dependent, not artifact:** v1 gpt discovery was genuinely above chance
  on a frozen board and collapsed when the board went live — competence depends on the
  venue.
- **Frozen-board artifacts:** gpt's $333.33 budget/3 anchor, the orthogonality 2×2,
  and the band_width↔volatility axis hierarchy.

See `summary_docs/battery_comparison_v1_vs_v2.md` for audit tables, p-values, and open
questions.

---

## Design discipline

- **Effect vs spread** — compare against the measured within-config noise floor (four
  identical draws), not eyeballing.
- **Permutation floors** — ≥20k shuffles under every candidate effect.
- **End-state-independent discovery** — use `peak_top_pick` and `buy_flow_on_winner`;
  `top_pick` conditions on holding to the bell and inflates skill.
- **Information discipline** — fictional token, no real dates/prices; synthetic flow
  reads only the revealed price path, never the settlement outcome.
- **Matched seeds** — v1 and v2 share identical price paths and winning bands per
  `(cell, seed)`; only the board-alive flag differs.

---

## License

See repository defaults. API usage is billed through your OpenRouter account.
