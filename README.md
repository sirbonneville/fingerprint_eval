# fingerprint_eval

<img width="1536" height="768" alt="image" src="https://github.com/user-attachments/assets/cb06cde0-2f72-4a72-88f0-6ad2ac6a55f1" />


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
cd fingerprint_eval
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
python3 -m fingerprint_eval --ping
```

*(After `pip install -e ".[dev]"`, `PYTHONPATH=src` is not required.)*

### Single-axis sweep

```bash
python3 -m fingerprint_eval \
  --model openai/gpt-4o \
  --axis volatility \
  --values 0.5,1.0,1.5,2.0 \
  --n 10
```

### Full battery (resumable)

```bash
PYTHONPATH=src python3 -m fingerprint_eval.battery \
  --models openai/gpt-4o,anthropic/claude-sonnet-4 \
  --label official-v1 \
  --memory both \
  --n 30 \
  --concurrency 4
```

**Live-board battery (v2)** — same seeds and prompts; synthetic traders move the board
between turns:

```bash
PYTHONPATH=src python3 -m fingerprint_eval.battery \
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
PYTHONPATH=src python3 -m fingerprint_eval.battery \
  --dry-run --models offline-stand-in --label smoke --n 2
```

Resume a crashed battery into the same session folder:

```bash
PYTHONPATH=src python3 -m fingerprint_eval.battery \
  --resume eval_runs/battery_WITH_SYNTH_TRADING/2026-06-29_official-v2 \
  --models openai/gpt-4o,anthropic/claude-sonnet-4 \
  --synthetic-traders --memory both --n 30
```

Add `--plain` for CI/log-friendly output (no color or Rich panels). Respect `NO_COLOR=1`.

### Branded terminal UI (Gensyn dashboard-dark)

Battery progress uses a TUI skin derived from the [Gensyn brand manifest](https://brand.gensyn.ai/brand/manifest.json) (`dashboard-dark`: Deep Brown `#230800`, Gensyn Pink `#fad7d1`, retro stat panels). Theme tokens live in `src/fingerprint_eval/brand_data/dashboard-dark.json`.

```bash
pip install -e ".[cli]"   # optional: Rich panels (falls back to ANSI, then plain)
python3 -m fingerprint_eval.battery --dry-run --models offline-stand-in --label smoke --n 2
python3 -m fingerprint_eval.battery --plain --dry-run ...   # CI / logs
```

Refresh the bundled theme from the live manifest:

```bash
PYTHONPATH=src python3 eval_runs/refresh_brand_theme.py --write
```

---

## Recompute the findings from raw data

**This is the payoff.** Every completed battery cell writes full episode logs to
`runs.jsonl`. The analysis scripts read those logs only — **no model calls, no API
cost, no writes** — and **recompute from scratch** the statistics, permutation
p-values, and cross-tables that the findings documents are built on.

If you have the battery data on disk, you can reproduce (or audit) the narrative in
[`battery_docs/`](battery_docs/) and [`summary_docs/`](summary_docs/) yourself. Point
the same script at **any session folder** (v1 frozen board, v2 live board, or a future
battery) and it grinds through all 52 cells × 30 episodes.

### `master_analysis.py` — the main report

The primary digest. Aggregates every cell in a session and prints six sections in
order (this is what you see scrolling in the terminal):

| Section | What it recomputes |
|---|---|
| **A. PERSONALITY** | Pooled behavioral fingerprint per model × memory (trade count, hold rate, volume, sizing, …) across all 13 core axis cells |
| **B. NOISE FLOOR** | Spread across four **config-identical** draws (baseline = vol 1.0 = know 0 = temp 0.7). Any claimed axis or memory effect must beat this spread |
| **C. MEMORY EFFECT** | mem-on vs mem-off, pooled; gap and two-sided permutation *p* (20k shuffles). `<== sig` when *p* < 0.05 |
| **D. AXIS SWEEPS** | Each knob (volatility, temperature, knowability, band width): metric series + endpoint permutation test per model × memory |
| **E. DISCOVERY** | GPT vs Claude on `top_pick`, `peak_top_pick`, `buy_flow` — the discovery story lives here |
| **F. WIN DECOMPOSITION** | Splits win into commitment (`P(held)`) vs conditional accuracy (`P(win\|held)`) |

**v1 (frozen board)** — default session; same data as `findings_without_synthetic_trading.md`:

```bash
PYTHONPATH=src python3 eval_runs/master_analysis.py \
  battery_WITHOUT_SYNTH_TRADING/2026-06-27_official-v1
```

**v2 (live board)** — same structure, different arm; compare to `findings_with_synth_trading.md`:

```bash
PYTHONPATH=src python3 eval_runs/master_analysis.py \
  battery_WITH_SYNTH_TRADING/2026-06-29_official-v2
```

Session paths are **relative to `eval_runs/`** (where the script lives). Omit the
argument to use the v1 default.

```bash
PYTHONPATH=src python3 eval_runs/master_analysis.py          # v1 default
PYTHONPATH=src python3 eval_runs/master_analysis.py --plain  # no branded header
```

**Runtime:** Section D runs ~96 permutation tests at 20,000 shuffles each, so output
appears in chunks with pauses — that is normal CPU work, not network I/O. Sections A
and B print quickly; C, D, and E are the slow parts.

**Data source:** each line in `runs.jsonl` is one full episode (every turn, footprint,
positions). The script never reads pre-aggregated `metrics.json` for its core stats —
it recomputes from episodes so you are looking at the same ground truth the write-ups
used.

---

## Other analysis commands (read-only)

All scripts live under `eval_runs/`, take a session path where noted, and consume
completed battery output only.

| Script | What it does | Example |
|---|---|---|
| `master_analysis.py` | **Full six-section report** (above) — start here | `python3 eval_runs/master_analysis.py battery_WITHOUT_SYNTH_TRADING/2026-06-27_official-v1` |
| `knob_diagnostics.py` | Knowability ceiling, top_pick scorability, volatility within-label adjudication | `python3 eval_runs/knob_diagnostics.py battery_WITHOUT_SYNTH_TRADING/2026-06-27_official-v1 off` |
| `within_label_check.py` | Response to realized price range within a fixed band label | `python3 eval_runs/within_label_check.py battery_WITHOUT_SYNTH_TRADING/2026-06-27_official-v1 openai-gpt-4o off` |
| `board_reaction.py` | Does the model track live board drift? (**v2 only**) | `python3 eval_runs/board_reaction.py battery_WITH_SYNTH_TRADING/2026-06-29_official-v2` |
| `allin_check.py` | All-in attempt rates v1 vs v2 (hardcoded session pair; pass model/mem) | `python3 eval_runs/allin_check.py openai-gpt-4o off 0.9` |
| `depth_table.py` | Market liquidity / conviction-unit tables | `python3 eval_runs/depth_table.py` |
| `calibration_gate.py` | Synthetic-flow sanity checks **before** launching v2 (offline stand-in) | `python3 eval_runs/calibration_gate.py` |
| `_audit.py` | Independent Phase 1 reducer — recomputes metrics without importing other analysis scripts | `python3 eval_runs/_audit.py` |
| `_phase2.py` | Phase 2 cross-battery deliverables (matched-seed v1↔v2) | `python3 eval_runs/_phase2.py` |
| `_review_naive_baseline.py` | Naive price-follower baseline + entry-timing check (backs the 2026-07-08 addendum) | `python3 eval_runs/_review_naive_baseline.py` |
| `refresh_brand_theme.py` | Update bundled Gensyn dashboard-dark theme from manifest | `python3 eval_runs/refresh_brand_theme.py --write` |

The `_audit.py` / `_phase2.py` pair is what backs
[`summary_docs/battery_comparison_v1_vs_v2.md`](summary_docs/battery_comparison_v1_vs_v2.md)
— deliberately **does not import** `master_analysis` so agreement between the two
pipelines is a real cross-check, not a shared bug.

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
fingerprint_eval/
├── src/fingerprint_eval/     # Core package
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
| [`battery_docs/findings_without_synthetic_trading.md`](battery_docs/findings_without_synthetic_trading.md) | v1 (frozen board) full findings — **recomputable via `master_analysis.py` on the v1 session** |
| [`battery_docs/findings_with_synth_trading.md`](battery_docs/findings_with_synth_trading.md) | v2 (live board) full findings — **recomputable via `master_analysis.py` on the v2 session** |
| [`architectural_specs/spec.md`](architectural_specs/spec.md) | Original build spec and design principles |
| [`architectural_specs/dpm_spec.md`](architectural_specs/dpm_spec.md) | DPM cost-function derivation and verification |
| [`architectural_specs/v3_decorrelated_flow_spec.md`](architectural_specs/v3_decorrelated_flow_spec.md) | Proposed v3 battery (decorrelated synthetic flow) |
| [`summary_docs/discoveries_versus_verified_literature.md`](summary_docs/discoveries_versus_verified_literature.md) | Overlap with published agent-trading benchmarks |

---

## Headline results (very short)

Two batteries, matched seeds, 1,560 episodes each:

- **Two stable personalities** — gpt-4o: low-activity, concentrated; claude-sonnet-4:
  active, diversified. Rank order survives both batteries.
- **Neither model out-picks a naive price-follower** (2026-07-08 revision): against a
  timing-matched "buy the current band" baseline, both models score at baseline in
  both batteries. The robust cross-model trait is timing/attention — claude spreads
  trades across the episode and ignores spurious board tilts; gpt concentrates its
  entry and anchors on them.
- **Memory → commitment, not accuracy** for claude (win decomposition); gpt memory
  touches activity only.
- **The v2 "discovery collapse" is an entry-timing effect** (2026-07-08 revision): on
  a live board gpt front-loads ~84% of its buy notional to t=0 — where no strategy
  can beat chance — anchoring on a pure-noise board tilt the synthetic flow creates
  before its first turn; claude keeps buying at later, informative turns.
- **Frozen-board artifacts:** gpt's $333.33 budget/3 anchor, the orthogonality 2×2,
  and the band_width↔volatility axis hierarchy.

See `summary_docs/battery_comparison_v1_vs_v2.md` for audit tables, p-values, open
questions, and the 2026-07-08 addendum (naive-baseline null + entry-timing
decomposition).

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
