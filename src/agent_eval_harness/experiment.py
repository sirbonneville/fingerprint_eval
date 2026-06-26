"""Experiment runner -- the actual product (spec.md piece 5).

Turns the simulator into a causal instrument: hold everything fixed, sweep ONE
axis, run each cell N times (N is your 'n'), and read the *distribution* of
behavior per cell -- never a single trace.

Two disciplines are baked in here on purpose:

1. **Band-relative axes.** Volatility is in band-width units and band widths are
   centered on a fixed anchor, so the volatility axis stays orthogonal to the
   band-width axis (see ``price_path`` for why this matters).

2. **Calibration before grid.** :func:`run_experiment` runs a control pair
   (an easy config the probe should discover, a hard one it should not) as cell
   zero, and -- for a trusted (real) model -- refuses to run the full grid unless
   the probe discovers the easy control more than the hard one. If the easy
   control fails, the instrument is not measuring configs yet, so stop.

CAVEAT: discovery numbers from a stand-in model (HoldModel/RandomTrader/...)
validate plumbing only. A non-strategic trader does not discover prices. Pass
``trust_discovery=True`` only when a real model sits behind the Model protocol.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional

from .dpm import MarketConfig, PriceBucketMarket
from .harness import EpisodeLog, run_episode
from .metrics import Discovery, Footprint, compute_discovery
from .model import RandomTrader
from .price_path import MODEL_ARITHMETIC, generate_path

# A model factory builds a fresh model for (cell, run_index). Fresh-per-run keeps
# stateful stand-ins reproducible (seed by run) and real models independent.
ModelFactory = Callable[["CellConfig", int], object]

SWEEPABLE_AXES = ("volatility", "knowability_min", "band_width")


@dataclass
class CellConfig:
    """One experimental cell: the three swept axes + the frozen environment."""

    # swept axes
    volatility: float       # band-widths per sqrt-hour (band-relative)
    knowability_min: float  # post-close dead-window length in minutes
    band_width: float       # dollar width of interior bands

    # frozen environment (identical across a sweep)
    anchor: float = 100.0
    n_outcomes: int = 3
    hours: float = 2.0
    interval_min: float = 30.0
    drift: float = 0.0
    k: float = 10.0
    fee: float = 0.02
    initial_shares: float = 100.0
    starting_cash: float = 1000.0
    price_model: str = MODEL_ARITHMETIC
    settlement_rule_text: str = (
        "Settles to the band containing the asset's price at the settlement reference."
    )
    label: str = ""

    def band_edges(self) -> List[float]:
        """Evenly spaced interior edges of width ``band_width``, centered on ``anchor``."""
        n_edges = self.n_outcomes - 1
        return [
            self.anchor + (i - (n_edges - 1) / 2.0) * self.band_width
            for i in range(n_edges)
        ]

    def outcome_labels(self) -> List[str]:
        return ["band %d" % i for i in range(self.n_outcomes)]

    def build_market(self) -> PriceBucketMarket:
        config = MarketConfig(
            k=self.k,
            fee=self.fee,
            band_edges=self.band_edges(),
            outcome_labels=self.outcome_labels(),
            settlement_rule_text=self.settlement_rule_text,
            trading_close_min=self.hours * 60.0,
        )
        return PriceBucketMarket(config, [self.initial_shares] * self.n_outcomes)

    def build_path(self, seed: int):
        return generate_path(
            anchor=self.anchor,
            hours=self.hours,
            interval_min=self.interval_min,
            volatility=self.volatility,
            drift=self.drift,
            band_width=self.band_width,
            post_close_min=self.knowability_min,
            seed=seed,
            model=self.price_model,
        )


@dataclass
class Distribution:
    """Summary of a metric across the N runs of a cell (the thing you actually read)."""

    n: int
    median: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    iqr: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "p25": self.p25,
            "p75": self.p75,
            "iqr": self.iqr,
            "min": self.minimum,
            "max": self.maximum,
        }


def summarize(values) -> Distribution:
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return Distribution(0)
    if n == 1:
        v = float(vals[0])
        return Distribution(1, v, v, 0.0, v, v, 0.0, v, v)
    q = statistics.quantiles(vals, n=4)  # [p25, p50, p75]
    return Distribution(
        n=n,
        median=statistics.median(vals),
        mean=statistics.fmean(vals),
        std=statistics.pstdev(vals),
        p25=q[0],
        p75=q[2],
        iqr=q[2] - q[0],
        minimum=min(vals),
        maximum=max(vals),
    )


@dataclass
class RunRecord:
    run: int
    footprint: Footprint
    discovery: Discovery
    winning_outcome: int
    pnl: float
    log: Optional[EpisodeLog] = None


# The behavior + discovery vocabulary aggregated per cell. Same names everywhere
# so cells are comparable (spec.md piece 5).
_METRIC_EXTRACTORS = {
    "discovery_prob_on_winner": lambda r: r.discovery.prob_on_winner,
    "discovery_brier": lambda r: r.discovery.brier,
    "discovery_edge_over_open": lambda r: r.discovery.edge_over_open,
    "trade_count": lambda r: r.footprint.trade_count,
    "total_volume": lambda r: r.footprint.total_volume,
    "first_trade_frac": lambda r: r.footprint.first_trade_frac,
    "direction_switches": lambda r: r.footprint.direction_switches,
    "outcomes_traded": lambda r: r.footprint.outcomes_traded,
    "median_trade_size": lambda r: r.footprint.median_trade_size,
    "n_rejected": lambda r: r.footprint.n_rejected,
    "n_parse_failures": lambda r: r.footprint.n_parse_failures,
    "pnl": lambda r: r.pnl,
    "return_pct": lambda r: r.footprint.return_pct,
    "win": lambda r: 1.0 if r.footprint.win else 0.0,
}


@dataclass
class CellResult:
    cell: CellConfig
    n_runs: int
    records: List[RunRecord]
    metrics: Dict[str, Distribution]

    def median(self, metric: str) -> Optional[float]:
        return self.metrics[metric].median

    def spread(self, metric: str) -> Optional[float]:
        return self.metrics[metric].iqr


def run_cell(
    cell: CellConfig,
    model_factory: ModelFactory,
    n_runs: int = 10,
    store_logs: bool = False,
) -> CellResult:
    """Run one cell N times (seed = run index) and summarize the distribution."""
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    records: List[RunRecord] = []
    for run in range(n_runs):
        market = cell.build_market()
        path = cell.build_path(seed=run)
        model = model_factory(cell, run)
        log = run_episode(market, model, path, starting_cash=cell.starting_cash, seed=run)
        disc = compute_discovery(
            log.closing_probabilities, log.winning_outcome, log.opening_probabilities
        )
        records.append(
            RunRecord(
                run=run,
                footprint=log.footprint,
                discovery=disc,
                winning_outcome=log.winning_outcome,
                pnl=log.pnl,
                log=log if store_logs else None,
            )
        )
    metrics = {
        name: summarize([extract(r) for r in records])
        for name, extract in _METRIC_EXTRACTORS.items()
    }
    return CellResult(cell=cell, n_runs=n_runs, records=records, metrics=metrics)


@dataclass
class GridResult:
    axis: str
    cells: "Dict[object, CellResult]"  # swept value -> result

    def deltas_vs_spread(self, metric: str) -> dict:
        """Piece-6 seed: per-cell median + spread, and whether the between-cell
        range exceeds the typical within-cell spread (signal vs noise)."""
        medians = {v: c.median(metric) for v, c in self.cells.items()}
        spreads = {v: c.spread(metric) for v, c in self.cells.items()}
        present = [m for m in medians.values() if m is not None]
        between = (max(present) - min(present)) if len(present) >= 2 else 0.0
        within = [s for s in spreads.values() if s is not None]
        typical_within = statistics.median(within) if within else 0.0
        return {
            "metric": metric,
            "medians": medians,
            "within_cell_iqr": spreads,
            "between_cell_range": between,
            "typical_within_cell_iqr": typical_within,
            "looks_like_signal": between > typical_within,
        }


def run_grid(
    base: CellConfig,
    axis: str,
    values: List[float],
    model_factory: ModelFactory,
    n_runs: int = 10,
    store_logs: bool = False,
) -> GridResult:
    """Sweep ONE axis across ``values``, freezing the rest (spec.md piece 5)."""
    if axis not in SWEEPABLE_AXES:
        raise ValueError("axis must be one of %r" % (SWEEPABLE_AXES,))
    cells: Dict[object, CellResult] = {}
    for v in values:
        cell = replace(base, label="%s=%s" % (axis, v), **{axis: v})
        cells[v] = run_cell(cell, model_factory, n_runs=n_runs, store_logs=store_logs)
    return GridResult(axis=axis, cells=cells)


# --------------------------------------------------------------- calibration


def control_pair(
    base: CellConfig,
    easy_volatility: float = 0.1,
    hard_volatility: float = 2.0,
    hard_knowability_min: Optional[float] = None,
) -> "tuple[CellConfig, CellConfig]":
    """Build the (easy, hard) calibration control pair.

    easy: low volatility + zero dead window -> the price at close essentially
    determines the outcome, so a competent probe should discover it.
    hard: high volatility + a long dead window -> the outcome is essentially
    unknowable at close, so even a competent probe should not discover it.
    """
    if hard_knowability_min is None:
        hard_knowability_min = base.hours * 60.0  # dead window as long as trading
    easy = replace(base, volatility=easy_volatility, knowability_min=0.0, label="control-easy")
    hard = replace(
        base, volatility=hard_volatility, knowability_min=hard_knowability_min, label="control-hard"
    )
    return easy, hard


@dataclass
class CalibrationResult:
    easy: CellResult
    hard: CellResult
    easy_discovery: float
    hard_discovery: float
    margin: float
    passed: bool
    trusted: bool
    note: str


def run_calibration(
    base: CellConfig,
    model_factory: ModelFactory,
    n_runs: int = 10,
    margin: float = 0.05,
    trust_discovery: bool = False,
    metric: str = "discovery_prob_on_winner",
) -> CalibrationResult:
    """Run the control pair and judge whether the probe discovers easy >> hard."""
    easy_cell, hard_cell = control_pair(base)
    easy = run_cell(easy_cell, model_factory, n_runs=n_runs)
    hard = run_cell(hard_cell, model_factory, n_runs=n_runs)
    easy_disc = easy.median(metric) or 0.0
    hard_disc = hard.median(metric) or 0.0
    passed = easy_disc > hard_disc + margin
    if not trust_discovery:
        note = (
            "PLUMBING ONLY: discovery is not meaningful with a stand-in model. "
            "The control pair ran mechanically; do not read the verdict as real "
            "until a genuine model is behind the Model protocol."
        )
    elif passed:
        note = "Calibration passed: probe discovers the easy control more than the hard one."
    else:
        note = (
            "Calibration FAILED: probe did not discover the easy control. The "
            "instrument is not measuring configs yet -- stop and fix the probe."
        )
    return CalibrationResult(
        easy=easy,
        hard=hard,
        easy_discovery=easy_disc,
        hard_discovery=hard_disc,
        margin=margin,
        passed=passed,
        trusted=trust_discovery,
        note=note,
    )


@dataclass
class ExperimentResult:
    calibration: CalibrationResult
    grid: Optional[GridResult]
    ran_grid: bool
    message: str


def run_experiment(
    base: CellConfig,
    axis: str,
    values: List[float],
    model_factory: Optional[ModelFactory] = None,
    n_runs: int = 10,
    trust_discovery: bool = False,
    require_calibration: bool = True,
    store_logs: bool = False,
) -> ExperimentResult:
    """Calibrate first (cell zero), then sweep -- gating the grid on the control.

    For a trusted (real) model, the full grid runs only if the easy control is
    discovered (``calibration.passed``). For a stand-in model the grid still runs
    for plumbing validation, but discovery output must not be trusted.
    """
    if model_factory is None:
        model_factory = default_stand_in_factory

    calibration = run_calibration(
        base, model_factory, n_runs=n_runs, trust_discovery=trust_discovery
    )

    if trust_discovery and require_calibration and not calibration.passed:
        return ExperimentResult(
            calibration=calibration,
            grid=None,
            ran_grid=False,
            message=(
                "Grid skipped: %s" % calibration.note
            ),
        )

    grid = run_grid(base, axis, values, model_factory, n_runs=n_runs, store_logs=store_logs)
    message = (
        "Grid ran for plumbing validation only (stand-in model)."
        if not trust_discovery
        else "Grid ran after passing calibration."
    )
    return ExperimentResult(
        calibration=calibration, grid=grid, ran_grid=True, message=message
    )


def default_stand_in_factory(cell: CellConfig, run: int):
    """A seeded RandomTrader -- exercises the loop; does NOT discover prices."""
    return RandomTrader(n_outcomes=cell.n_outcomes, seed=run)
