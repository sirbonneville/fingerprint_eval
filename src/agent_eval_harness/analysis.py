"""Analysis layer -- the honesty layer (spec.md piece 6).

This is where the instrument earns the word "causal" or quietly fakes it. Its job
is to be **honest about uncertainty, not to manufacture significance.** With N~10
runs per cell, reaching for p-values and significance stars would dress up a tiny
sample as authoritative -- the exact failure mode this project exists to avoid.

So the headline read is deliberately plain:

    effect_over_spread = |between-cell delta in medians| / (typical within-cell IQR)

If that ratio is below ~1, the between-cell delta is smaller than the run-to-run
noise and there is no detectable effect, full stop -- no test can rescue that. The
single most honest output is the **overlap visualization**: the two cells' N-run
distributions overlaid on a shared axis. Heavy overlap => no effect; clean
separation => you don't need a test to see it.

Design commitments (extends ``experiment.deltas_vs_spread``; does not replace it
with a significance framework):
- Spread is reported as IQR and min/max (robust, skew-revealing), never a lone std.
- CLES (common-language effect size, P(b>a)) is the overlap number -- an effect
  size, not a test.
- Any significance test is opt-in, secondary, and caveated as underpowered. Off
  by default. (:func:`mann_whitney`.)
- Hard "too thin to read" gates mirror the project's n<3 discipline.
- Until a real model is behind the Model protocol, EVERY number here is plumbing
  validation only and is stamped as such.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .experiment import METRIC_NAMES, GridResult

# Mirrors the project's run-discipline: below this, a cell is not trustworthy at
# all; below the recommended N it is provisional.
MIN_TRUSTWORTHY_N = 3
RECOMMENDED_N = 10

# Headline thresholds for the plain effect-vs-spread read (intentionally blunt).
EFFECT_RATIO_DETECTABLE = 1.0   # delta must exceed the typical within-cell IQR
CLES_SEPARATION = 0.8           # |CLES - 0.5| this far out => cleanly separated

# Curated headline metrics for reports (you can analyze any METRIC_NAMES).
DEFAULT_REPORT_METRICS = (
    "discovery_prob_on_winner",
    "discovery_brier",
    "trade_count",
    "total_volume",
    "first_trade_frac",
    "direction_switches",
    "outcomes_traded",
    "pnl",
    "win",
)

PLUMBING_BANNER = (
    "PLUMBING ONLY -- stand-in model. No number below is a finding; this validates "
    "the analysis machinery, not discovery. Trust nothing until a real model is "
    "behind the Model protocol."
)


# ----------------------------------------------------------------- effect-size primitives


def cles(a: Sequence[float], b: Sequence[float]) -> float:
    """Common-language effect size: P(b > a), ties counted as 0.5.

    0.5 == no effect (full overlap); ->1 or ->0 == clean separation. This is an
    effect size (probability of superiority), not a hypothesis test.
    """
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if not a or not b:
        return 0.5
    wins = 0.0
    for x in a:
        for y in b:
            if y > x:
                wins += 1.0
            elif y == x:
                wins += 0.5
    return wins / (len(a) * len(b))


def range_overlap_fraction(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of the combined [min,max] span where the two ranges overlap."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if not a or not b:
        return 1.0
    lo = max(min(a), min(b))
    hi = min(max(a), max(b))
    overlap = max(0.0, hi - lo)
    span = max(max(a), max(b)) - min(min(a), min(b))
    if span <= 0.0:
        return 1.0  # all values identical -> total overlap
    return overlap / span


def _iqr(values: Sequence[float]) -> float:
    vals = sorted(v for v in values if v is not None)
    if len(vals) < 2:
        return 0.0
    q = statistics.quantiles(vals, n=4)
    return q[2] - q[0]


def _median(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


# ----------------------------------------------------------------- pairwise effect


# Verdicts -- deliberately plain language, never "significant".
V_TOO_THIN = "TOO THIN"
V_NO_VARIATION = "NO VARIATION"
V_NO_EFFECT = "NO DETECTABLE EFFECT"
V_AMBIGUOUS = "WEAK / AMBIGUOUS"
V_POSSIBLE = "POSSIBLE EFFECT"


@dataclass
class PairwiseEffect:
    metric: str
    value_a: object
    value_b: object
    n_a: int
    n_b: int
    median_a: Optional[float]
    median_b: Optional[float]
    iqr_a: float
    iqr_b: float
    range_a: "tuple"
    range_b: "tuple"
    delta_median: Optional[float]
    typical_within_iqr: float
    effect_over_spread: Optional[float]   # the headline: |delta| / typical IQR
    cles: float
    range_overlap: float
    verdict: str
    notes: List[str] = field(default_factory=list)


def pairwise_effect(
    values_a: Sequence[float],
    values_b: Sequence[float],
    metric: str,
    value_a: object,
    value_b: object,
    trusted: bool = False,
) -> PairwiseEffect:
    """Compare two cells for one metric, foregrounding effect-vs-spread + overlap."""
    a = [v for v in values_a if v is not None]
    b = [v for v in values_b if v is not None]
    n_a, n_b = len(a), len(b)
    med_a, med_b = _median(a), _median(b)
    iqr_a, iqr_b = _iqr(a), _iqr(b)
    rng_a = (min(a), max(a)) if a else (None, None)
    rng_b = (min(b), max(b)) if b else (None, None)

    delta = None if (med_a is None or med_b is None) else (med_b - med_a)
    typical_iqr = statistics.median([iqr_a, iqr_b])
    effect_ratio = None
    if delta is not None and typical_iqr > 0.0:
        effect_ratio = abs(delta) / typical_iqr

    c = cles(a, b)
    overlap = range_overlap_fraction(a, b)

    notes: List[str] = []
    if not trusted:
        notes.append(PLUMBING_BANNER)

    # Gate 1: too thin to read at all.
    if n_a < MIN_TRUSTWORTHY_N or n_b < MIN_TRUSTWORTHY_N:
        notes.append(
            "N too small to trust (n_a=%d, n_b=%d; floor is %d). Mirrors the "
            "project's n<3 discipline." % (n_a, n_b, MIN_TRUSTWORTHY_N)
        )
        verdict = V_TOO_THIN
    elif delta is None:
        notes.append("a cell has no usable values for this metric.")
        verdict = V_TOO_THIN
    elif typical_iqr == 0.0:
        # No within-cell scatter at all (e.g. a stand-in that ignores the market).
        if delta == 0.0:
            notes.append("Both cells are identical with zero spread -- no variation to read.")
            verdict = V_NO_VARIATION
        else:
            notes.append(
                "Zero within-cell spread but a nonzero delta -- suspicious at small N; "
                "verify this is real and not a degenerate stand-in artifact."
            )
            verdict = V_AMBIGUOUS
    else:
        # The honest read: delta in units of run-to-run noise + overlap.
        separated = abs(c - 0.5) >= (CLES_SEPARATION - 0.5)
        if effect_ratio is not None and effect_ratio < EFFECT_RATIO_DETECTABLE:
            notes.append(
                "Spread exceeds delta: between-cell delta %.4g is within the typical "
                "within-cell IQR %.4g (ratio %.2f). No detectable effect at this N."
                % (delta, typical_iqr, effect_ratio)
            )
            verdict = V_NO_EFFECT
        elif effect_ratio is not None and effect_ratio >= EFFECT_RATIO_DETECTABLE and separated:
            verdict = V_POSSIBLE
            notes.append(
                "Delta (%.4g) exceeds the typical within-cell IQR (%.4g; ratio %.2f) "
                "and the distributions are largely separated (CLES %.2f). Possible "
                "effect -- judge the overlap plot; not a significance claim."
                % (delta, typical_iqr, effect_ratio, c)
            )
        else:
            verdict = V_AMBIGUOUS
            notes.append(
                "Delta is comparable to the spread (ratio %.2f, CLES %.2f). Ambiguous "
                "at this N -- look at the overlap, gather more runs before concluding."
                % (effect_ratio if effect_ratio is not None else float("nan"), c)
            )

    if min(n_a, n_b) < RECOMMENDED_N and verdict != V_TOO_THIN:
        notes.append("Below recommended N=%d -- treat as provisional." % RECOMMENDED_N)

    return PairwiseEffect(
        metric=metric,
        value_a=value_a,
        value_b=value_b,
        n_a=n_a,
        n_b=n_b,
        median_a=med_a,
        median_b=med_b,
        iqr_a=iqr_a,
        iqr_b=iqr_b,
        range_a=rng_a,
        range_b=rng_b,
        delta_median=delta,
        typical_within_iqr=typical_iqr,
        effect_over_spread=effect_ratio,
        cles=c,
        range_overlap=overlap,
        verdict=verdict,
        notes=notes,
    )


# ----------------------------------------------------------------- overlap visualization


def overlap_plot(
    values_a: Sequence[float],
    values_b: Sequence[float],
    label_a: str,
    label_b: str,
    width: int = 51,
) -> str:
    """ASCII strip plot of two N-run distributions on a shared axis.

    Each column is a bin; the digit is the count of runs landing there (capped at
    9), '|' marks the median. Read overlap by comparing the two rows column by
    column -- heavy column-overlap means no effect.
    """
    a = [v for v in values_a if v is not None]
    b = [v for v in values_b if v is not None]
    if not a or not b:
        return "(no data to plot)"

    gmin = min(min(a), min(b))
    gmax = max(max(a), max(b))
    if gmax <= gmin:
        return (
            "all values identical at %.4g (no variation to visualize)\n"
            "  %s: n=%d\n  %s: n=%d" % (gmin, label_a, len(a), label_b, len(b))
        )

    def col(x: float) -> int:
        frac = (x - gmin) / (gmax - gmin)
        return min(width - 1, max(0, int(round(frac * (width - 1)))))

    def strip(vals: Sequence[float]) -> str:
        counts = [0] * width
        for x in vals:
            counts[col(x)] += 1
        cells = [(str(min(9, c)) if c > 0 else ".") for c in counts]
        med_col = col(statistics.median(vals))
        marker = cells[med_col]
        cells[med_col] = "|" if marker == "." else marker  # keep count if collision
        return "".join(cells)

    lbl_w = max(len(label_a), len(label_b))
    axis = "%s  %.4g%s%.4g" % (" " * lbl_w, gmin, " " * max(1, width - 12), gmax)
    return (
        "%s: %s\n%s: %s\n%s"
        % (
            label_a.ljust(lbl_w),
            strip(a),
            label_b.ljust(lbl_w),
            strip(b),
            axis,
        )
    )


# ----------------------------------------------------------------- significance (opt-in, secondary)


@dataclass
class MannWhitney:
    u: float
    p_two_sided_approx: float
    caveat: str


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_whitney(a: Sequence[float], b: Sequence[float]) -> MannWhitney:
    """Mann-Whitney U with a normal-approx p. SECONDARY and underpowered at small N.

    Provided only because someone will ask for it. Do not let it drive the
    conclusion -- the effect-vs-spread read and the overlap plot do that. The
    p-value here is a fragile normal approximation and should be read as "weak
    corroboration at best," never as a significance verdict.
    """
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return MannWhitney(float("nan"), float("nan"), "no data")
    u_b = cles(a, b) * n_a * n_b  # U for sample b
    mu = n_a * n_b / 2.0
    sigma = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12.0)
    if sigma == 0.0:
        p = 1.0
    else:
        z = (u_b - mu) / sigma
        p = 2.0 * (1.0 - _phi(abs(z)))
    caveat = (
        "SECONDARY + UNDERPOWERED at N=%d/%d: normal-approx p, no exact distribution "
        "or multiple-comparison correction. Defer to effect-vs-spread and the overlap "
        "plot; this cannot rescue heavily overlapping distributions." % (n_a, n_b)
    )
    return MannWhitney(u=u_b, p_two_sided_approx=p, caveat=caveat)


# ----------------------------------------------------------------- axis / grid analysis


@dataclass
class AxisMetricAnalysis:
    axis: str
    metric: str
    baseline_value: object
    per_cell: Dict[object, dict]          # value -> {n, median, iqr, min, max}
    effects: List[PairwiseEffect]         # each non-baseline cell vs baseline
    headline: Optional[PairwiseEffect]    # the largest effect-over-spread
    trusted: bool


def analyze_axis_metric(
    grid: GridResult,
    metric: str,
    trusted: bool = False,
    baseline: object = None,
) -> AxisMetricAnalysis:
    """Compare every cell to a baseline cell for one metric (extends deltas_vs_spread)."""
    values = list(grid.cells.keys())
    if not values:
        raise ValueError("grid has no cells")
    if baseline is None:
        baseline = values[0]
    if baseline not in grid.cells:
        raise ValueError("baseline %r not in grid" % baseline)

    per_cell: Dict[object, dict] = {}
    for v, cell_result in grid.cells.items():
        vals = [x for x in cell_result.metric_values(metric) if x is not None]
        per_cell[v] = {
            "n": len(vals),
            "median": _median(vals),
            "iqr": _iqr(vals),
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }

    base_vals = grid.cells[baseline].metric_values(metric)
    effects: List[PairwiseEffect] = []
    for v, cell_result in grid.cells.items():
        if v == baseline:
            continue
        effects.append(
            pairwise_effect(
                base_vals, cell_result.metric_values(metric),
                metric=metric, value_a=baseline, value_b=v, trusted=trusted,
            )
        )

    headline = None
    for e in effects:
        if e.effect_over_spread is None:
            continue
        if headline is None or (
            headline.effect_over_spread is None
            or e.effect_over_spread > headline.effect_over_spread
        ):
            headline = e

    return AxisMetricAnalysis(
        axis=grid.axis,
        metric=metric,
        baseline_value=baseline,
        per_cell=per_cell,
        effects=effects,
        headline=headline,
        trusted=trusted,
    )


def analyze_grid(
    grid: GridResult,
    metrics: Optional[Sequence[str]] = None,
    trusted: bool = False,
    baseline: object = None,
) -> Dict[str, AxisMetricAnalysis]:
    metrics = metrics or DEFAULT_REPORT_METRICS
    for m in metrics:
        if m not in METRIC_NAMES:
            raise ValueError("unknown metric %r" % m)
    return {m: analyze_axis_metric(grid, m, trusted=trusted, baseline=baseline) for m in metrics}


# ----------------------------------------------------------------- text report


def format_report(
    grid: GridResult,
    metrics: Optional[Sequence[str]] = None,
    trusted: bool = False,
    baseline: object = None,
    show_plots: bool = True,
) -> str:
    """Human-readable report foregrounding effect-vs-spread and overlap."""
    analyses = analyze_grid(grid, metrics=metrics, trusted=trusted, baseline=baseline)
    lines: List[str] = []
    if not trusted:
        lines.append("=" * 78)
        lines.append(PLUMBING_BANNER)
        lines.append("=" * 78)
    lines.append("AXIS SWEPT: %s   (baseline = first cell unless noted)" % grid.axis)
    lines.append("")

    for metric, am in analyses.items():
        lines.append("-" * 78)
        lines.append("METRIC: %s" % metric)
        lines.append("  per-cell  (median [IQR]  range[min,max]  n):")
        for v, s in am.per_cell.items():
            lines.append(
                "    %-14s median=%s  IQR=%.4g  range=[%s, %s]  n=%d"
                % (
                    "%s=%s" % (grid.axis, v),
                    _fmt(s["median"]),
                    s["iqr"],
                    _fmt(s["min"]),
                    _fmt(s["max"]),
                    s["n"],
                )
            )
        for e in am.effects:
            lines.append("")
            lines.append(
                "  %s=%s vs %s=%s  ->  delta(median)=%s, effect/spread=%s, CLES=%.2f  =>  %s"
                % (
                    grid.axis, e.value_a, grid.axis, e.value_b,
                    _fmt(e.delta_median),
                    ("%.2f" % e.effect_over_spread) if e.effect_over_spread is not None else "n/a",
                    e.cles,
                    e.verdict,
                )
            )
            for note in e.notes:
                if note == PLUMBING_BANNER:
                    continue
                lines.append("      - %s" % note)
            if show_plots:
                plot = overlap_plot(
                    grid.cells[e.value_a].metric_values(metric),
                    grid.cells[e.value_b].metric_values(metric),
                    label_a="%s=%s" % (grid.axis, e.value_a),
                    label_b="%s=%s" % (grid.axis, e.value_b),
                )
                lines.extend("      " + ln for ln in plot.splitlines())
        lines.append("")
    return "\n".join(lines)


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return "%.4g" % x
