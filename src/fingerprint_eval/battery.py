"""Official evaluation battery: a thin storage/organization layer around the
existing experiment runner.

This module does NOT change the harness, metrics, or the causal-read analysis. It
orchestrates many single-knob cells (baseline + one-knob-at-a-time), runs each
under both memory conditions and across multiple models, and writes every run to
a properly named, reproducible, resumable folder tree:

    eval_runs/<ISO-timestamp>_<label>/
        manifest.json                      # one index of the whole battery
        calibration__<model>__mem-<x>.json # control-pair read per model+memory
        <model>__mem-<x>__axis-<axis>__val-<value>__n<N>/
            config.json    # full CellConfig + frozen knobs + git hash (reproducible)
            runs.jsonl     # one JSON line per episode (full per-turn log nested)
            metrics.json   # per-cell footprint/discovery distribution (median, IQR, ...)
            summary.txt    # human one-pager: condition, headline medians+spreads, calib

Design notes
------------
* A "cell" here is a single condition (one knob at one value, or baseline), run N
  times -- not the multi-value grid that ``run_grid`` builds. We drive ``run_cell``
  directly via ``dataclasses.replace`` so the battery can vary ANY CellConfig field
  (including ``n_outcomes``/``band_width``/``memory`` that the --axis sweep can't).
* Calibration is a property of (model, memory, base cell), so it runs ONCE per
  (model, memory) and is recorded -- never used to gate (a behavior battery does
  not need the discovery gate; record-only is the honest treatment).
* Resumable: a cell whose folder already has a complete ``runs.jsonl`` (N lines)
  plus ``metrics.json`` is skipped, so a crashed battery resumes where it stopped.
* Crash-safe manifest: it is rewritten after every completed cell, so a cell that
  finished is indexed even if a later one fails.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from .experiment import (
    CellConfig,
    METRIC_NAMES,
    run_calibration,
    run_cell,
)
from .market_template import money_band_market
from .model import openrouter_factory
from .experiment import default_stand_in_factory

# Behavior metrics surfaced in the human summary + manifest headline (the
# un-confounded footprint the instrument is built to read).
HEADLINE_METRICS = (
    "trade_count",
    "hold_rate",
    "total_volume",
    "median_trade_size",
    "first_trade_frac",
    "direction_switches",
    "outcomes_traded",
    "n_rejected",
)


# --------------------------------------------------------------- battery spec

# The CORE battery: baseline + the axes that intervene on a FIXED market and are
# either confirmed or earned-live. One knob at a time, in descending order of how
# strongly it is expected/known to move BEHAVIOR. Each entry is (CellConfig field,
# [values]); "baseline" is the frozen base cell with nothing changed.
#
# NOTE on knowability: it is a LIVE behavioral axis only because the prompt now
# discloses the settlement schedule (CellConfig.disclose_dead_window=True), so the
# model can perceive the post-close dead window the way a real Delphi agent does.
# Sweeping it literally writes "0/60/120 min after close" into the prompt -- i.e.
# you are measuring "told-about-window", which is what a real agent faces. The
# DISCLOSURE preset (disclose_dead_window off/on) is what disambiguates
# "window matters" from "being told about the window matters" -- run it at least
# once alongside the core.
DEFAULT_BATTERY: Tuple[Tuple[str, Tuple[object, ...]], ...] = (
    ("baseline", (None,)),
    # 1) Volatility -- confirmed market-side CENTER mover (engagement).
    ("volatility", (0.5, 1.0, 1.5, 2.0)),
    # 2) Temperature -- confirmed model-side SPREAD mover (sizing dispersion).
    ("temperature", (0.0, 0.7, 1.2)),
    # 3) Knowability (dead window) -- now PERCEPTIBLE; does being locked out for
    #    longer change how the model trades before close? (real Delphi channel).
    ("knowability_min", (0.0, 60.0, 120.0)),
    # 4) Band width -- distinct mechanism from n_outcomes: how often the price
    #    crosses a bucket boundary during trading. Observed: tighter bands -> more,
    #    smaller trades; wider -> fewer, larger. (seed matched across cells.)
    ("band_width", (5.0, 20.0)),
)

# Quarantined studies -- run with their OWN --label so they aren't conflated with
# the core. n_outcomes CHANGES THE MARKET (different stage), not an intervention on
# a fixed one; starting_cash is speculative (an optional follow-up, not core spend);
# disclosure is the on/off cross that disambiguates the knowability read; drift is
# parked (its DISCOVERY ambiguity is structural, not a prompt issue) but its
# behavioral footprint is available here if wanted.
GEOMETRY_BATTERY = (("baseline", (None,)), ("n_outcomes", (2, 5, 7)))
BUDGET_BATTERY = (("baseline", (None,)), ("starting_cash", (250.0, 4000.0)))
DISCLOSURE_BATTERY = (("disclose_dead_window", (False, True)),)
DRIFT_BATTERY = (("baseline", (None,)), ("drift", (0.0, 1.0, 2.0)))

PRESETS = {
    "core": DEFAULT_BATTERY,
    "geometry": GEOMETRY_BATTERY,
    "budget": BUDGET_BATTERY,
    "disclosure": DISCLOSURE_BATTERY,
    "drift": DRIFT_BATTERY,
}


def _slug(text: str) -> str:
    """Filesystem-safe token: slashes/colons/spaces -> dashes; trim repeats."""
    s = re.sub(r"[^A-Za-z0-9._=-]+", "-", str(text))
    return re.sub(r"-{2,}", "-", s).strip("-")


def _fmt_value(value: object) -> str:
    if value is None:
        return "default"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def cell_folder_name(model: str, memory: bool, axis: str, value: object, n: int) -> str:
    return "%s__mem-%s__axis-%s__val-%s__n%d" % (
        _slug(model),
        "on" if memory else "off",
        _slug(axis),
        _slug(_fmt_value(value)),
        n,
    )


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover - git absent / not a repo
        pass
    return None


def _config_dict(cell: CellConfig, meta: dict) -> dict:
    """Full reproducibility record: every CellConfig field + battery metadata."""
    cfg = dataclasses.asdict(cell)
    return {"meta": meta, "cell_config": cfg}


def _config_hash(config: dict) -> str:
    blob = json.dumps(config["cell_config"], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _distribution_payload(metrics: Dict[str, object], records) -> dict:
    """metrics.json body: per-metric distribution + count of undefined runs."""
    from .experiment import _METRIC_EXTRACTORS  # local: avoid leaking the symbol

    payload = {}
    for name in METRIC_NAMES:
        dist = metrics[name]
        raw = [_METRIC_EXTRACTORS[name](r) for r in records]
        undefined = sum(1 for v in raw if v is None)
        payload[name] = {**dist.as_dict(), "undefined_runs": undefined}
    return payload


def _episode_line(run_index: int, seed: int, log) -> str:
    """One JSON line = one full episode (all turns + settlement, nested)."""
    return json.dumps(
        {"run": run_index, "seed": seed, "episode": dataclasses.asdict(log)},
        default=str,
    )


def _write_summary(
    path: str,
    *,
    model: str,
    memory: bool,
    axis: str,
    value: object,
    n: int,
    metrics: Dict[str, object],
    calibration: Optional[dict],
) -> None:
    lines: List[str] = []
    cond = "baseline" if axis == "baseline" else "%s = %s" % (axis, _fmt_value(value))
    lines.append("CONDITION : %s" % cond)
    lines.append("MODEL     : %s" % model)
    lines.append("MEMORY    : %s" % ("on (rationale replayed)" if memory else "off (stateless)"))
    lines.append("RUNS (N)  : %d" % n)
    lines.append("")
    lines.append("HEADLINE BEHAVIOR (median [IQR], min..max, n):")
    for name in HEADLINE_METRICS:
        d = metrics[name]
        if d.median is None:
            lines.append("  %-20s (no defined runs)" % name)
        else:
            lines.append(
                "  %-20s %s [%s]   %s..%s   n=%d"
                % (
                    name,
                    _num(d.median), _num(d.iqr),
                    _num(d.minimum), _num(d.maximum), d.n,
                )
            )
    if calibration is not None:
        lines.append("")
        lines.append(
            "CALIBRATION (record-only): easy=%s hard=%s passed=%s"
            % (
                _num(calibration.get("easy_discovery")),
                _num(calibration.get("hard_discovery")),
                calibration.get("passed"),
            )
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _num(v) -> str:
    if v is None:
        return "n/a"
    try:
        return "%.4g" % float(v)
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------- resumability

def _cell_complete(folder: str, n: int) -> bool:
    runs = os.path.join(folder, "runs.jsonl")
    metrics = os.path.join(folder, "metrics.json")
    if not (os.path.isfile(runs) and os.path.isfile(metrics)):
        return False
    try:
        with open(runs, "r", encoding="utf-8") as fh:
            count = sum(1 for line in fh if line.strip())
        return count >= n
    except OSError:  # pragma: no cover - defensive
        return False


def _load_manifest(path: str) -> dict:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):  # pragma: no cover - defensive
            pass
    return {"cells": []}


def _write_manifest(path: str, manifest: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    os.replace(tmp, path)  # atomic: a crash mid-write can't corrupt the index


def _headline_payload(metrics: Dict[str, object]) -> dict:
    return {
        name: {"median": metrics[name].median, "iqr": metrics[name].iqr}
        for name in HEADLINE_METRICS
    }


# --------------------------------------------------------------- base cell

def build_base_cell(args, memory: bool) -> CellConfig:
    """Construct the frozen base cell from CLI args (mirrors cli.build_base_cell)."""
    if args.mid is not None:
        cell = money_band_market(
            token=args.token,
            mid=args.mid,
            volatility=args.volatility,
            knowability_min=args.knowability,
            lower_factor=args.lower_factor,
            upper_factor=args.upper_factor,
            currency=args.currency,
            question=args.question,
            data_source=args.data_source,
            real_identity=args.real_identity,
            hours=args.hours,
            interval_min=args.interval,
            drift=args.drift,
            k=args.k,
            fee=args.fee,
            starting_cash=args.budget,
            temperature=args.temperature,
        )
        return dataclasses.replace(cell, memory=memory)

    kwargs = dict(
        volatility=args.volatility,
        knowability_min=args.knowability,
        band_width=args.band_width,
        anchor=args.anchor,
        n_outcomes=args.n_outcomes,
        hours=args.hours,
        interval_min=args.interval,
        drift=args.drift,
        k=args.k,
        fee=args.fee,
        starting_cash=args.budget,
        token=args.token,
        temperature=args.temperature,
        memory=memory,
    )
    if args.settlement_rule:
        kwargs["settlement_rule_text"] = args.settlement_rule
    return CellConfig(**kwargs)


def build_cell(base: CellConfig, axis: str, value: object) -> CellConfig:
    """One condition: baseline (unchanged) or base with a single field replaced."""
    if axis == "baseline" or value is None:
        return dataclasses.replace(base, label="baseline")
    if axis not in {f.name for f in dataclasses.fields(CellConfig)}:
        raise ValueError("unknown knob %r (not a CellConfig field)" % axis)
    coerced = value
    fields = {f.name: f for f in dataclasses.fields(CellConfig)}
    if axis == "n_outcomes":
        coerced = int(value)
    elif fields[axis].type in ("bool", bool):
        coerced = bool(value)
    return dataclasses.replace(base, label="%s=%s" % (axis, value), **{axis: coerced})


def select_battery(args):
    """Battery conditions: explicit --battery JSON wins, else the named --preset."""
    if args.battery:
        text = args.battery
        if os.path.isfile(args.battery):
            with open(args.battery, "r", encoding="utf-8") as fh:
                text = fh.read()
        data = json.loads(text)
        return tuple((str(a), tuple(vs)) for a, vs in data)
    return PRESETS[args.preset]


# --------------------------------------------------------------- factories

def _provider_routing(model: str, args) -> Optional[dict]:
    """Build the OpenRouter ``provider`` routing object for a model.

    ``--byok-only`` pins the model to its first-party provider (the vendor in the
    slug, e.g. ``anthropic/claude-...`` -> ``anthropic``). With ``only`` set,
    OpenRouter uses your BYOK key for that provider and never silently falls back
    to a different provider (Bedrock/Vertex/Azure) billed to OpenRouter credits;
    if your key is rate-limited the request fails loudly instead. ``--ignore-providers``
    is the softer alternative (just exclude named providers, keep fallbacks).
    """
    if getattr(args, "byok_only", False):
        vendor = model.split("/", 1)[0].strip().lower()
        if vendor:
            return {"only": [vendor]}
    ignore = [p.strip() for p in (getattr(args, "ignore_providers", "") or "").split(",") if p.strip()]
    if ignore:
        return {"ignore": ignore}
    return None


def _make_factory(model: str, args, dry_run: bool) -> Callable:
    if dry_run:
        return default_stand_in_factory
    return openrouter_factory(
        model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        provider=_provider_routing(model, args),
    )


# --------------------------------------------------------------- the battery

def run_battery(args) -> int:
    dry_run = args.dry_run
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("No models given. Pass --models a,b (or --dry-run for the stand-in).",
              file=sys.stderr)
        return 2
    mem_conditions = {
        "off": [False],
        "on": [True],
        "both": [False, True],
    }[args.memory]
    battery = select_battery(args)
    n = args.n

    # v2 synthetic-trader flow (None = v1 sole-trader control) + episode concurrency.
    from .synthetic import NoiseConfig
    synthetic_config = NoiseConfig() if getattr(args, "synthetic_traders", False) else None
    max_workers = max(1, int(getattr(args, "concurrency", 1) or 1))

    # Session folder. Deterministic when a --label is given (date + label), so a
    # re-run the same day RESUMES the same session rather than forking a new one.
    # --resume <path> reuses an explicit existing session regardless of date.
    date = _dt.datetime.now().strftime("%Y-%m-%d")
    if args.resume:
        session = args.resume
        label = os.path.basename(session.rstrip("/"))
    elif args.label:
        label = _slug(args.label)
        session = os.path.join(args.out, "%s_%s" % (date, label))
    else:
        label = _dt.datetime.now().strftime("%H%M%S")
        session = os.path.join(args.out, "%s_%s" % (date, label))
    stamp = os.path.basename(session.rstrip("/"))
    os.makedirs(session, exist_ok=True)
    manifest_path = os.path.join(session, "manifest.json")
    manifest = _load_manifest(manifest_path)
    done_folders = {c.get("folder") for c in manifest.get("cells", [])}

    commit = git_commit()
    total = len(models) * len(mem_conditions) * sum(len(vs) for _, vs in battery)
    from .console import BatteryUI

    ui = BatteryUI(plain=getattr(args, "plain", False))
    ui.banner(
        total=total,
        session=session,
        models=models,
        memory=args.memory,
        n=n,
        dry_run=dry_run,
        synthetic=synthetic_config is not None,
        concurrency=max_workers,
    )

    ran_cells = 0
    skipped_cells = 0

    # Record the battery plan up front (so the index exists even before runs).
    manifest["session"] = {
        "label": label,
        "timestamp": stamp,
        "models": models,
        "memory": args.memory,
        "n": n,
        "dry_run": dry_run,
        "git_commit": commit,
        "battery": [[a, list(vs)] for a, vs in battery],
        "metric_names": list(METRIC_NAMES),
        # The single arm-defining difference + the (result-neutral) perf setting.
        "synthetic_traders": synthetic_config is not None,
        "synthetic_config": dataclasses.asdict(synthetic_config) if synthetic_config else None,
        "concurrency": max_workers,
        "ignore_providers": [p.strip() for p in (getattr(args, "ignore_providers", "") or "").split(",") if p.strip()],
        "byok_only": bool(getattr(args, "byok_only", False)),
        "provider_routing": {m: _provider_routing(m, args) for m in models},
    }
    _write_manifest(manifest_path, manifest)

    idx = 0
    for model in models:
        factory = _make_factory(model, args, dry_run)
        for mem in mem_conditions:
            base = build_base_cell(args, memory=mem)
            calibration = _run_and_store_calibration(
                session, model, mem, base, factory, args, dry_run,
                synthetic_config=synthetic_config, max_workers=max_workers,
            )
            for axis, values in battery:
                for value in values:
                    idx += 1
                    folder_name = cell_folder_name(model, mem, axis, value, n)
                    folder = os.path.join(session, folder_name)
                    prefix = "[%d/%d] %s mem-%s %s=%s" % (
                        idx, total, model, "on" if mem else "off", axis, _fmt_value(value),
                    )

                    if _cell_complete(folder, n):
                        ui.cell_skip(prefix)
                        skipped_cells += 1
                        if folder_name not in done_folders:
                            _index_existing(manifest, folder, folder_name, model, mem,
                                             axis, value, n)
                            _write_manifest(manifest_path, manifest)
                            done_folders.add(folder_name)
                        continue

                    os.makedirs(folder, exist_ok=True)
                    cell = build_cell(base, axis, value)
                    ui.cell_running(prefix)
                    t0 = time.time()
                    result = run_cell(
                        cell, factory, n_runs=n, store_logs=True,
                        synthetic_config=synthetic_config, max_workers=max_workers,
                    )
                    elapsed = (time.time() - t0) / 60.0

                    # Write artifacts.
                    meta = {
                        "model": model,
                        "memory": mem,
                        "axis": axis,
                        "value": value,
                        "n": n,
                        "timestamp": _dt.datetime.now().isoformat(),
                        "git_commit": commit,
                        "label": label,
                        "dry_run": dry_run,
                        "synthetic_traders": synthetic_config is not None,
                    }
                    config = _config_dict(cell, meta)
                    chash = _config_hash(config)
                    config["meta"]["config_hash"] = chash
                    _dump_json(os.path.join(folder, "config.json"), config)

                    with open(os.path.join(folder, "runs.jsonl"), "w", encoding="utf-8") as fh:
                        for rec in result.records:
                            fh.write(_episode_line(rec.run, rec.run, rec.log) + "\n")

                    _dump_json(
                        os.path.join(folder, "metrics.json"),
                        _distribution_payload(result.metrics, result.records),
                    )
                    _write_summary(
                        os.path.join(folder, "summary.txt"),
                        model=model, memory=mem, axis=axis, value=value, n=n,
                        metrics=result.metrics, calibration=calibration,
                    )

                    # Crash-safe manifest append.
                    manifest["cells"].append({
                        "folder": folder_name,
                        "model": model,
                        "memory": "on" if mem else "off",
                        "axis": axis,
                        "value": value,
                        "n": n,
                        "config_hash": chash,
                        "headline": _headline_payload(result.metrics),
                        "calibration_passed": (calibration or {}).get("passed"),
                        "status": "complete",
                    })
                    done_folders.add(folder_name)
                    _write_manifest(manifest_path, manifest)

                    ui.cell_done(prefix, _calib_status(calibration), elapsed)
                    ran_cells += 1

    ui.complete(
        session=session,
        manifest_path=manifest_path,
        ran=ran_cells,
        skipped=skipped_cells,
        total=total,
    )
    return 0


def _run_and_store_calibration(session, model, mem, base, factory, args, dry_run,
                               synthetic_config=None, max_workers=1):
    """Run the control pair ONCE per (model, memory); record-only (never gates)."""
    if not args.calibrate:
        return None
    path = os.path.join(session, "calibration__%s__mem-%s.json" % (_slug(model), "on" if mem else "off"))
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):  # pragma: no cover
            pass
    cal = run_calibration(
        base, factory, n_runs=args.calib_n, trust_discovery=not dry_run,
        synthetic_config=synthetic_config, max_workers=max_workers,
    )
    payload = {
        "model": model,
        "memory": "on" if mem else "off",
        "metric": "discovery_rank_score",
        "easy_discovery": cal.easy_discovery,
        "hard_discovery": cal.hard_discovery,
        "margin": cal.margin,
        "passed": cal.passed,
        "trusted": cal.trusted,
        "note": cal.note,
        "calib_n": args.calib_n,
    }
    _dump_json(path, payload)
    return payload


def _index_existing(manifest, folder, folder_name, model, mem, axis, value, n):
    """Reconstruct a manifest entry for a pre-existing complete cell (resume)."""
    headline = {}
    chash = None
    try:
        with open(os.path.join(folder, "metrics.json"), "r", encoding="utf-8") as fh:
            m = json.load(fh)
        headline = {
            name: {"median": m.get(name, {}).get("median"),
                   "iqr": m.get(name, {}).get("iqr")}
            for name in HEADLINE_METRICS
        }
    except (OSError, ValueError):  # pragma: no cover
        pass
    try:
        with open(os.path.join(folder, "config.json"), "r", encoding="utf-8") as fh:
            chash = json.load(fh).get("meta", {}).get("config_hash")
    except (OSError, ValueError):  # pragma: no cover
        pass
    manifest["cells"].append({
        "folder": folder_name, "model": model, "memory": "on" if mem else "off",
        "axis": axis, "value": value, "n": n, "config_hash": chash,
        "headline": headline, "status": "complete (resumed)",
    })


def _calib_status(calibration: Optional[dict]) -> str:
    if calibration is None:
        return "NA"
    if not calibration.get("trusted"):
        return "untrusted"  # stand-in / dry-run: discovery not meaningful
    return "PASS" if calibration.get("passed") else "fail"


def _dump_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


# --------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fingerprint_eval.battery",
        description="Run an official, organized, resumable evaluation battery.",
    )
    p.add_argument("--models", default=None,
                   help="comma-separated OpenRouter model slugs (the 2 probes)")
    p.add_argument("--label", default=None,
                   help="session label (default: time only). Same --label same day RESUMES.")
    p.add_argument("--resume", default=None,
                   help="explicit existing session folder to resume (overrides --label/date)")
    p.add_argument("--out", default="eval_runs", help="top-level output folder")
    p.add_argument("--memory", choices=["off", "on", "both"], default="both")
    p.add_argument("--n", type=int, default=30, help="runs per cell")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default="core",
                   help="named battery: core (default), geometry, budget, disclosure, drift")
    p.add_argument("--battery", default=None,
                   help="JSON file/string of [[axis,[values]],...] (overrides --preset)")
    p.add_argument("--calibrate", dest="calibrate", action="store_true", default=True,
                   help="run the control pair per (model,memory), record-only (default on)")
    p.add_argument("--no-calibrate", dest="calibrate", action="store_false")
    p.add_argument("--calib-n", dest="calib_n", type=int, default=10,
                   help="runs per control cell (default 10)")
    p.add_argument("--dry-run", action="store_true",
                   help="offline stand-in model (no API calls); validates plumbing/structure")
    p.add_argument("--plain", action="store_true",
                   help="plain text output (no color/boxes; also set NO_COLOR=1)")
    p.add_argument("--env-file", dest="env_file", default=".env")

    # frozen base-cell knobs (identical across the battery unless that knob is swept)
    p.add_argument("--volatility", type=float, default=1.0)
    p.add_argument("--knowability", type=float, default=0.0)
    p.add_argument("--band-width", dest="band_width", type=float, default=10.0)
    p.add_argument("--anchor", type=float, default=100.0)
    p.add_argument("--n-outcomes", dest="n_outcomes", type=int, default=3)
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--drift", type=float, default=0.0)
    p.add_argument("--k", type=float, default=10.0)
    p.add_argument("--fee", type=float, default=0.02)
    p.add_argument("--budget", type=float, default=1000.0)
    p.add_argument("--token", default="ZQX")
    p.add_argument("--settlement-rule", dest="settlement_rule", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=512)
    p.add_argument("--ignore-providers", dest="ignore_providers", default="",
                   help="comma-separated OpenRouter provider slugs to exclude "
                        "(e.g. 'azure' to route around Azure's content filter)")
    p.add_argument("--byok-only", dest="byok_only", action="store_true",
                   help="pin each model to its first-party provider via "
                        "provider.only (e.g. anthropic/openai) so requests use "
                        "your BYOK key and never silently fall back to a "
                        "credit-billed provider. Fails loudly if your key is "
                        "rate-limited rather than falling back.")

    # v2 treatment + performance (neither changes any v1 result):
    p.add_argument("--synthetic-traders", dest="synthetic_traders", action="store_true",
                   help="V2 ARM: add synthetic ('noise') trader flow so the board is "
                        "live between turns. Off = the v1 sole-trader control.")
    p.add_argument("--concurrency", type=int, default=1,
                   help="episodes to run CONCURRENTLY per cell (bound under the API "
                        "rate limit). Determinism-preserving: only execution order "
                        "changes, never an episode's seeded inputs.")

    # money-band market (Delphi-style)
    p.add_argument("--mid", type=float, default=None)
    p.add_argument("--lower-factor", dest="lower_factor", type=float, default=0.984)
    p.add_argument("--upper-factor", dest="upper_factor", type=float, default=1.016)
    p.add_argument("--currency", default="$")
    p.add_argument("--question", default=None)
    p.add_argument("--data-source", dest="data_source", default=None)
    p.add_argument("--real-identity", dest="real_identity", action="store_true")
    return p


def main(argv=None) -> int:
    from .cli import load_dotenv

    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv(args.env_file)
    if not args.models and not args.dry_run:
        env_model = os.environ.get("OPENROUTER_MODEL")
        if env_model:
            args.models = env_model
        else:
            print("No --models given. Pass --models a,b or use --dry-run.", file=sys.stderr)
            return 2
    if args.dry_run and not args.models:
        args.models = "offline-stand-in"
    return run_battery(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
