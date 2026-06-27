"""Command-line entrypoint: run the full instrument against a real model.

Wires an OpenRouter model into the calibration-gated experiment runner and prints
the honest analysis report. The API key is read from ``OPENROUTER_API_KEY`` and
never logged.

Examples
--------
    export OPENROUTER_API_KEY=sk-or-...
    python -m agent_eval_harness --model openai/gpt-4o-mini --axis volatility \
        --values 0.1,0.5,2.0 --n 10

    # quick connectivity check (one call), no sweep:
    python -m agent_eval_harness --model openai/gpt-4o-mini --ping
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from .analysis import format_report
from .experiment import CellConfig, default_stand_in_factory, run_experiment
from .market_template import money_band_market
from .model import openrouter_factory


def load_dotenv(path: str = ".env") -> bool:
    """Load simple ``KEY=VALUE`` pairs from a .env file into os.environ.

    Dependency-free. Existing environment variables win (so an explicit
    ``export`` overrides the file). Lines that are blank or start with ``#`` are
    ignored; surrounding quotes on the value are stripped. Returns True if the
    file was found.
    """
    if not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    return True


def _parse_values(axis: str, raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip() != ""]


def build_base_cell(args) -> CellConfig:
    # Money-band (Delphi-style) market when --mid is given.
    if args.mid is not None:
        return money_band_market(
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
    )
    if args.settlement_rule:
        kwargs["settlement_rule_text"] = args.settlement_rule
    return CellConfig(**kwargs)


def _ping(args) -> int:
    """One-shot call to confirm the model + key work before spending a full sweep."""
    factory = openrouter_factory(
        args.model, temperature=args.temperature, max_tokens=args.max_tokens
    )
    model = factory(None, 0)
    prompt = (
        "Reply with ONLY this JSON object and nothing else: "
        '{ "action": "hold", "outcome": 0, "size": 0, "rationale": "ping" }'
    )
    print("Pinging %s via OpenRouter ..." % args.model)
    try:
        out = model(prompt)
    except Exception as exc:  # surface auth/config errors plainly
        print("PING FAILED: %s" % exc, file=sys.stderr)
        return 2
    print("Model replied:\n%s" % out)
    print("\nOK -- the model is reachable and the key works.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_eval_harness",
        description="Run the prediction-market agent-evaluation instrument.",
    )
    parser.add_argument("--model", default=None,
                        help="OpenRouter model slug (default: OPENROUTER_MODEL from .env/env)")
    parser.add_argument("--env-file", dest="env_file", default=".env",
                        help="path to the .env file to load (default: .env)")
    parser.add_argument("--ping", action="store_true", help="just test connectivity and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="use the offline stand-in model (no API calls); plumbing only")

    # sweep
    parser.add_argument("--axis", default="volatility",
                        choices=["volatility", "knowability_min", "band_width", "temperature", "drift"])
    parser.add_argument("--values", default=None, help="comma-separated swept values")
    parser.add_argument("--n", type=int, default=10, help="runs per cell (your 'n')")

    # base cell / environment (frozen across the sweep)
    parser.add_argument("--volatility", type=float, default=0.5, help="base volatility (band-widths/sqrt-hour)")
    parser.add_argument("--knowability", type=float, default=0.0, help="base dead-window minutes")
    parser.add_argument("--band-width", dest="band_width", type=float, default=10.0)
    parser.add_argument("--anchor", type=float, default=100.0)
    parser.add_argument("--n-outcomes", dest="n_outcomes", type=int, default=3)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=30.0, help="timestep minutes")
    parser.add_argument("--drift", type=float, default=0.0)
    parser.add_argument("--k", type=float, default=10.0, help="DPM liquidity constant")
    parser.add_argument("--fee", type=float, default=0.02)
    parser.add_argument("--budget", type=float, default=1000.0, help="starting cash per run")
    parser.add_argument("--token", default="ZQX", help="fictional asset ticker shown to the model")
    parser.add_argument("--settlement-rule", dest="settlement_rule", default=None,
                        help="settlement rule text shown to the model (overrides the default)")

    # Money-band (Delphi-style) market: 4 outcomes from MID x factors.
    parser.add_argument("--mid", type=float, default=None,
                        help="MID/anchor price -> builds a 4-outcome money-band market")
    parser.add_argument("--lower-factor", dest="lower_factor", type=float, default=0.984)
    parser.add_argument("--upper-factor", dest="upper_factor", type=float, default=1.016)
    parser.add_argument("--currency", default="$")
    parser.add_argument("--question", default=None, help="settlement question text (keep it info-safe)")
    parser.add_argument("--data-source", dest="data_source", default=None,
                        help="data source (only shown with --real-identity)")
    parser.add_argument("--real-identity", dest="real_identity", action="store_true",
                        help="OPT-IN: allow real token/date/source -- NOT a controlled experiment")

    # model sampling
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=512)

    # gating / trust
    parser.add_argument("--no-trust", dest="trust", action="store_false",
                        help="do not gate the grid on calibration (and treat results as untrusted)")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="omit ASCII overlap plots")

    args = parser.parse_args(argv)

    # Load .env so the key + model can come from the file instead of the CLI.
    load_dotenv(args.env_file)
    if not args.model:
        args.model = os.environ.get("OPENROUTER_MODEL")
    if not args.model and not args.dry_run:
        print(
            "No model specified. Set OPENROUTER_MODEL in %s (or pass --model), "
            "or use --dry-run for the offline stand-in." % args.env_file,
            file=sys.stderr,
        )
        return 2

    if args.ping:
        return _ping(args)

    default_values = {
        "volatility": "0.1,0.5,2.0",
        "knowability_min": "0,60,120",
        "band_width": "5,20",
        "temperature": "0.0,0.7,1.2",
        "drift": "0,0.5,1.0,2.0",
    }
    values = _parse_values(args.axis, args.values or default_values[args.axis])

    if args.mid is not None and args.axis == "band_width":
        print(
            "Cannot sweep band_width on a --mid money-band market (bands are fixed by "
            "the factors). Sweep volatility or knowability_min, or use the uniform "
            "market without --mid.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        factory = default_stand_in_factory
        trust = False
        print("DRY RUN: offline stand-in model. Output validates plumbing only.\n")
    else:
        factory = openrouter_factory(
            args.model, temperature=args.temperature, max_tokens=args.max_tokens
        )
        trust = args.trust

    base = build_base_cell(args)
    model_label = args.model if not args.dry_run else "offline-stand-in"
    print("Sweeping %s over %s with n=%d per cell (model=%s, trusted=%s)\n"
          % (args.axis, values, args.n, model_label, trust))

    result = run_experiment(
        base, args.axis, values, model_factory=factory, n_runs=args.n, trust_discovery=trust
    )

    cal = result.calibration
    print("CALIBRATION (control pair, cell zero):")
    print("  easy discovery (median) = %.4f" % cal.easy_discovery)
    print("  hard discovery (median) = %.4f" % cal.hard_discovery)
    print("  passed = %s" % cal.passed)
    print("  %s\n" % cal.note)

    if not result.ran_grid:
        print(result.message)
        return 1

    real_model = not args.dry_run
    if real_model and not trust:
        print(
            "Grid ran with --no-trust: calibration gate bypassed. BEHAVIOR metrics "
            "below are real %s output; DISCOVERY is ungated and not a finding." % args.model
        )
    else:
        print(result.message)
    print()
    print(format_report(result.grid, trusted=trust, show_plots=args.plots, real_model=real_model))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
