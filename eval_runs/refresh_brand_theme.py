#!/usr/bin/env python3
"""Refresh bundled dashboard-dark theme from live Gensyn brand manifest.

Usage:
  PYTHONPATH=src python3 eval_runs/refresh_brand_theme.py [--write]

Without --write, prints JSON to stdout. With --write, updates:
  src/fingerprint_eval/brand_data/dashboard-dark.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fingerprint_eval import brand  # noqa: E402

TARGET = os.path.join(ROOT, "src", "fingerprint_eval", "brand_data", "dashboard-dark.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh Gensyn dashboard-dark TUI theme cache")
    ap.add_argument("--write", action="store_true", help="write bundled theme file")
    ap.add_argument("--url", default=brand.MANIFEST_URL)
    args = ap.parse_args()
    data = brand.refresh_theme_from_manifest(url=args.url)
    text = json.dumps(data, indent=2) + "\n"
    if args.write:
        with open(TARGET, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("wrote %s" % TARGET, file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
