"""The rigid action schema and a robust parser (spec.md piece 4).

The model chooses *what / when / how much* freely, but must emit a fixed
structure:

    { "action": "buy" | "sell" | "hold",
      "outcome": <int>,
      "size":    <number>,        # NOTE: notional -- currency to spend/receive
      "rationale": "<free text>" }

``size`` is a *notional* amount (currency), not a share count: traders think in
money, and that is how the real Delphi interface presents it. The harness owns
the notional->shares conversion.

Parsing must be robust: a real model wraps JSON in prose or code fences, or emits
something malformed. The parser never raises -- on failure it returns a parsed
"hold" with ``parse_ok=False`` and the error, so the harness logs the bad turn
(itself behavioral signal) and continues. The ``rationale`` is logged as observed
behavior, never trusted as the cause of a decision.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

ACTION_BUY = "buy"
ACTION_SELL = "sell"
ACTION_HOLD = "hold"
VALID_ACTIONS = (ACTION_BUY, ACTION_SELL, ACTION_HOLD)


@dataclass
class ParsedAction:
    """A normalized action plus parse provenance."""

    action: str
    outcome: int
    size: float
    rationale: str
    parse_ok: bool
    raw: str
    parse_error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "outcome": self.outcome,
            "size": self.size,
            "rationale": self.rationale,
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
        }


def _hold(raw: str, error: str) -> ParsedAction:
    return ParsedAction(
        action=ACTION_HOLD,
        outcome=0,
        size=0.0,
        rationale="",
        parse_ok=False,
        raw=raw,
        parse_error=error,
    )


def _extract_json_object(text: str) -> Optional[str]:
    """Pull the first balanced ``{...}`` object out of arbitrary model text."""
    # Drop ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    # Otherwise scan for the first balanced brace span.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse(response: str) -> ParsedAction:
    """Parse a model response into a :class:`ParsedAction` (never raises)."""
    if response is None:
        return _hold("", "empty response")
    raw = str(response)

    blob = _extract_json_object(raw)
    if blob is None:
        return _hold(raw, "no JSON object found")
    try:
        data = json.loads(blob)
    except (ValueError, TypeError) as exc:
        return _hold(raw, "JSON decode error: %s" % exc)
    if not isinstance(data, dict):
        return _hold(raw, "parsed JSON is not an object")

    action = str(data.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        return _hold(raw, "invalid or missing action: %r" % data.get("action"))

    rationale = str(data.get("rationale", ""))

    if action == ACTION_HOLD:
        return ParsedAction(ACTION_HOLD, 0, 0.0, rationale, True, raw, None)

    # buy / sell require an outcome and a size.
    try:
        outcome = int(data["outcome"])
    except (KeyError, TypeError, ValueError):
        return _hold(raw, "invalid or missing outcome: %r" % data.get("outcome"))

    try:
        size = float(data["size"])
    except (KeyError, TypeError, ValueError):
        return _hold(raw, "invalid or missing size: %r" % data.get("size"))

    if not (size == size):  # NaN guard
        return _hold(raw, "size is NaN")

    return ParsedAction(action, outcome, size, rationale, True, raw, None)
