"""The model interface and baseline/test implementations (spec.md piece 4).

A model is anything callable as ``model(prompt: str) -> str`` returning a raw
response (the harness parses it). Real LLM backends plug in here later (and the
logged input->output pairs are exactly what an REE replay needs); this module
ships only deterministic stand-ins so the full loop is runnable and testable
without a network or API keys.
"""

from __future__ import annotations

import json
import random
from typing import List, Optional

try:  # Protocol is the cleanest type, but keep it optional for older runtimes.
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore


class Model(Protocol):
    """Anything that maps a prompt to a raw response string."""

    def __call__(self, prompt: str) -> str:  # pragma: no cover - structural
        ...


class HoldModel:
    """Always holds. Useful as a control / no-op baseline."""

    def __call__(self, prompt: str) -> str:
        return json.dumps({"action": "hold", "rationale": "control: always hold"})


class ScriptedModel:
    """Replays a fixed list of responses in order (then holds). For tests.

    Responses may be raw strings or dicts (dicts are JSON-encoded).
    """

    def __init__(self, responses: List[object]):
        self._responses = list(responses)
        self._i = 0

    def __call__(self, prompt: str) -> str:
        if self._i >= len(self._responses):
            return json.dumps({"action": "hold", "rationale": "script exhausted"})
        item = self._responses[self._i]
        self._i += 1
        if isinstance(item, dict):
            return json.dumps(item)
        return str(item)


class RandomTrader:
    """Seeded random valid actions. A baseline to exercise/read the loop end to end.

    It does not read the prompt -- it is a synthetic stand-in for a real model, so
    it is configured directly with the number of outcomes and a size range.
    """

    def __init__(
        self,
        n_outcomes: int,
        seed: Optional[int] = None,
        max_size: float = 100.0,
        hold_prob: float = 0.3,
    ):
        self.n_outcomes = n_outcomes
        self.max_size = max_size
        self.hold_prob = hold_prob
        self._rng = random.Random(seed)

    def __call__(self, prompt: str) -> str:
        if self._rng.random() < self.hold_prob:
            return json.dumps({"action": "hold", "rationale": "random: hold"})
        action = self._rng.choice(["buy", "sell"])
        outcome = self._rng.randrange(self.n_outcomes)
        size = round(self._rng.uniform(1.0, self.max_size), 2)
        return json.dumps(
            {
                "action": action,
                "outcome": outcome,
                "size": size,
                "rationale": "random: %s %g on %d" % (action, size, outcome),
            }
        )
