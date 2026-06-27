"""The model interface, deterministic stand-ins, and the OpenRouter backend.

A model is anything callable as ``model(prompt: str) -> str`` returning a raw
response (the harness parses it). The deterministic stand-ins
(:class:`HoldModel`, :class:`ScriptedModel`, :class:`RandomTrader`) let the full
loop run and be tested without a network; :class:`OpenRouterModel` is the real
backend that puts an actual LLM behind the protocol -- the moment discovery
results become meaningful (see the analysis layer's plumbing-only caveat).

The OpenRouter client uses only the standard library (``urllib``) so the project
stays dependency-free, and the HTTP call is injectable (``transport=``) so it is
fully testable offline. The API key is read from the ``OPENROUTER_API_KEY``
environment variable and is never logged.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional

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


# --------------------------------------------------------------- OpenRouter backend

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_SYSTEM_PROMPT = (
    "You are a disciplined trader on a prediction market. Decide using only the "
    "information in the prompt; you cannot see the future. Respond with ONLY a "
    "single JSON object matching the schema given in the prompt -- no prose, no "
    "code fences, no explanation outside the JSON's rationale field."
)

# Transient HTTP statuses worth retrying; auth errors are not retried.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_AUTH_STATUS = {401, 403}


class OpenRouterError(RuntimeError):
    """Raised when an OpenRouter call ultimately fails. ``retryable`` is advisory."""

    def __init__(self, message: str, retryable: bool = False, status: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


# A transport maps a request payload dict to the parsed JSON response dict. The
# default is real HTTP; tests (and custom backends) inject their own.
Transport = Callable[[dict], dict]


class OpenRouterModel:
    """A real LLM behind the Model protocol, via OpenRouter's OpenAI-compatible API.

    Parameters
    ----------
    model : OpenRouter model slug, e.g. ``"openai/gpt-4o-mini"``,
        ``"anthropic/claude-3.5-sonnet"``, ``"google/gemini-2.0-flash-001"``.
    api_key : defaults to the ``OPENROUTER_API_KEY`` environment variable. Never
        logged or persisted.
    temperature, max_tokens, seed : sampling controls. ``seed`` is best-effort
        reproducibility (provider-dependent).
    system_prompt : the system role text (JSON-only discipline by default).
    timeout, max_retries, backoff_base : network robustness. Retries transient
        errors with exponential backoff; auth errors fail fast.
    on_error : ``"raise"`` (default) to surface persistent failures, or ``"hold"``
        to degrade a failed turn into a logged hold so a long sweep survives.
    referer, title : optional OpenRouter ranking headers.
    transport : inject a callable ``payload -> response_dict`` to bypass HTTP
        (used by tests and custom backends).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        on_error: str = "raise",
        referer: Optional[str] = None,
        title: Optional[str] = None,
        transport: Optional[Transport] = None,
    ):
        if on_error not in ("raise", "hold"):
            raise ValueError("on_error must be 'raise' or 'hold'")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.on_error = on_error
        self.referer = referer
        self.title = title
        self._transport = transport or self._http_post

        # Only require a key for the real transport (tests inject their own).
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if transport is None and not self._api_key:
            raise ValueError(
                "no OpenRouter API key: set the OPENROUTER_API_KEY environment "
                "variable or pass api_key=..."
            )

    # ---- public protocol method

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        try:
            response = self._complete(payload)
            return _extract_content(response)
        except OpenRouterError as exc:
            if self.on_error == "hold":
                return json.dumps(
                    {"action": "hold", "rationale": "model error (degraded to hold): %s" % exc}
                )
            raise

    # ---- retry loop (transport-agnostic, so injected transports are retried too)

    def _complete(self, payload: dict) -> dict:
        attempt = 0
        while True:
            try:
                return self._transport(payload)
            except OpenRouterError as exc:
                attempt += 1
                if not exc.retryable or attempt > self.max_retries:
                    raise
                time.sleep(self.backoff_base * (2 ** (attempt - 1)))

    # ---- default real HTTP transport

    def _http_post(self, payload: dict) -> dict:
        url = self.base_url + "/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": "Bearer %s" % self._api_key,
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.title:
            headers["X-Title"] = self.title
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            status = exc.code
            try:
                detail = exc.read().decode("utf-8")
            except Exception:  # pragma: no cover - defensive
                detail = ""
            retryable = status in _RETRYABLE_STATUS
            raise OpenRouterError(
                "OpenRouter HTTP %s: %s" % (status, detail[:500]),
                retryable=retryable,
                status=status,
            )
        except urllib.error.URLError as exc:  # network/DNS/timeout
            raise OpenRouterError("OpenRouter network error: %s" % exc, retryable=True)
        except json.JSONDecodeError as exc:
            raise OpenRouterError("OpenRouter returned non-JSON: %s" % exc, retryable=False)


def _extract_content(response: dict) -> str:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError("malformed OpenRouter response: %s" % exc, retryable=False)


def openrouter_factory(model: str, seed_per_run: bool = True, **kwargs) -> Callable:
    """Build a ModelFactory ``(cell, run) -> OpenRouterModel`` for the experiment runner.

    With ``seed_per_run`` the per-run seed is set to the run index for best-effort
    reproducibility across the N runs of a cell. Extra kwargs pass through to
    :class:`OpenRouterModel` (temperature, max_tokens, on_error, referer, ...).

    The cell's ``temperature`` is authoritative when present so that sweeping the
    ``temperature`` axis actually varies the model. The ``temperature`` kwarg here
    only acts as a fallback (e.g. for a cell-less ``--ping``).
    """

    def factory(cell, run):
        kw = dict(kwargs)
        if seed_per_run and "seed" not in kw:
            kw["seed"] = run
        cell_temp = getattr(cell, "temperature", None)
        if cell_temp is not None:
            kw["temperature"] = cell_temp
        return OpenRouterModel(model=model, **kw)

    return factory
