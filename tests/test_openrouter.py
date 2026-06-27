"""Offline tests for the OpenRouter backend (no network).

The HTTP call is injected via ``transport=`` so we exercise payload shape, content
extraction, retry/backoff, auth-fast-fail, on_error degradation, and the factory
without ever touching the network.
"""

import json
import os

import pytest

from agent_eval_harness.action import parse
from agent_eval_harness.model import (
    OpenRouterError,
    OpenRouterModel,
    openrouter_factory,
)


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


# ----------------------------------------------------------------- happy path


def test_returns_model_content_and_sends_expected_payload():
    captured = {}

    def transport(payload):
        captured.update(payload)
        return _chat_response('{"action":"buy","outcome":1,"size":25,"rationale":"ok"}')

    m = OpenRouterModel("openai/gpt-4o-mini", transport=transport, seed=3)
    out = m("PROMPT TEXT")
    assert parse(out).action == "buy"

    assert captured["model"] == "openai/gpt-4o-mini"
    assert captured["seed"] == 3
    roles = [msg["role"] for msg in captured["messages"]]
    assert roles == ["system", "user"]
    assert captured["messages"][1]["content"] == "PROMPT TEXT"


def test_no_api_key_required_when_transport_injected():
    # Injected transport must not require OPENROUTER_API_KEY.
    m = OpenRouterModel("x/y", transport=lambda p: _chat_response("{}"))
    assert m("hi") == "{}"


def test_missing_api_key_raises_for_real_transport(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenRouterModel("openai/gpt-4o-mini")


# ----------------------------------------------------------------- malformed response


def test_malformed_response_raises_openrouter_error():
    m = OpenRouterModel("x/y", transport=lambda p: {"unexpected": True})
    with pytest.raises(OpenRouterError):
        m("hi")


# ----------------------------------------------------------------- retries / backoff


def test_retries_transient_errors_then_succeeds():
    calls = {"n": 0}

    def flaky(payload):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OpenRouterError("503", retryable=True, status=503)
        return _chat_response('{"action":"hold"}')

    m = OpenRouterModel("x/y", transport=flaky, max_retries=5, backoff_base=0.0)
    assert parse(m("hi")).action == "hold"
    assert calls["n"] == 3


def test_non_retryable_error_fails_fast():
    calls = {"n": 0}

    def auth_fail(payload):
        calls["n"] += 1
        raise OpenRouterError("401 unauthorized", retryable=False, status=401)

    m = OpenRouterModel("x/y", transport=auth_fail, max_retries=5, backoff_base=0.0)
    with pytest.raises(OpenRouterError):
        m("hi")
    assert calls["n"] == 1  # no retries on auth failure


def test_retries_exhausted_raises():
    def always_503(payload):
        raise OpenRouterError("503", retryable=True, status=503)

    m = OpenRouterModel("x/y", transport=always_503, max_retries=2, backoff_base=0.0)
    with pytest.raises(OpenRouterError):
        m("hi")


# ----------------------------------------------------------------- on_error degradation


def test_on_error_hold_degrades_to_hold():
    def always_fail(payload):
        raise OpenRouterError("boom", retryable=False)

    m = OpenRouterModel("x/y", transport=always_fail, on_error="hold")
    out = m("hi")
    assert parse(out).action == "hold"


def test_invalid_on_error_rejected():
    with pytest.raises(ValueError):
        OpenRouterModel("x/y", transport=lambda p: {}, on_error="explode")


# ----------------------------------------------------------------- factory


def test_factory_seeds_per_run_and_builds_models():
    transport = lambda p: _chat_response('{"action":"hold"}')
    factory = openrouter_factory("x/y", transport=transport)
    m0 = factory(None, 0)
    m1 = factory(None, 1)
    assert isinstance(m0, OpenRouterModel)
    assert m0.seed == 0 and m1.seed == 1
    assert parse(m0("hi")).action == "hold"


def test_load_dotenv_sets_missing_keys_only(tmp_path, monkeypatch):
    from agent_eval_harness.cli import load_dotenv

    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "OPENROUTER_API_KEY=sk-or-fromfile\n"
        'OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"\n'
        "\n"
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_MODEL", "already/set")  # existing env must win

    assert load_dotenv(str(env)) is True
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-fromfile"   # filled from file
    assert os.environ["OPENROUTER_MODEL"] == "already/set"        # not overwritten


def test_load_dotenv_missing_file_is_noop():
    from agent_eval_harness.cli import load_dotenv

    assert load_dotenv("definitely-not-a-real-file.env") is False


def test_factory_runs_a_full_episode_offline():
    # End-to-end: a scripted OpenRouter transport drives a real episode.
    from agent_eval_harness.experiment import CellConfig, run_cell

    def transport(payload):
        return _chat_response('{"action":"buy","outcome":2,"size":40,"rationale":"bull"}')

    factory = openrouter_factory("x/y", transport=transport)
    cell = CellConfig(volatility=0.5, knowability_min=0.0, band_width=10.0, n_outcomes=3)
    result = run_cell(cell, factory, n_runs=3)
    assert result.n_runs == 3
    # the model always buys outcome 2, so trades should execute
    assert result.median("trade_count") >= 1
