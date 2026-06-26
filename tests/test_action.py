"""Tests for the action parser -- robustness is the point (never raise)."""

from agent_eval_harness.action import parse


def test_parses_clean_json():
    a = parse('{"action":"buy","outcome":2,"size":50,"rationale":"because"}')
    assert a.parse_ok
    assert a.action == "buy"
    assert a.outcome == 2
    assert a.size == 50.0
    assert a.rationale == "because"


def test_parses_json_in_code_fence_with_prose():
    text = "Sure, here is my decision:\n```json\n{\"action\": \"sell\", \"outcome\": 1, \"size\": 12.5}\n```\nHope that helps!"
    a = parse(text)
    assert a.parse_ok
    assert a.action == "sell"
    assert a.outcome == 1
    assert a.size == 12.5


def test_parses_object_embedded_in_prose():
    a = parse('I will hold. {"action":"hold","rationale":"waiting"} done.')
    assert a.parse_ok
    assert a.action == "hold"


def test_hold_does_not_require_outcome_or_size():
    a = parse('{"action":"hold"}')
    assert a.parse_ok
    assert a.action == "hold"


def test_garbage_becomes_safe_hold():
    a = parse("I refuse to answer in JSON.")
    assert not a.parse_ok
    assert a.action == "hold"
    assert a.parse_error is not None


def test_invalid_action_value_rejected():
    a = parse('{"action":"yolo","outcome":0,"size":1}')
    assert not a.parse_ok
    assert a.action == "hold"


def test_missing_size_on_buy_rejected():
    a = parse('{"action":"buy","outcome":0}')
    assert not a.parse_ok


def test_empty_response_is_hold():
    assert parse("").action == "hold"
    assert parse(None).action == "hold"
