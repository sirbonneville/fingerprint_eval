"""Tests for the money-band market template (Delphi-style settlement format)."""

import pytest

from fingerprint_eval.experiment import run_cell
from fingerprint_eval.market_template import (
    band_labels_money,
    format_money,
    money_band_market,
    settlement_rules_text,
)
from fingerprint_eval.model import HoldModel


def test_format_money():
    assert format_money(61273.52) == "$61,273.52"
    assert format_money(60293.14) == "$60,293.14"
    assert format_money(1000, currency="USD ") == "USD 1,000.00"


def test_band_labels_money_four_outcomes():
    edges = [60293.14, 61273.52, 62253.90]
    labels = band_labels_money(edges)
    assert labels == [
        "Below $60,293.14",
        "Between $60,293.14 and $61,273.52",
        "Between $61,273.52 and $62,253.90",
        "Above $62,253.90",
    ]


def test_money_band_market_maps_btc_example():
    # The BTC example: MID=61273.52, +/-1.6% -> the same band edges.
    cell = money_band_market(token="ZQX", mid=61273.52, volatility=0.5, knowability_min=0.0)
    edges = cell.band_edges()
    assert edges[0] == pytest.approx(61273.52 * 0.984)   # LOWER 60293.14
    assert edges[1] == pytest.approx(61273.52)            # MID
    assert edges[2] == pytest.approx(61273.52 * 1.016)   # UPPER 62253.90
    assert cell.n_outcomes == 4
    assert cell.outcome_labels()[0].startswith("Below $")
    assert cell.outcome_labels()[-1].startswith("Above $")


def test_settlement_text_is_info_safe_by_default():
    cell = money_band_market(
        token="ZQX", mid=61273.52, volatility=0.5, knowability_min=0.0,
        question=None, data_source="https://www.coingecko.com/en/coins/bitcoin/historical_data",
    )
    text = cell.settlement_rule_text
    # safe mode: no real source leaked, no calendar date, describing voice
    assert "coingecko" not in text.lower()
    assert "2026" not in text
    assert "You are an AI judge" not in text
    assert "settled by an AI judge" in text
    assert "POSSIBLE OUTCOMES" in text


def test_real_identity_includes_source():
    text = settlement_rules_text(
        token="BTC",
        edges=[60293.14, 61273.52, 62253.90],
        labels=band_labels_money([60293.14, 61273.52, 62253.90]),
        data_source="https://www.coingecko.com/en/coins/bitcoin/historical_data",
        real_identity=True,
    )
    assert "coingecko" in text.lower()


def test_boundaries_match_mechanical_resolver():
    # The market must resolve a reference price to the band the rules describe.
    cell = money_band_market(token="ZQX", mid=100.0, volatility=0.5, knowability_min=0.0,
                             lower_factor=0.9, upper_factor=1.1)
    m = cell.build_market()  # edges [90, 100, 110]
    assert m.resolve_outcome(85.0) == 0   # Below $90
    assert m.resolve_outcome(95.0) == 1   # Between 90 and 100
    assert m.resolve_outcome(105.0) == 2  # Between 100 and 110
    assert m.resolve_outcome(120.0) == 3  # Above $110


def test_template_prompt_shows_labels_and_rules_and_token():
    cell = money_band_market(token="ZQX", mid=100.0, volatility=0.5, knowability_min=0.0,
                             lower_factor=0.9, upper_factor=1.1)
    result = run_cell(cell, lambda c, r: HoldModel(), n_runs=1, store_logs=True)
    prompt = result.records[0].log.turns[0].prompt
    assert "Below $90.00" in prompt
    assert "Above $110.00" in prompt
    assert "POSSIBLE OUTCOMES" in prompt
    assert "ZQX" in prompt


def test_rejects_bad_factors():
    with pytest.raises(ValueError):
        money_band_market(token="ZQX", mid=100.0, volatility=0.5, knowability_min=0.0,
                          lower_factor=1.1, upper_factor=1.2)  # lower not < 1
