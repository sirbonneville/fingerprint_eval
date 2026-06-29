"""Tests for Gensyn brand theme + console output."""
import json
import os

import pytest

from fingerprint_eval import brand
from fingerprint_eval.console import BatteryUI, stat_cards


def test_load_theme_has_dashboard_colors():
    t = brand.load_theme()
    assert t.theme_id == "dashboard-dark"
    assert t.background == "#230800"
    assert t.accent == "#fad7d1"
    assert t.text == "#e5e5e5"


def test_parse_hex_and_ansi():
    assert brand._parse_hex("#fad7d1") == (250, 215, 209)
    assert "\033[38;2;250;215;209m" in brand.ansi_fg("#fad7d1")


def test_plain_mode_env(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FINGERPRINT_PLAIN", raising=False)
    assert brand.plain_mode(False) is False
    monkeypatch.setenv("FINGERPRINT_PLAIN", "1")
    assert brand.plain_mode() is True
    monkeypatch.delenv("FINGERPRINT_PLAIN", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    assert brand.plain_mode() is True
    assert brand.plain_mode(False) is False


def test_battery_ui_rich_banner_no_crash(monkeypatch):
    """Rich path must not mix Table into Text.assemble (regression)."""
    from fingerprint_eval.console import BatteryUI, _HAS_RICH

    if not _HAS_RICH:
        pytest.skip("rich not installed")
    monkeypatch.setattr(brand, "supports_color", lambda stream=None: True)

    class _TTY:
        def isatty(self):
            return True

        def write(self, _s):
            pass

        def flush(self):
            pass

    ui = BatteryUI(plain=False, stream=_TTY())
    assert ui._rich is not None
    ui.banner(
        total=4,
        session="/tmp/session",
        models=["openai/gpt-4o"],
        memory="both",
        n=30,
        dry_run=True,
        synthetic=False,
        concurrency=1,
    )


def test_battery_ui_plain_banner(capsys):
    ui = BatteryUI(plain=True)
    ui.banner(
        total=4,
        session="/tmp/session",
        models=["openai/gpt-4o"],
        memory="both",
        n=30,
        dry_run=True,
        synthetic=False,
        concurrency=1,
    )
    out = capsys.readouterr().out
    assert "BATTERY: 4 cells" in out
    assert "/tmp/session" in out


def test_battery_ui_plain_complete(capsys):
    ui = BatteryUI(plain=True)
    ui.complete(
        session="/tmp/session",
        manifest_path="/tmp/session/manifest.json",
        ran=2,
        skipped=2,
        total=4,
    )
    out = capsys.readouterr().out
    assert "BATTERY COMPLETE" in out
    assert "manifest.json" in out


def test_stat_cards_plain(capsys):
    stat_cards([("peak", "0.47"), ("buy_flow", "0.46")], plain=True)
    out = capsys.readouterr().out
    assert "peak=0.47" in out


def test_refresh_theme_from_manifest_shape():
    """Offline: validate bundled JSON matches expected schema (no network)."""
    from importlib import resources

    raw = resources.files("fingerprint_eval.brand_data").joinpath("dashboard-dark.json").read_text()
    data = json.loads(raw)
    assert "colors" in data
    assert data["colors"]["background"] == "#230800"
