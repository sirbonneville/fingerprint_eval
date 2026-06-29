"""Gensyn brand theme loader for terminal output (dashboard-dark subset).

Loads a bundled slim theme derived from https://brand.gensyn.ai/brand/manifest.json
(data-dashboard / dashboard-dark). Proprietary fonts (Mondwest, Aux Mono) are never
used — system monospace only.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Dict, Optional, Tuple

MANIFEST_URL = "https://brand.gensyn.ai/brand/manifest.json"
_THEME_FILE = "dashboard-dark.json"


@dataclass(frozen=True)
class Theme:
    """Terminal-facing color tokens."""

    background: str
    surface: str
    surface_elevated: str
    surface_muted: str
    text: str
    text_muted: str
    accent: str
    accent_hover: str
    border: str
    border_strong: str
    button_text: str
    button_bg: str
    success: str
    warning: str
    error: str
    ui_font: str
    source: str
    theme_id: str

    @classmethod
    def from_dict(cls, data: dict) -> "Theme":
        c = data["colors"]
        return cls(
            background=c["background"],
            surface=c["surface"],
            surface_elevated=c["surfaceElevated"],
            surface_muted=c["surfaceMuted"],
            text=c["text"],
            text_muted=c["textMuted"],
            accent=c["accent"],
            accent_hover=c["accentHover"],
            border=c.get("border", "#fad7d133"),
            border_strong=c.get("borderStrong", "#fad7d173"),
            button_text=c["buttonText"],
            button_bg=c["buttonBg"],
            success=c.get("success", "#86efac"),
            warning=c.get("warning", "#fcd34d"),
            error=c.get("error", "#f87171"),
            ui_font=data.get("typography", {}).get(
                "ui",
                "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
            ),
            source=data.get("source", MANIFEST_URL),
            theme_id=data.get("themeId", "dashboard-dark"),
        )


def plain_mode(force: Optional[bool] = None) -> bool:
    """True when color/boxes should be disabled (CI, pipes, --plain, NO_COLOR)."""
    if force is not None:
        return force
    if os.environ.get("FINGERPRINT_PLAIN", "").strip().lower() in ("1", "true", "yes"):
        return True
    if os.environ.get("NO_COLOR", "").strip():
        return True
    return False


def supports_color(stream=None) -> bool:
    if plain_mode():
        return False
    stream = stream or sys.stdout
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return True


def _parse_hex(hex_color: str) -> Tuple[int, int, int]:
    s = hex_color.strip().lstrip("#")
    if len(s) == 8:  # rgba shorthand from manifest border tokens
        s = s[:6]
    if len(s) != 6:
        raise ValueError("expected #RRGGBB hex, got %r" % hex_color)
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def ansi_fg(hex_color: str) -> str:
    r, g, b = _parse_hex(hex_color)
    return "\033[38;2;%d;%d;%dm" % (r, g, b)


def ansi_bg(hex_color: str) -> str:
    r, g, b = _parse_hex(hex_color)
    return "\033[48;2;%d;%d;%dm" % (r, g, b)


ANSI_RESET = "\033[0m"


def style(text: str, *, fg: Optional[str] = None, bg: Optional[str] = None, bold: bool = False) -> str:
    if not supports_color():
        return text
    parts = []
    if bold:
        parts.append("\033[1m")
    if fg:
        parts.append(ansi_fg(fg))
    if bg:
        parts.append(ansi_bg(bg))
    if not parts:
        return text
    return "".join(parts) + text + ANSI_RESET


@lru_cache(maxsize=1)
def load_theme() -> Theme:
    raw = resources.files("fingerprint_eval.brand_data").joinpath(_THEME_FILE).read_text(
        encoding="utf-8"
    )
    return Theme.from_dict(json.loads(raw))


def refresh_theme_from_manifest(url: str = MANIFEST_URL, timeout: float = 30.0) -> dict:
    """Fetch live manifest and extract dashboard-dark tokens (maintenance helper).

    Returns the slim dict; does not write to disk automatically.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "fingerprint_eval/brand"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        manifest = json.load(resp)

    theme = None
    for t in manifest.get("designSystem", {}).get("themes", []):
        if t.get("id") == "dashboard-dark":
            theme = t
            break
    if theme is None:
        raise KeyError("dashboard-dark theme not found in manifest")

    tokens = theme.get("tokens", {})
    colors: Dict[str, str] = {}
    key_map = {
        "color.background": "background",
        "color.surface": "surface",
        "color.surfaceElevated": "surfaceElevated",
        "color.surfaceMuted": "surfaceMuted",
        "color.text": "text",
        "color.textMuted": "textMuted",
        "color.accent": "accent",
        "color.accentHover": "accentHover",
        "color.border": "border",
        "color.borderStrong": "borderStrong",
        "color.buttonText": "buttonText",
        "color.buttonBg": "buttonBg",
    }
    for src, dst in key_map.items():
        entry = tokens.get(src, {})
        val = entry.get("value", "")
        if isinstance(val, str) and val.startswith("#"):
            colors[dst] = val
        elif isinstance(val, str) and val.startswith("rgba"):
            m = re.match(
                r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9.]+)\s*\)", val
            )
            if m:
                r, g, b, a = m.groups()
                alpha = int(float(a) * 255)
                colors[dst] = "#%02x%02x%02x%02x" % (int(r), int(g), int(b), alpha)

    return {
        "source": url,
        "manifestVersion": manifest.get("meta", {}).get("version"),
        "ruleId": "data-dashboard",
        "themeId": "dashboard-dark",
        "updatedAt": manifest.get("meta", {}).get("updatedAt"),
        "colors": {
            **colors,
            "success": "#86efac",
            "warning": "#fcd34d",
            "error": "#f87171",
        },
        "typography": {"ui": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"},
    }
