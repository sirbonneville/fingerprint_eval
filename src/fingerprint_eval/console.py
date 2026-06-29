"""Branded terminal output for fingerprint_eval (Gensyn dashboard-dark TUI skin).

Uses Rich when installed (`pip install fingerprint_eval[cli]`); falls back to ANSI
or plain text. Always honour ``--plain``, ``NO_COLOR``, and non-TTY streams.
"""
from __future__ import annotations

import sys
from typing import Iterable, List, Optional, Sequence, Tuple

from . import brand

try:
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def _short_model(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


class BatteryUI:
    """Progress + summary chrome for ``fingerprint_eval.battery``."""

    def __init__(self, *, plain: bool = False, stream=None):
        self.plain = brand.plain_mode(plain)
        self.theme = brand.load_theme()
        self.stream = stream or sys.stdout
        self._rich: Optional[Console] = None
        if _HAS_RICH and not self.plain and brand.supports_color(self.stream):
            self._rich = Console(file=self.stream, highlight=False, soft_wrap=True)

    def _println(self, line: str = "") -> None:
        print(line, file=self.stream)

    def banner(
        self,
        *,
        total: int,
        session: str,
        models: Sequence[str],
        memory: str,
        n: int,
        dry_run: bool,
        synthetic: bool,
        concurrency: int,
    ) -> None:
        title = "fingerprint_eval"
        subtitle = "prediction-market fingerprint battery"
        arm = "live board (synthetic flow)" if synthetic else "frozen board (control)"
        stats = [
            ("cells", str(total)),
            ("runs/cell", str(n)),
            ("memory", memory),
            ("arm", arm),
            ("concurrency", str(concurrency)),
            ("dry_run", str(dry_run)),
        ]
        model_line = ", ".join(_short_model(m) for m in models)

        if self._rich is not None:
            t = self.theme
            grid = Table.grid(padding=(0, 2))
            grid.add_column(style=t.text_muted, justify="right")
            grid.add_column(style=t.text)
            for k, v in stats:
                grid.add_row(k, v)
            grid.add_row("models", model_line)
            grid.add_row("session", session)
            panel = Panel(
                Group(Text(subtitle, style=t.text_muted), "", grid),
                title=Text(title, style=t.accent),
                border_style=t.accent,
                box=box.HEAVY,
                padding=(1, 2),
                subtitle=Text("gensyn dashboard-dark", style=t.text_muted),
            )
            self._rich.print(panel)
            return

        if not self.plain and brand.supports_color(self.stream):
            t = self.theme
            self._println(brand.style("═" * 72, fg=t.border_strong))
            self._println(brand.style("  fingerprint_eval", fg=t.accent, bold=True))
            self._println(brand.style("  " + subtitle, fg=t.text_muted))
            self._println(brand.style("  session: " + session, fg=t.text))
            self._println(
                brand.style(
                    "  %d cells · N=%d · memory=%s · %s" % (total, n, memory, arm),
                    fg=t.text_muted,
                )
            )
            self._println(brand.style("  models: " + model_line, fg=t.text))
            self._println(brand.style("═" * 72, fg=t.border_strong))
            self._println()
            return

        self._println("BATTERY: %d cells -> %s" % (total, session))
        self._println(
            "  models=%s  memory=%s  N=%d  dry_run=%s  synthetic=%s"
            % (list(models), memory, n, dry_run, synthetic)
        )
        self._println()

    def cell_skip(self, prefix: str) -> None:
        if self._rich is not None:
            self._rich.print(
                Text(prefix + " … SKIP (complete)", style=self.theme.text_muted)
            )
            return
        if not self.plain and brand.supports_color(self.stream):
            self._println(
                brand.style(prefix + " ... SKIP (already complete)", fg=self.theme.text_muted)
            )
            return
        self._println("%s ... SKIP (already complete)" % prefix)

    def cell_done(self, prefix: str, calib_status: str, elapsed_min: float) -> None:
        tail = "calib %s, done (%.1f min)" % (calib_status, elapsed_min)
        if self._rich is not None:
            self._rich.print(
                Text.assemble(
                    (prefix + " … ", self.theme.text),
                    ("✓ ", self.theme.success),
                    (tail, self.theme.accent),
                )
            )
            return
        if not self.plain and brand.supports_color(self.stream):
            self._println(
                brand.style(prefix + " ... ", fg=self.theme.text)
                + brand.style(tail, fg=self.theme.accent)
            )
            return
        self._println("%s ... %s" % (prefix, tail))

    def cell_running(self, prefix: str) -> None:
        """Optional: emit before a long cell starts (visible with concurrency)."""
        if self._rich is not None:
            self._rich.print(Text(prefix + " … running", style=self.theme.text_muted))
            return
        if not self.plain and brand.supports_color(self.stream):
            self._println(brand.style(prefix + " ... running", fg=self.theme.text_muted))
            return
        self._println("%s ... running" % prefix)

    def complete(
        self,
        *,
        session: str,
        manifest_path: str,
        ran: int,
        skipped: int,
        total: int,
    ) -> None:
        if self._rich is not None:
            t = self.theme
            table = Table(show_header=False, box=box.HEAVY, border_style=t.accent, padding=(0, 1))
            table.add_column(style=t.text_muted)
            table.add_column(style=t.text)
            table.add_row("completed", str(ran))
            table.add_row("skipped", str(skipped))
            table.add_row("total", str(total))
            table.add_row("session", session)
            table.add_row("manifest", manifest_path)
            self._rich.print(
                Panel(
                    table,
                    title=Text("battery complete", style=t.accent),
                    border_style=t.accent,
                    padding=(1, 2),
                )
            )
            return

        if not self.plain and brand.supports_color(self.stream):
            t = self.theme
            self._println()
            self._println(brand.style("BATTERY COMPLETE", fg=t.accent, bold=True))
            self._println(
                brand.style(
                    "  ran=%d  skipped=%d  total=%d" % (ran, skipped, total),
                    fg=t.text_muted,
                )
            )
            self._println(brand.style("  %s" % session, fg=t.text))
            self._println(brand.style("  %s" % manifest_path, fg=t.text_muted))
            return

        self._println("\nBATTERY COMPLETE: %s" % session)
        self._println("Index: %s" % manifest_path)


def analysis_header(title: str, lines: Sequence[str], *, plain: bool = False, stream=None) -> None:
    """Branded section header for analysis scripts."""
    ui = BatteryUI(plain=plain, stream=stream)
    if ui._rich is not None:
        from rich.panel import Panel
        from rich.text import Text

        t = ui.theme
        body = Text("\n".join(lines), style=t.text)
        ui._rich.print(
            Panel(
                body,
                title=Text(title, style=t.accent),
                border_style=t.accent,
                padding=(0, 2),
            )
        )
        return
    if not ui.plain and brand.supports_color(stream or sys.stdout):
        t = ui.theme
        ui._println(brand.style(title, fg=t.accent, bold=True))
        for line in lines:
            ui._println(brand.style("  " + line, fg=t.text_muted if line.startswith("session") else t.text))
        ui._println()
        return
    ui._println("\n%s  %s\n" % (title.upper(), "  ".join(lines)))


def stat_cards(rows: Iterable[Tuple[str, str]], *, plain: bool = False, stream=None) -> None:
    """Render a row of metric stat-cards (for analysis script summaries)."""
    items: List[Tuple[str, str]] = list(rows)
    if not items:
        return
    ui = BatteryUI(plain=plain, stream=stream)
    t = ui.theme
    if ui._rich is not None:
        table = Table(show_header=False, box=box.SQUARE, border_style=t.border_strong, expand=True)
        table.add_column(justify="center")
        for label, value in items:
            table.add_column(justify="center")
        labels = [label for label, _ in items]
        values = [value for _, value in items]
        table.add_row(*[Text(l, style=t.text_muted) for l in labels])
        table.add_row(*[Text(v, style=t.accent) for v in values])
        ui._rich.print(table)
        return
    if not ui.plain and brand.supports_color(stream or sys.stdout):
        parts = ["  ".join("%s: %s" % (l, brand.style(v, fg=t.accent)) for l, v in items)]
        print(parts[0], file=stream or sys.stdout)
        return
    print("  ".join("%s=%s" % (l, v) for l, v in items), file=stream or sys.stdout)
