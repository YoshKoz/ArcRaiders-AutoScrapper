from __future__ import annotations

import sys
from typing import Optional

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .key_reader import key_reader


def interactive_menu_supported(console: Console) -> bool:
    return console.is_terminal and sys.stdin.isatty() and sys.stdout.isatty()


def menu_help_text(actions: dict[str, str]) -> Text:
    base = "↑/↓ move • Enter select • number/letter jump"
    extras = []
    if "b" in actions:
        extras.append("b back")
    if "q" in actions:
        extras.append("q quit")
    if extras:
        base = f"{base} • {' • '.join(extras)}"
    return Text(base, style="dim")


def render_menu_panel(
    title: str,
    actions: dict[str, str],
    *,
    default_key: str,
    selected_key: Optional[str] = None,
    border_style: str = "cyan",
    help_text: Optional[Text] = None,
) -> Panel:
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=4)
    table.add_column()

    for key, label in actions.items():
        is_default = key == default_key
        is_selected = key == selected_key
        key_style = "bold yellow" if is_default else "bold cyan"
        label_text = Text(label, style="bold" if is_default else "")
        if is_default:
            label_text.append("  (recommended)", style="dim")
        row_style = "reverse" if is_selected else None
        table.add_row(Text(key, style=key_style), label_text, style=row_style)

    body = Group(table, help_text) if help_text else table
    return Panel(
        body,
        title=Text(title, style="bold"),
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def choose_menu_action(
    console: Console,
    title: str,
    actions: dict[str, str],
    *,
    default_key: str,
    prompt: str,
    border_style: str = "cyan",
    header: Optional[RenderableType] = None,
    help_text: Optional[Text] = None,
    transient: bool = True,
) -> str:
    if not actions:
        return default_key

    if not interactive_menu_supported(console):
        if header:
            console.print(header)
        console.print(
            render_menu_panel(
                title,
                actions,
                default_key=default_key,
                border_style=border_style,
            )
        )
        raw = Prompt.ask(prompt, default=default_key)
        return raw.strip().lower()

    keys = list(actions.keys())
    selected_index = keys.index(default_key) if default_key in keys else 0
    resolved_help = help_text or menu_help_text(actions)

    def _render(selected_key: str) -> RenderableType:
        panel = render_menu_panel(
            title,
            actions,
            default_key=default_key,
            selected_key=selected_key,
            border_style=border_style,
            help_text=resolved_help,
        )
        return Group(header, panel) if header else panel

    with key_reader() as read_key:
        with Live(
            _render(keys[selected_index]),
            console=console,
            refresh_per_second=20,
            transient=transient,
        ) as live:
            while True:
                key = read_key()
                if key.name == "UP":
                    selected_index = (selected_index - 1) % len(keys)
                elif key.name == "DOWN":
                    selected_index = (selected_index + 1) % len(keys)
                elif key.name == "HOME":
                    selected_index = 0
                elif key.name == "END":
                    selected_index = len(keys) - 1
                elif key.name == "ENTER":
                    return keys[selected_index]
                elif key.name == "ESC":
                    if "b" in actions:
                        return "b"
                    if "q" in actions:
                        return "q"
                    return keys[selected_index]
                elif key.name == "CHAR" and key.char:
                    char = key.char.lower()
                    if char in actions:
                        return char
                live.update(_render(keys[selected_index]), refresh=True)
