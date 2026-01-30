from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from rich.console import Console as RichConsole
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.errors import SuspendNotSupported
from textual.screen import Screen
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

from ..cli import config as config_cli
from ..cli import scan as scan_cli
from ..cli.home import _reset_progress, _reset_rules
from ..cli.progress_flow import (
    edit_saved_workshops,
    generate_rules_from_saved_progress,
    review_saved_quests,
    run_progress_wizard,
    run_update_data,
)
from ..items.rules_viewer import run_rules_viewer
from .status import build_status_panel, has_progress


MenuAction = Callable[["MenuScreen"], None]


@dataclass(frozen=True)
class MenuItem:
    key: str
    label: str
    action: MenuAction


class StatusPanel(Static):
    def refresh_status(self) -> None:
        self.update(build_status_panel())

    def on_mount(self) -> None:
        self.refresh_status()


class MenuScreen(Screen):
    def __init__(
        self,
        title: str,
        items: Iterable[MenuItem],
        *,
        default_key: str,
        recommended_key: Optional[str] = None,
        show_status: bool = False,
    ) -> None:
        super().__init__()
        self.title = title
        self.items = list(items)
        self.default_key = default_key
        self.recommended_key = recommended_key
        self.show_status = show_status
        self._actions: dict[str, MenuItem] = {}
        self._keys: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-root"):
            if self.show_status:
                yield StatusPanel(id="status")
            yield Static(self.title, classes="menu-title")
            yield OptionList(id="menu")
        yield Footer()

    def on_mount(self) -> None:
        self._render_menu()
        self._focus_menu()

    def on_screen_resume(self, _event: events.ScreenResume) -> None:
        if self.show_status:
            status = self.query_one(StatusPanel)
            status.refresh_status()
        self._render_menu()
        self._focus_menu()

    def _focus_menu(self) -> None:
        menu = self.query_one(OptionList)
        menu.focus()

    def _render_menu(self) -> None:
        self._actions = {item.key: item for item in self.items}
        self._keys = [item.key for item in self.items]
        menu = self.query_one(OptionList)
        menu.set_options([self._build_option(item) for item in self.items])
        self._highlight_default()

    def _build_option(self, item: MenuItem) -> Option:
        text = Text.assemble((item.key, "bold cyan"), " ", item.label)
        if self.recommended_key and item.key == self.recommended_key:
            text.append("  (recommended)", style="dim")
        return Option(text, id=item.key)

    def _highlight_default(self) -> None:
        menu = self.query_one(OptionList)
        try:
            index = self._keys.index(self.default_key)
        except ValueError:
            index = 0
        menu.highlighted = index

    def on_key(self, event: events.Key) -> None:
        if event.key in {"escape"}:
            if "b" in self._actions:
                self._select_key("b")
                event.stop()
                return
        if event.key.isalnum():
            key = event.key.lower()
            if key in self._actions:
                self._select_key(key)
                event.stop()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        option_id = event.option_id
        if option_id and option_id in self._actions:
            self._actions[option_id].action(self)

    def _select_key(self, key: str) -> None:
        if key not in self._actions:
            return
        menu = self.query_one(OptionList)
        try:
            index = self._keys.index(key)
        except ValueError:
            return
        menu.highlighted = index
        menu.action_select()

    def run_with_suspend(self, func: Callable[[], object]) -> None:
        try:
            with self.app.suspend():
                func()
        except SuspendNotSupported:
            func()


class HomeScreen(MenuScreen):
    def __init__(self) -> None:
        super().__init__(
            "Main menu",
            [],
            default_key="1",
            recommended_key=None,
            show_status=True,
        )

    def _refresh_items(self) -> None:
        recommended = "2" if not has_progress() else "1"
        self.default_key = recommended
        self.recommended_key = recommended
        self.items = [
            MenuItem("1", "Scan", lambda screen: screen.app.push_screen(screen.app._scan_menu())),
            MenuItem(
                "2",
                "Generate Personalized Rule List (Quests / Workshop Level)",
                lambda screen: screen.app.push_screen(screen.app._progress_menu()),
            ),
            MenuItem("3", "Rules", lambda screen: screen.app.push_screen(screen.app._rules_menu())),
            MenuItem("4", "Settings", lambda screen: screen.app.push_screen(screen.app._settings_menu())),
            MenuItem(
                "5",
                "Maintenance",
                lambda screen: screen.app.push_screen(screen.app._maintenance_menu()),
            ),
            MenuItem("q", "Quit", lambda screen: screen.app.exit()),
        ]

    def on_mount(self) -> None:
        self._refresh_items()
        super().on_mount()

    def on_screen_resume(self, event: events.ScreenResume) -> None:
        self._refresh_items()
        super().on_screen_resume(event)


class AutoScrapperApp(App[None]):
    CSS_PATH = "app.tcss"
    TITLE = "Autoscrapper"

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())

    def _scan_menu(self) -> MenuScreen:
        items = [
            MenuItem("1", "Scan now", lambda screen: _run_scan(screen, dry_run=False)),
            MenuItem("2", "Dry run (no clicks)", lambda screen: _run_scan(screen, dry_run=True)),
            MenuItem("b", "Back", lambda screen: screen.app.pop_screen()),
        ]
        return MenuScreen("Scan", items, default_key="1")

    def _progress_menu(self) -> MenuScreen:
        items = [
            MenuItem("1", "Set up / update progress", _run_progress_wizard),
            MenuItem("2", "Review quests", _run_review_quests),
            MenuItem("3", "Edit workshop levels", _run_edit_workshops),
            MenuItem("4", "Update rules from saved progress", _run_update_rules),
            MenuItem("b", "Back", lambda screen: screen.app.pop_screen()),
        ]
        return MenuScreen("Progress", items, default_key="1")

    def _rules_menu(self) -> MenuScreen:
        items = [
            MenuItem("1", "Review / edit rules", _run_rules_viewer),
            MenuItem("b", "Back", lambda screen: screen.app.pop_screen()),
        ]
        return MenuScreen("Rules", items, default_key="1")

    def _settings_menu(self) -> MenuScreen:
        items = [
            MenuItem("1", "Scan configuration", _run_scan_config),
            MenuItem("b", "Back", lambda screen: screen.app.pop_screen()),
        ]
        return MenuScreen("Settings", items, default_key="1")

    def _maintenance_menu(self) -> MenuScreen:
        items = [
            MenuItem("1", "Update game data snapshot", _run_update_snapshot),
            MenuItem("2", "Reset saved progress", _run_reset_progress),
            MenuItem("3", "Reset rules to default", _run_reset_rules),
            MenuItem("b", "Back", lambda screen: screen.app.pop_screen()),
        ]
        return MenuScreen("Maintenance", items, default_key="1")


def _console() -> RichConsole:
    return RichConsole()


def _run_scan(screen: MenuScreen, *, dry_run: bool) -> None:
    def _work() -> None:
        args = ["--dry-run"] if dry_run else []
        scan_cli.main(args)

    screen.run_with_suspend(_work)


def _run_progress_wizard(screen: MenuScreen) -> None:
    screen.run_with_suspend(lambda: run_progress_wizard(_console()))


def _run_review_quests(screen: MenuScreen) -> None:
    screen.run_with_suspend(lambda: review_saved_quests(_console()))


def _run_edit_workshops(screen: MenuScreen) -> None:
    screen.run_with_suspend(lambda: edit_saved_workshops(_console()))


def _run_update_rules(screen: MenuScreen) -> None:
    screen.run_with_suspend(lambda: generate_rules_from_saved_progress(_console()))


def _run_rules_viewer(screen: MenuScreen) -> None:
    screen.run_with_suspend(lambda: run_rules_viewer(_console()))


def _run_scan_config(screen: MenuScreen) -> None:
    screen.run_with_suspend(lambda: config_cli.main([]))


def _run_update_snapshot(screen: MenuScreen) -> None:
    screen.run_with_suspend(lambda: run_update_data(_console()))


def _run_reset_progress(screen: MenuScreen) -> None:
    screen.run_with_suspend(lambda: _reset_progress(_console()))


def _run_reset_rules(screen: MenuScreen) -> None:
    screen.run_with_suspend(lambda: _reset_rules(_console()))


def run_tui() -> int:
    app = AutoScrapperApp()
    app.run()
    return 0
