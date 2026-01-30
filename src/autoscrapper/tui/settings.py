from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Footer, Input, Static

from .common import AppScreen, MessageScreen
from ..config import (
    ScanSettings,
    config_path,
    load_scan_settings,
    reset_scan_settings,
    save_scan_settings,
)
from ..interaction.ui_windows import SCROLL_CLICKS_PER_PAGE


class ScanConfigScreen(AppScreen):
    DEFAULT_CSS = """
    ScanConfigScreen {
        padding: 1 2;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin: 1 0 0 0;
    }

    .field-row {
        margin: 0 0 1 0;
    }

    .field-label {
        width: 28;
        color: $text-muted;
    }

    .hint {
        color: $text-muted;
    }

    #config-actions {
        margin-top: 1;
        height: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.settings = load_scan_settings()

    def compose(self) -> ComposeResult:
        yield Static("Scan Configuration", classes="menu-title")

        with Vertical():
            yield Static("Pages", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Static("Manual pages", classes="field-label")
                yield Checkbox(id="pages-manual")
                yield Input(
                    id="pages-count", placeholder="Pages", classes="field-input"
                )

            yield Static("Scrolling", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Static("Use default scroll clicks", classes="field-label")
                yield Checkbox(id="scroll-default")
                yield Input(
                    id="scroll-clicks",
                    placeholder=f"Default ({SCROLL_CLICKS_PER_PAGE})",
                    classes="field-input",
                )

            yield Static("OCR", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Static("OCR retries (0 disables)", classes="field-label")
                yield Input(id="ocr-retries", classes="field-input")
            with Horizontal(classes="field-row"):
                yield Static("OCR retry delay (ms)", classes="field-label")
                yield Input(id="ocr-delay", classes="field-input")

            yield Static("Diagnostics", classes="section-title")
            with Horizontal(classes="field-row"):
                yield Static("Debug OCR", classes="field-label")
                yield Checkbox(id="debug-ocr")
            with Horizontal(classes="field-row"):
                yield Static("Profile timing", classes="field-label")
                yield Checkbox(id="profile-timing")

            yield Static(
                Text(f"Config file: {config_path()}", style="dim"),
                classes="hint",
            )

        with Horizontal(id="config-actions"):
            yield Button("Save", id="save", variant="primary")
            yield Button("Reset to defaults", id="reset", variant="warning")
            yield Button("Back", id="back")
        yield Footer()

    def on_mount(self) -> None:
        self._load_into_fields()

    def _load_into_fields(self) -> None:
        settings = self.settings
        self.query_one("#pages-manual", Checkbox).value = (
            settings.pages_mode == "manual"
        )
        pages_input = self.query_one("#pages-count", Input)
        pages_input.value = "" if settings.pages is None else str(settings.pages)
        pages_input.disabled = settings.pages_mode != "manual"

        scroll_default = self.query_one("#scroll-default", Checkbox)
        scroll_default.value = settings.scroll_clicks_per_page is None
        scroll_input = self.query_one("#scroll-clicks", Input)
        scroll_input.value = (
            ""
            if settings.scroll_clicks_per_page is None
            else str(settings.scroll_clicks_per_page)
        )
        scroll_input.disabled = scroll_default.value

        self.query_one("#ocr-retries", Input).value = str(
            settings.ocr_unreadable_retries
        )
        self.query_one("#ocr-delay", Input).value = str(
            settings.ocr_unreadable_retry_delay_ms
        )
        self.query_one("#debug-ocr", Checkbox).value = settings.debug_ocr
        self.query_one("#profile-timing", Checkbox).value = settings.profile

    def _save(self) -> None:
        pages_manual = self.query_one("#pages-manual", Checkbox).value
        pages_input = self.query_one("#pages-count", Input).value.strip()
        pages_mode = "manual" if pages_manual else "auto"
        pages_value = None
        if pages_manual:
            if not pages_input.isdigit() or int(pages_input) < 1:
                self.app.push_screen(
                    MessageScreen("Enter a valid number of pages (>= 1).")
                )
                return
            pages_value = int(pages_input)

        scroll_default = self.query_one("#scroll-default", Checkbox).value
        scroll_input = self.query_one("#scroll-clicks", Input).value.strip()
        scroll_value = None
        if not scroll_default:
            if not scroll_input.isdigit() or int(scroll_input) < 0:
                self.app.push_screen(
                    MessageScreen("Enter a valid scroll click count (>= 0).")
                )
                return
            scroll_value = int(scroll_input)

        retries_raw = self.query_one("#ocr-retries", Input).value.strip()
        delay_raw = self.query_one("#ocr-delay", Input).value.strip()
        if not retries_raw.isdigit() or int(retries_raw) < 0:
            self.app.push_screen(MessageScreen("Enter a valid OCR retry count (>= 0)."))
            return
        if not delay_raw.isdigit() or int(delay_raw) < 0:
            self.app.push_screen(
                MessageScreen("Enter a valid OCR retry delay (>= 0 ms).")
            )
            return

        self.settings = ScanSettings(
            pages_mode=pages_mode,
            pages=pages_value,
            scroll_clicks_per_page=scroll_value,
            ocr_unreadable_retries=int(retries_raw),
            ocr_unreadable_retry_delay_ms=int(delay_raw),
            debug_ocr=self.query_one("#debug-ocr", Checkbox).value,
            profile=self.query_one("#profile-timing", Checkbox).value,
        )
        save_scan_settings(self.settings)
        self.app.push_screen(MessageScreen("Scan settings saved."))

    def _reset(self) -> None:
        reset_scan_settings()
        self.settings = load_scan_settings()
        self._load_into_fields()
        self.app.push_screen(MessageScreen("Settings reset to defaults."))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "pages-manual":
            pages_input = self.query_one("#pages-count", Input)
            pages_input.disabled = not event.checkbox.value
        if event.checkbox.id == "scroll-default":
            scroll_input = self.query_one("#scroll-clicks", Input)
            scroll_input.disabled = event.checkbox.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "save":
            self._save()
        elif button_id == "reset":
            self._reset()
        elif button_id == "back":
            self.app.pop_screen()
