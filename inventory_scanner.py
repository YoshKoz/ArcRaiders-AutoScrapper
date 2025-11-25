"""
inventory_scanner.py

Scan the 4x6 inventory grid by hovering each cell, opening the context
menu, locating the light infobox (#f9eedf), and OCR-ing the item title.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple, cast

import cv2
import numpy as np
import pywinctl as pwc
import pytesseract
import pydirectinput as pdi
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from grid_navigation import Cell, Grid

try:
    import mss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mss = None

try:
    import pyautogui  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyautogui = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_APP = "PioneerGame.exe"
WINDOW_TIMEOUT = 30.0
WINDOW_POLL_INTERVAL = 0.05

# Infobox visual characteristics
INFOBOX_COLOR_BGR = np.array([223, 238, 249], dtype=np.uint8)  # #f9eedf in BGR
INFOBOX_TOLERANCE = 8
MIN_INFOBOX_WIDTH = 230
MIN_INFOBOX_HEIGHT = 80

# Item title placement inside the infobox (relative to infobox size)
TITLE_HEIGHT_REL = 0.18

# Action rows (Split Stack, Move to Backpack, Inspect, Sell, Recycle)
ACTION_START_REL = 0.24
ACTION_LINE_HEIGHT_REL = 0.11
ACTION_PADDING_X_REL = 0.05
# Confirmation buttons (window-normalized rectangles)
SELL_CONFIRM_RECT_NORM = (0.5047, 0.6941, 0.1791, 0.0531)
RECYCLE_CONFIRM_RECT_NORM = (0.5058, 0.6274, 0.1777, 0.0544)

# Click pacing
ACTION_DELAY = 0.05
MENU_APPEAR_DELAY = 0.05
INFOBOX_RETRY_DELAY = 0.05
INFOBOX_RETRIES = 3
MOVE_DURATION = 0.05
SELL_RECYCLE_SPEED_MULT = 2.25  # extra slack vs default pacing (MOVE_DURATION/ACTION_DELAY)
SELL_RECYCLE_MOVE_DURATION = MOVE_DURATION * SELL_RECYCLE_SPEED_MULT
SELL_RECYCLE_ACTION_DELAY = ACTION_DELAY * SELL_RECYCLE_SPEED_MULT

# Scrolling
# 16 downward scroll clicks advances the list by exactly one 6x4 grid of items.
SCROLL_CLICKS_PER_PAGE = 16
SCROLL_INTERVAL = 0.0
SCROLL_SETTLE_DELAY = 0.05

# Keyboard
VK_ESCAPE = 0x1B
ITEM_ACTIONS_PATH = Path(__file__).resolve().parent / "items" / "items_actions.json"

# Optional user32 handle for escape detection (Windows only)
try:
    _USER32 = ctypes.windll.user32  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - platform dependent
    _USER32 = None


# ---------------------------------------------------------------------------
# Item action definitions
# ---------------------------------------------------------------------------

Decision = Literal["KEEP", "RECYCLE", "SELL", "CRAFTING MATERIAL"]
DecisionList = List[Decision]
ActionMap = Dict[str, DecisionList]


@dataclass
class ItemActionResult:
    page: int
    cell: Cell
    item_name: str
    decision: Optional[Decision]
    action_taken: str
    note: Optional[str] = None


VALID_DECISIONS = {"KEEP", "RECYCLE", "SELL", "CRAFTING MATERIAL"}


# ---------------------------------------------------------------------------
# Escape handling
# ---------------------------------------------------------------------------

def _escape_pressed() -> bool:
    """
    Detect whether Escape is currently pressed (Windows).
    """
    if _USER32 is None:
        return False
    # High bit of GetAsyncKeyState indicates key is down
    return bool(_USER32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)


def _abort_if_escape_pressed() -> None:
    """
    Raise KeyboardInterrupt if Escape is down.
    """
    if _escape_pressed():
        raise KeyboardInterrupt("Escape pressed")


# ---------------------------------------------------------------------------
# Item action helpers
# ---------------------------------------------------------------------------

def _normalize_item_name(name: str) -> str:
    return name.strip().lower()


def load_item_actions(path: Path = ITEM_ACTIONS_PATH) -> ActionMap:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[warn] Item actions file not found at {path}; defaulting to skip actions.")
        return {}
    except json.JSONDecodeError as exc:
        print(f"[warn] Could not parse item actions file {path}: {exc}; defaulting to skip actions.")
        return {}

    if not isinstance(raw, list):
        print(f"[warn] Item actions file {path} must be a JSON array; defaulting to skip actions.")
        return {}

    actions: ActionMap = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        decisions = entry.get("decision")
        if not isinstance(name, str) or not isinstance(decisions, list):
            continue

        normalized_name = _normalize_item_name(name)
        cleaned: DecisionList = []
        for decision in decisions:
            if not isinstance(decision, str):
                continue
            candidate = decision.strip().upper()
            if candidate in VALID_DECISIONS:
                cleaned.append(cast(Decision, candidate))
        if normalized_name and cleaned:
            actions[normalized_name] = cleaned

    return actions


def _choose_decision(item_name: str, actions: ActionMap) -> Tuple[Optional[Decision], Optional[str]]:
    normalized = _normalize_item_name(item_name)
    if not normalized:
        return None, None

    decision_list = actions.get(normalized)
    if not decision_list:
        return None, None

    decision = decision_list[0]
    note = None
    if len(decision_list) > 1:
        note = (
            f"Multiple decisions {decision_list}; chose {decision} "
            "(TODO: revisit multi-decision handling)."
        )

    return decision, note


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def wait_for_target_window(
    target_app: str = TARGET_APP,
    timeout: float = WINDOW_TIMEOUT,
    poll_interval: float = WINDOW_POLL_INTERVAL,
) -> pwc.Window:
    """
    Wait until the active window belongs to the target process.
    """
    start = time.monotonic()
    target_lower = target_app.lower()

    while time.monotonic() - start < timeout:
        _abort_if_escape_pressed()
        win = pwc.getActiveWindow()
        if win is not None:
            app = (win.getAppName() or "").lower()
            if app == target_lower:
                return win
        time.sleep(poll_interval)

    raise TimeoutError(f"Timed out waiting for active window {target_app!r}")


def window_rect(win: pwc.Window) -> Tuple[int, int, int, int]:
    """
    (left, top, width, height) in screen coordinates for the window.
    """
    return int(win.left), int(win.top), int(win.width), int(win.height)


# ---------------------------------------------------------------------------
# Screen capture
# ---------------------------------------------------------------------------

def _capture_with_mss(region: Tuple[int, int, int, int]) -> np.ndarray:
    left, top, width, height = region
    with mss.mss() as sct:
        raw = np.array(
            sct.grab(
                {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }
            )
        )
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)


def _capture_with_pyautogui(region: Tuple[int, int, int, int]) -> np.ndarray:
    left, top, width, height = region
    img = pyautogui.screenshot(region=(left, top, width, height))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def capture_region(region: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Capture a BGR screenshot of the given region (left, top, width, height).
    """
    if mss is not None:
        return _capture_with_mss(region)
    if pyautogui is not None:
        return _capture_with_pyautogui(region)
    raise RuntimeError("Install either 'mss' or 'pyautogui' for screenshots")


# ---------------------------------------------------------------------------
# Infobox + OCR helpers
# ---------------------------------------------------------------------------

def find_infobox(bgr_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate the largest rectangle that matches the infobox background color.
    Returns (x, y, w, h) relative to the provided image, or None if not found.
    """
    kernel = np.ones((3, 3), np.uint8)

    def _find_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= MIN_INFOBOX_WIDTH and h >= MIN_INFOBOX_HEIGHT:
                candidates.append((w * h, (x, y, w, h)))
        if not candidates:
            return None
        _, best_rect = max(candidates, key=lambda item: item[0])
        return best_rect

    # First try exact color match
    mask_exact = (np.all(bgr_image == INFOBOX_COLOR_BGR, axis=2)).astype(np.uint8) * 255
    mask_exact = cv2.morphologyEx(mask_exact, cv2.MORPH_CLOSE, kernel, iterations=1)
    rect = _find_from_mask(mask_exact)
    if rect:
        return rect

    # Fallback to tolerance-based mask
    lower = np.clip(INFOBOX_COLOR_BGR - INFOBOX_TOLERANCE, 0, 255).astype(np.uint8)
    upper = np.clip(INFOBOX_COLOR_BGR + INFOBOX_TOLERANCE, 0, 255).astype(np.uint8)
    mask_tol = cv2.inRange(bgr_image, lower, upper)
    mask_tol = cv2.morphologyEx(mask_tol, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _find_from_mask(mask_tol)


def title_roi(infobox_rect: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Compute the ROI for the title text within the infobox.
    """
    x, y, w, h = infobox_rect
    title_h = int(TITLE_HEIGHT_REL * h)
    return x, y, w, max(1, title_h)


def _rect_center(rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Center (cx, cy) of a rectangle.
    """
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def _normalized_rect_to_window(
    norm_rect: Tuple[float, float, float, float],
    window_width: int,
    window_height: int,
) -> Tuple[int, int, int, int]:
    """
    Scale a normalized rectangle (x,y,w,h in [0,1]) to window-relative pixels.
    """
    nx, ny, nw, nh = norm_rect
    x = int(round(nx * window_width))
    y = int(round(ny * window_height))
    w = max(1, int(round(nw * window_width)))
    h = max(1, int(round(nh * window_height)))
    return x, y, w, h


def _window_relative_to_screen(
    rect: Tuple[int, int, int, int],
    window_left: int,
    window_top: int,
) -> Tuple[int, int, int, int]:
    """
    Convert a window-relative rectangle to absolute screen coordinates.
    """
    x, y, w, h = rect
    return window_left + x, window_top + y, w, h


def sell_confirm_button_rect(
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
) -> Tuple[int, int, int, int]:
    """
    Absolute screen rectangle for the Sell confirmation button.
    """
    rel_rect = _normalized_rect_to_window(SELL_CONFIRM_RECT_NORM, window_width, window_height)
    return _window_relative_to_screen(rel_rect, window_left, window_top)


def recycle_confirm_button_rect(
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
) -> Tuple[int, int, int, int]:
    """
    Absolute screen rectangle for the Recycle confirmation button.
    """
    rel_rect = _normalized_rect_to_window(RECYCLE_CONFIRM_RECT_NORM, window_width, window_height)
    return _window_relative_to_screen(rel_rect, window_left, window_top)


def sell_confirm_button_center(
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
) -> Tuple[int, int]:
    """
    Center of the Sell confirmation button (absolute screen coords).
    """
    return _rect_center(sell_confirm_button_rect(window_left, window_top, window_width, window_height))


def recycle_confirm_button_center(
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
) -> Tuple[int, int]:
    """
    Center of the Recycle confirmation button (absolute screen coords).
    """
    return _rect_center(recycle_confirm_button_rect(window_left, window_top, window_width, window_height))


def _preprocess_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _extract_action_line_bbox(
    ocr_data: Dict[str, List],
    target: Literal["sell", "recycle"],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Given pytesseract image_to_data output, return a bbox (left, top, w, h) for
    the line containing the target action (infobox-relative coords).
    """
    groups: Dict[Tuple[int, int, int, int], List[int]] = {}
    texts = ocr_data.get("text", [])
    n = len(texts)
    for i in range(n):
        raw_text = texts[i] or ""
        cleaned = re.sub(r"[^a-z]", "", raw_text.lower())
        if not cleaned or target not in cleaned:
            continue
        key = (
            int(ocr_data["page_num"][i]),
            int(ocr_data["block_num"][i]),
            int(ocr_data["par_num"][i]),
            int(ocr_data["line_num"][i]),
        )
        groups.setdefault(key, []).append(i)

    if not groups:
        return None

    def _group_score(indices: List[int]) -> float:
        confs = []
        for idx in indices:
            conf_str = ocr_data["conf"][idx]
            try:
                confs.append(float(conf_str))
            except Exception:
                continue
        return sum(confs) / len(confs) if confs else -1.0

    best_key = max(groups.keys(), key=lambda k: _group_score(groups[k]))
    indices = groups[best_key]
    lefts = [int(ocr_data["left"][i]) for i in indices]
    tops = [int(ocr_data["top"][i]) for i in indices]
    rights = [int(ocr_data["left"][i]) + int(ocr_data["width"][i]) for i in indices]
    bottoms = [int(ocr_data["top"][i]) + int(ocr_data["height"][i]) for i in indices]

    x1, y1 = min(lefts), min(tops)
    x2, y2 = max(rights), max(bottoms)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _find_action_bbox_by_ocr(
    infobox_bgr: np.ndarray,
    target: Literal["sell", "recycle"],
) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray]:
    """
    Run OCR over the full infobox to locate the line containing the target
    action. Returns (bbox, processed_image) where bbox is infobox-relative.
    """
    processed = _preprocess_for_ocr(infobox_bgr)
    try:
        data = pytesseract.image_to_data(processed, config="--psm 6", output_type=pytesseract.Output.DICT)
    except Exception:
        return None, processed

    bbox = _extract_action_line_bbox(data, target)
    return bbox, processed


def ocr_item_name(roi_bgr: np.ndarray) -> str:
    """
    OCR the item name from the pre-cropped title ROI.
    """
    if roi_bgr.size == 0:
        return ""

    processed = _preprocess_for_ocr(roi_bgr)
    raw = pytesseract.image_to_string(processed, config="--psm 6")
    return _clean_ocr_text(raw)


# ---------------------------------------------------------------------------
# Navigation + scanning
# ---------------------------------------------------------------------------

def _progress(seq: Iterable[Cell], enabled: bool, total: int) -> Iterable[Cell]:
    """
    Wrap an iterable with tqdm when available and enabled.
    """
    if enabled and tqdm is not None:
        return tqdm(seq, total=total, desc="Scanning grid")
    return seq


def _clean_ocr_text(raw: str) -> str:
    text = " ".join(raw.split())
    text = re.sub(r"[^-A-Za-z0-9 '()\\]+", "", text)
    return text.strip()


def _cell_screen_center(cell: Cell, window_left: int, window_top: int) -> Tuple[int, int]:
    cx, cy = cell.center
    # Game quirk: on the last row the infobox can render off-screen when we click dead-center,
    # hiding Sell/Recycle. Nudge down slightly to keep the infobox fully visible.
    if cell.row == Grid.ROWS - 1:
        cy += 2
    return int(window_left + cx), int(window_top + cy)


def _sleep(duration: float, label: str) -> None:
    """
    Sleep for a specific duration and honor Escape aborts.
    """
    time.sleep(duration)
    _abort_if_escape_pressed()


def _pause_action(label: str = "pause", duration: float = ACTION_DELAY) -> None:
    """
    Standard pause to keep a safe delay between input/processing steps.
    """
    _sleep(duration, label)


def _timed_action(label: str, func, *args, **kwargs) -> None:
    """
    Run an input action while checking for Escape.
    """
    _abort_if_escape_pressed()
    func(*args, **kwargs)


def _click_absolute(x: int, y: int, label: str, pause: float = ACTION_DELAY) -> None:
    _timed_action(label, pdi.leftClick, x, y, _pause=False)
    _pause_action(f"after {label}", duration=pause)


def _click_window_relative(
    x: int,
    y: int,
    window_left: int,
    window_top: int,
    label: str,
    pause: float = ACTION_DELAY,
) -> None:
    _click_absolute(int(window_left + x), int(window_top + y), label, pause=pause)


def _move_absolute(
    x: int,
    y: int,
    label: str,
    duration: float = MOVE_DURATION,
    pause: float = ACTION_DELAY,
) -> None:
    _timed_action(f"{label} moveTo", pdi.moveTo, x, y, duration=duration)
    _pause_action(f"after {label} move", duration=pause)


def _move_window_relative(
    x: int,
    y: int,
    window_left: int,
    window_top: int,
    label: str,
    duration: float = MOVE_DURATION,
    pause: float = ACTION_DELAY,
) -> None:
    _move_absolute(int(window_left + x), int(window_top + y), label, duration=duration, pause=pause)


def _perform_sell(
    infobox_rect: Tuple[int, int, int, int],
    action_bbox_rel: Tuple[int, int, int, int],
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
) -> None:
    bx, by, bw, bh = action_bbox_rel
    sell_bbox_win = (infobox_rect[0] + bx, infobox_rect[1] + by, bw, bh)
    sx, sy = _rect_center(sell_bbox_win)
    _move_window_relative(
        sx,
        sy,
        window_left,
        window_top,
        "sell",
        duration=SELL_RECYCLE_MOVE_DURATION,
        pause=SELL_RECYCLE_ACTION_DELAY,
    )
    _click_window_relative(
        sx,
        sy,
        window_left,
        window_top,
        "sell",
        pause=SELL_RECYCLE_ACTION_DELAY,
    )
    _sleep(MENU_APPEAR_DELAY, "after sell action")

    cx, cy = sell_confirm_button_center(window_left, window_top, window_width, window_height)
    _move_absolute(
        cx,
        cy,
        "sell confirm",
        duration=SELL_RECYCLE_MOVE_DURATION,
        pause=SELL_RECYCLE_ACTION_DELAY,
    )
    _click_absolute(cx, cy, "sell confirm", pause=SELL_RECYCLE_ACTION_DELAY)


def _perform_recycle(
    infobox_rect: Tuple[int, int, int, int],
    action_bbox_rel: Tuple[int, int, int, int],
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
) -> None:
    bx, by, bw, bh = action_bbox_rel
    recycle_bbox_win = (infobox_rect[0] + bx, infobox_rect[1] + by, bw, bh)
    rx, ry = _rect_center(recycle_bbox_win)
    _move_window_relative(
        rx,
        ry,
        window_left,
        window_top,
        "recycle",
        duration=SELL_RECYCLE_MOVE_DURATION,
        pause=SELL_RECYCLE_ACTION_DELAY,
    )
    _click_window_relative(
        rx,
        ry,
        window_left,
        window_top,
        "recycle",
        pause=SELL_RECYCLE_ACTION_DELAY,
    )
    _sleep(MENU_APPEAR_DELAY, "after recycle action")

    cx, cy = recycle_confirm_button_center(window_left, window_top, window_width, window_height)
    _move_absolute(
        cx,
        cy,
        "recycle confirm",
        duration=SELL_RECYCLE_MOVE_DURATION,
        pause=SELL_RECYCLE_ACTION_DELAY,
    )
    _click_absolute(cx, cy, "recycle confirm", pause=SELL_RECYCLE_ACTION_DELAY)


def _open_cell_menu(cell: Cell, window_left: int, window_top: int) -> None:
    """
    Hover the cell, then left-click and right-click to open its context menu.
    """
    _abort_if_escape_pressed()
    cx, cy = _cell_screen_center(cell, window_left, window_top)
    _timed_action("moveTo", pdi.moveTo, cx, cy, duration=MOVE_DURATION)
    _pause_action("after move")
    _timed_action("leftClick", pdi.leftClick, cx, cy, _pause=False)
    _pause_action("after leftClick")
    _timed_action("rightClick", pdi.rightClick, cx, cy, _pause=False)
    _pause_action("after rightClick")


def _scroll_to_next_grid(scroll_clicks_per_page: int = SCROLL_CLICKS_PER_PAGE) -> None:
    """
    Scroll quickly to reveal the next 6x4 grid of items.
    """
    _abort_if_escape_pressed()
    clicks = -abs(scroll_clicks_per_page)
    _timed_action("scroll", pdi.scroll, clicks, _pause=False, interval=SCROLL_INTERVAL)
    _sleep(SCROLL_SETTLE_DELAY, "after scroll")


def scan_inventory(
    window_timeout: float = WINDOW_TIMEOUT,
    infobox_retries: int = INFOBOX_RETRIES,
    show_progress: bool = True,
    pages: int = 1,
    scroll_clicks_per_page: int = SCROLL_CLICKS_PER_PAGE,
    apply_actions: bool = True,
    actions_path: Path = ITEM_ACTIONS_PATH,
    actions_override: Optional[ActionMap] = None,
) -> List[ItemActionResult]:
    """
    Walk each 6x4 grid (top-to-bottom, left-to-right), OCR each cell's item
    title, and apply the configured keep/recycle/sell decision when possible.
    Decisions come from items_actions.json unless an override map is provided.
    """
    if pages < 1:
        raise ValueError("pages must be >= 1")

    print("waiting for Arc Raiders to be active window...", flush=True)
    window = wait_for_target_window(timeout=window_timeout)
    win_left, win_top, win_width, win_height = window_rect(window)

    actions: ActionMap = {}
    if apply_actions:
        actions = actions_override if actions_override is not None else load_item_actions(actions_path)

    grid = Grid()
    cells = list(grid)
    cells_per_page = len(cells)
    total_cells = cells_per_page * pages
    page_cells = [(page, cell) for page in range(pages) for cell in cells]
    results: List[ItemActionResult] = []

    _abort_if_escape_pressed()

    if not cells:
        return results

    progress = tqdm(total=total_cells, desc="Scanning grid") if show_progress and tqdm is not None else None
    current_page = -1
    idx = 0

    try:
        while idx < len(page_cells):
            page, cell = page_cells[idx]
            if page != current_page:
                if current_page != -1:
                    _scroll_to_next_grid(scroll_clicks_per_page)
                _open_cell_menu(cell, win_left, win_top)
                current_page = page

            global_idx = page * cells_per_page + cell.index

            _abort_if_escape_pressed()
            if hasattr(window, "isAlive") and not window.isAlive:  # type: ignore[attr-defined]
                raise RuntimeError("Target window closed during scan")

            time.sleep(MENU_APPEAR_DELAY)
            _pause_action()

            infobox_rect: Optional[Tuple[int, int, int, int]] = None
            window_bgr: Optional[np.ndarray] = None
            tries = 0

            for tries in range(infobox_retries):
                _abort_if_escape_pressed()
                window_bgr = capture_region((win_left, win_top, win_width, win_height))
                infobox_rect = find_infobox(window_bgr)
                if infobox_rect:
                    break
                time.sleep(INFOBOX_RETRY_DELAY)
                _pause_action()

            item_name = ""
            processed_title: Optional[np.ndarray] = None
            infobox_crop: Optional[np.ndarray] = None
            if infobox_rect and window_bgr is not None:
                _pause_action()
                title_x, title_y, title_w, title_h = title_roi(infobox_rect)
                title_crop = window_bgr[title_y:title_y + title_h, title_x:title_x + title_w]
                processed_title = _preprocess_for_ocr(title_crop)
                raw = pytesseract.image_to_string(processed_title, config="--psm 6")
                item_name = _clean_ocr_text(raw)
                x, y, w, h = infobox_rect
                infobox_crop = window_bgr[y:y + h, x:x + w]

            decision: Optional[Decision] = None
            decision_note: Optional[str] = None
            action_taken = "SCAN_ONLY" if not apply_actions else ("SKIP_NO_ACTION_MAP" if not actions else "SKIP_UNLISTED")

            if apply_actions:
                if actions and item_name:
                    decision, decision_note = _choose_decision(item_name, actions)
                    if decision is None:
                        action_taken = "SKIP_UNLISTED"
                    elif decision in {"KEEP", "CRAFTING MATERIAL"}:
                        action_taken = decision
                    elif decision == "SELL":
                        if infobox_rect is not None and infobox_crop is not None:
                            sell_bbox_rel, _ = _find_action_bbox_by_ocr(infobox_crop, "sell")
                            if sell_bbox_rel is None:
                                action_taken = "SKIP_NO_ACTION_BBOX"
                            else:
                                _perform_sell(infobox_rect, sell_bbox_rel, win_left, win_top, win_width, win_height)
                                action_taken = "SELL"
                        else:
                            action_taken = "SKIP_NO_INFOBOX"
                    elif decision == "RECYCLE":
                        if infobox_rect is not None and infobox_crop is not None:
                            recycle_bbox_rel, _ = _find_action_bbox_by_ocr(infobox_crop, "recycle")
                            if recycle_bbox_rel is None:
                                action_taken = "SKIP_NO_ACTION_BBOX"
                            else:
                                _perform_recycle(
                                    infobox_rect,
                                    recycle_bbox_rel,
                                    win_left,
                                    win_top,
                                    win_width,
                                    win_height,
                                )
                                action_taken = "RECYCLE"
                        else:
                            action_taken = "SKIP_NO_INFOBOX"
                elif not item_name:
                    action_taken = "SKIP_NO_NAME"
                elif not actions:
                    action_taken = "SKIP_NO_ACTION_MAP"
            elif not item_name:
                action_taken = "SKIP_NO_NAME"

            note_suffix = f" note={decision_note}" if decision_note else ""
            infobox_status = "found" if infobox_rect else "missing"
            action_label = "SKIPPED" if action_taken.startswith("SKIP") else action_taken
            detail_suffix = f" detail={action_taken}" if action_label != action_taken else ""
            item_label = item_name or "<unreadable>"
            print(
                f"[item] idx={global_idx:03d} page={page + 1:02d} cell={cell.index:02d} "
                f"item='{item_label}' action={action_label}{detail_suffix} "
                f"infobox={infobox_status}{note_suffix}"
            )

            results.append(
                ItemActionResult(
                    page=page,
                    cell=cell,
                    item_name=item_name,
                    decision=decision,
                    action_taken=action_taken,
                    note=decision_note,
                )
            )

            destructive_action = action_taken in {"SELL", "RECYCLE"}
            if destructive_action:
                _open_cell_menu(cell, win_left, win_top)
                continue

            if progress:
                progress.update(1)

            idx += 1
            if idx < len(page_cells):
                next_page, next_cell = page_cells[idx]
                if next_page == page:
                    _open_cell_menu(next_cell, win_left, win_top)
    finally:
        if progress:
            progress.close()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scan the ARC Raiders inventory grid(s).")
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of 6x4 grids to scan by scrolling down between pages.",
    )
    parser.add_argument(
        "--scroll-clicks",
        type=int,
        default=SCROLL_CLICKS_PER_PAGE,
        help="Scroll clicks to reach the next grid (positive scrolls downward).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar output.",
    )
    parser.add_argument(
        "--actions-file",
        type=Path,
        default=ITEM_ACTIONS_PATH,
        help="Path to items_actions.json for keep/recycle/sell decisions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan only; do not click sell/recycle actions.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        results = scan_inventory(
            show_progress=not args.no_progress,
            pages=args.pages,
            scroll_clicks_per_page=args.scroll_clicks,
            apply_actions=not args.dry_run,
            actions_path=args.actions_file,
        )
    except KeyboardInterrupt:
        print("Aborted by Escape key.")
        return 0
    except TimeoutError as exc:
        print(exc)
        return 1
    except RuntimeError as exc:
        print(f"Fatal: {exc}")
        return 1

    cells_per_page = len(Grid())
    for result in results:
        label = result.item_name or "<unreadable>"
        global_idx = result.page * cells_per_page + result.cell.index
        decision_label = result.decision or result.action_taken
        action_suffix = f" ({result.action_taken})" if result.action_taken != decision_label else ""
        note_suffix = f" {result.note}" if result.note else ""
        print(
            f"[page {result.page + 1:02d}] global_idx={global_idx:03d} "
            f"Cell r{result.cell.row} c{result.cell.col} idx={result.cell.index:02d}: "
            f"{label} -> {decision_label}{action_suffix}{note_suffix}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
