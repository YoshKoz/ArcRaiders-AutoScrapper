from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

from ..core.item_actions import clean_ocr_text
from .tesseract import image_to_data, image_to_string

# Infobox visual characteristics
INFOBOX_COLOR_BGR = np.array([223, 238, 249], dtype=np.uint8)  # #f9eedf in BGR
INFOBOX_TOLERANCE = 30  # Increased tolerance for color variations
# Expected infobox size, normalized to the active window
INFOBOX_TARGET_NORM_W = 0.132
INFOBOX_TARGET_NORM_H = 0.268
INFOBOX_SCALE_MIN = 0.5  # accept down to 50% of the expected size
INFOBOX_MIN_NORM_W = INFOBOX_TARGET_NORM_W * INFOBOX_SCALE_MIN
INFOBOX_MIN_NORM_H = INFOBOX_TARGET_NORM_H * INFOBOX_SCALE_MIN

# Item title placement inside the infobox (relative to infobox size)
TITLE_HEIGHT_REL = 0.18

# Confirmation buttons (window-normalized rectangles)
SELL_CONFIRM_RECT_NORM = (0.5047, 0.6941, 0.1791, 0.0531)
RECYCLE_CONFIRM_RECT_NORM = (0.5058, 0.6274, 0.1777, 0.0544)

# Inventory count ROI (window-normalized rectangle)
# Matches the always-visible "items in stash" label near the top-left.
INVENTORY_COUNT_RECT_NORM = (0.0734, 0.1583, 0.0380, 0.0231)

_OCR_DEBUG_DIR: Optional[Path] = None


@dataclass
class InfoboxOcrResult:
    item_name: str
    raw_item_text: str
    sell_bbox: Optional[Tuple[int, int, int, int]]
    recycle_bbox: Optional[Tuple[int, int, int, int]]
    processed: np.ndarray
    preprocess_time: float
    ocr_time: float
    ocr_failed: bool = False


def is_empty_cell(
    bright_fraction: float, gray_var: float, edge_fraction: float
) -> bool:
    """
    Decide if a slot is empty based on precomputed metrics.

    Empirically tuned heuristic: mostly dark with low texture and few edges.
    """
    # Primary test: looks dark with few bright pixels
    if bright_fraction >= 0.03:
        return False

    # Fallback
    if gray_var > 700:
        return False
    if edge_fraction > 0.09:
        return False

    return True


def slot_metrics(
    slot_bgr: np.ndarray,
    v_thresh: int = 120,
    canny1: int = 50,
    canny2: int = 150,
) -> Tuple[float, float, float]:
    """
    Compute simple statistics for an inventory slot.

    Returns:
        (bright_fraction, gray_var, edge_fraction)
    """
    if slot_bgr.size == 0:
        raise ValueError("slot_bgr is empty (ROI outside image bounds?)")

    # Brightness stats from HSV V channel
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    bright_fraction = float(np.mean(v > v_thresh))

    # Grayscale variance = how textured / high-contrast the cell is
    gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)
    gray_var = float(gray.var())

    # Edge density via Canny
    edges = cv2.Canny(gray, canny1, canny2)
    edge_fraction = float(np.count_nonzero(edges)) / edges.size

    return bright_fraction, gray_var, edge_fraction


def is_slot_empty(
    slot_bgr: np.ndarray,
    v_thresh: int = 120,
    canny1: int = 50,
    canny2: int = 150,
) -> bool:
    """
    Decide if an inventory slot is visually empty using slot metrics.
    """
    bright_fraction, gray_var, edge_fraction = slot_metrics(
        slot_bgr, v_thresh, canny1, canny2
    )
    return is_empty_cell(bright_fraction, gray_var, edge_fraction)


def find_infobox(bgr_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate the item infobox rectangle that matches the infobox background color.
    The infobox is a tall, narrow panel showing item details.
    We prefer taller rectangles but accept any that meet minimum size.
    Returns (x, y, w, h) relative to the provided image, or None if not found.
    """
    kernel = np.ones((3, 3), np.uint8)
    window_height, window_width = bgr_image.shape[:2]

    # Prefer taller rectangles but don't strictly require it
    # Context menus are typically wider than tall, infobox is taller than wide
    PREFERRED_ASPECT_RATIO = 0.8  # h/w >= 0.8 preferred (slightly lenient)

    def _meets_min_infobox_size(
        w: int, h: int, window_width: int, window_height: int
    ) -> bool:
        if window_width <= 0 or window_height <= 0:
            return False
        norm_w = w / float(window_width)
        norm_h = h / float(window_height)
        # More lenient: require at least 40% of expected size
        return norm_w >= INFOBOX_MIN_NORM_W * 0.8 and norm_h >= INFOBOX_MIN_NORM_H * 0.8

    def _infobox_score(x: int, y: int, w: int, h: int) -> float:
        """Score a candidate rectangle - higher is better."""
        area = w * h
        aspect = h / max(1, w)
        # Bonus for tall aspect ratios (more likely to be infobox)
        aspect_bonus = 1.5 if aspect >= PREFERRED_ASPECT_RATIO else 1.0
        # Prefer rectangles on the right side of screen (where infobox appears)
        position_bonus = 1.2 if x > window_width * 0.3 else 1.0
        return area * aspect_bonus * position_bonus

    def _find_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if _meets_min_infobox_size(w, h, window_width, window_height):
                # Score candidates instead of strict filtering
                score = _infobox_score(x, y, w, h)
                candidates.append((score, (x, y, w, h)))
        if not candidates:
            return None
        _, best_rect = max(candidates, key=lambda item: item[0])
        return best_rect

    # Use tolerance-based mask around the expected infobox color
    color = INFOBOX_COLOR_BGR.astype(
        np.int16
    )  # promote to avoid uint8 overflow when adding tolerance
    lower = np.clip(color - INFOBOX_TOLERANCE, 0, 255).astype(np.uint8)
    upper = np.clip(color + INFOBOX_TOLERANCE, 0, 255).astype(np.uint8)
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


def rect_center(rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Center (cx, cy) of a rectangle.
    """
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def normalized_rect_to_window(
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


def window_relative_to_screen(
    rect: Tuple[int, int, int, int],
    window_left: int,
    window_top: int,
) -> Tuple[int, int, int, int]:
    """
    Convert a window-relative rectangle to absolute screen coordinates.
    """
    x, y, w, h = rect
    return window_left + x, window_top + y, w, h


def inventory_count_rect(
    window_width: int, window_height: int
) -> Tuple[int, int, int, int]:
    """
    Window-relative rectangle for the always-visible inventory count label.
    """
    return normalized_rect_to_window(
        INVENTORY_COUNT_RECT_NORM, window_width, window_height
    )


def sell_confirm_button_rect(
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
) -> Tuple[int, int, int, int]:
    """
    Absolute screen rectangle for the Sell confirmation button.
    """
    rel_rect = normalized_rect_to_window(
        SELL_CONFIRM_RECT_NORM, window_width, window_height
    )
    return window_relative_to_screen(rel_rect, window_left, window_top)


def recycle_confirm_button_rect(
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
) -> Tuple[int, int, int, int]:
    """
    Absolute screen rectangle for the Recycle confirmation button.
    """
    rel_rect = normalized_rect_to_window(
        RECYCLE_CONFIRM_RECT_NORM, window_width, window_height
    )
    return window_relative_to_screen(rel_rect, window_left, window_top)


def sell_confirm_button_center(
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
) -> Tuple[int, int]:
    """
    Center of the Sell confirmation button (absolute screen coords).
    """
    return rect_center(
        sell_confirm_button_rect(window_left, window_top, window_width, window_height)
    )


def recycle_confirm_button_center(
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
) -> Tuple[int, int]:
    """
    Center of the Recycle confirmation button (absolute screen coords).
    """
    return rect_center(
        recycle_confirm_button_rect(
            window_left, window_top, window_width, window_height
        )
    )


def preprocess_for_ocr(roi_bgr: np.ndarray, upscale: bool = True) -> np.ndarray:
    """
    Preprocess an image region for OCR.

    Steps:
    1. Optionally upscale small images for better OCR accuracy
    2. Convert to grayscale
    3. Apply bilateral filter to reduce noise while preserving edges
    4. Apply adaptive thresholding with Otsu fallback
    """
    if roi_bgr.size == 0:
        return roi_bgr

    # Upscale small images for better OCR accuracy
    h, w = roi_bgr.shape[:2]
    if upscale and (h < 50 or w < 100):
        scale = max(2, min(4, 100 // max(1, min(h, w))))
        roi_bgr = cv2.resize(roi_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping text edges sharp
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Try adaptive thresholding first, fall back to Otsu if it fails
    try:
        # Adaptive thresholding works better for varying lighting conditions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        # Check if the result is mostly one color (likely failed)
        white_ratio = np.mean(binary > 127)
        if white_ratio < 0.1 or white_ratio > 0.9:
            # Fall back to Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


def enable_ocr_debug(debug_dir: Path) -> None:
    """
    Enable saving OCR debug images into the provided directory.
    """
    global _OCR_DEBUG_DIR
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        _OCR_DEBUG_DIR = debug_dir
        print(f"[vision_ocr] OCR debug output enabled at {_OCR_DEBUG_DIR}", flush=True)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        print(f"[vision_ocr] failed to enable OCR debug dir: {exc}", flush=True)
        _OCR_DEBUG_DIR = None


def _save_debug_image(name: str, image: np.ndarray) -> None:
    """
    Write a debug image if a debug directory has been configured.
    """
    if _OCR_DEBUG_DIR is None:
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{time.time_ns() % 1_000_000_000:09d}_{name}.png"
    path = _OCR_DEBUG_DIR / filename
    try:
        cv2.imwrite(str(path), image)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        print(f"[vision_ocr] failed to save debug image {path}: {exc}", flush=True)


# Context menu text patterns that should NOT be recognized as item names
CONTEXT_MENU_PATTERNS = {
    "split stack",
    "move to backpack",
    "move to stash",
    "sell",
    "recycle",
    "drop",
    "equip",
    "unequip",
    "use",
    "obtained from",
    "used to craft",
    "can be recycled",
    "cannot be recycled",
    "into crafting materials",
    "a wide range of items",
    # Description text patterns
    "explosives, and utility",
    "utility items",
    "grenade, pulse mine",
    "pulse mine, snap",
    "snap blast",
    "crafting materials",
    "can be used",
    "used for",
    "required for",
    "workshop project",
    # More description fragments
    "grenade, shrapnel",
    "shrapnel grenade",
    "singularity",
    "ferro, kettle",
    "kettle, medium",
}


def _is_context_menu_text(text: str) -> bool:
    """Check if the text looks like context menu or description text, not an item name."""
    text_lower = text.lower().strip()
    for pattern in CONTEXT_MENU_PATTERNS:
        if pattern in text_lower:
            return True
    return False


def _extract_title_from_data(
    ocr_data: Dict[str, List],
    image_height: int,
    top_fraction: float = 0.45,
) -> Tuple[str, str]:
    """
    Choose the best-confidence line near the top of the infobox as the title.
    Filters out context menu text and description fragments.
    Increased top_fraction to 0.45 to catch titles that appear lower.
    """
    texts = ocr_data.get("text", [])
    if not texts:
        return "", ""

    cutoff = max(1.0, float(image_height) * top_fraction)
    groups: Dict[Tuple[int, int, int, int], List[int]] = {}
    n = len(texts)
    for i in range(n):
        raw_text = texts[i] or ""
        cleaned = clean_ocr_text(raw_text)
        if not cleaned:
            continue

        top = float(ocr_data["top"][i])
        height = float(ocr_data["height"][i])
        center_y = top + (height / 2.0)
        if center_y > cutoff:
            continue

        key = (
            int(ocr_data["page_num"][i]),
            int(ocr_data["block_num"][i]),
            int(ocr_data["par_num"][i]),
            int(ocr_data["line_num"][i]),
        )
        groups.setdefault(key, []).append(i)

    if not groups:
        return "", ""

    def _group_score(indices: List[int]) -> float:
        confs = []
        for idx in indices:
            try:
                confs.append(float(ocr_data["conf"][idx]))
            except Exception:
                continue
        return sum(confs) / len(confs) if confs else -1.0

    # Sort groups by score and try to find a valid title (not context menu text)
    sorted_keys = sorted(groups.keys(), key=lambda k: _group_score(groups[k]), reverse=True)

    for key in sorted_keys:
        ordered_indices = sorted(groups[key])
        cleaned_parts = [
            clean_ocr_text(texts[i] or "") for i in ordered_indices if texts[i]
        ]
        raw_parts = [(texts[i] or "").strip() for i in ordered_indices if texts[i]]
        cleaned = " ".join(p for p in cleaned_parts if p).strip()
        raw = " ".join(p for p in raw_parts if p).strip()

        # Skip if this looks like context menu or description text
        if _is_context_menu_text(cleaned) or _is_context_menu_text(raw):
            continue

        if cleaned:  # Found a valid title
            return cleaned, raw

    # Fallback: if we found raw text but cleaning removed everything, try less aggressive cleaning
    for key in sorted_keys:
        ordered_indices = sorted(groups[key])
        raw_parts = [(texts[i] or "").strip() for i in ordered_indices if texts[i]]
        raw = " ".join(p for p in raw_parts if p).strip()
        if raw and not _is_context_menu_text(raw):
            # Extract just uppercase words (item names are typically all caps)
            words = re.findall(r'[A-Z][A-Z0-9]+(?:\s+[A-Z][A-Z0-9]+)*', raw)
            if words:
                return words[0], raw

    return "", ""


def _extract_action_line_bbox(
    ocr_data: Dict[str, List],
    target: Literal["sell", "recycle"],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Given OCR data, return a bbox (left, top, w, h) for
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


def find_action_bbox_by_ocr(
    infobox_bgr: np.ndarray,
    target: Literal["sell", "recycle"],
) -> Tuple[Optional[Tuple[int, int, int, int]], np.ndarray]:
    """
    Run OCR over the full infobox to locate the line containing the target
    action. Returns (bbox, processed_image) where bbox is infobox-relative.
    """
    processed = preprocess_for_ocr(infobox_bgr)
    _save_debug_image(f"infobox_action_{target}_processed", processed)
    try:
        data = image_to_data(processed)
    except Exception as exc:
        print(
            f"[vision_ocr] ocr_backend image_to_data failed for target={target}; falling back to no bbox. "
            f"error={exc}",
            flush=True,
        )
        return None, processed

    bbox = _extract_action_line_bbox(data, target)
    return bbox, processed


def ocr_infobox(infobox_bgr: np.ndarray) -> InfoboxOcrResult:
    """
    OCR the full infobox once to derive the title and action line positions.
    """
    preprocess_start = time.perf_counter()
    _save_debug_image("infobox_raw", infobox_bgr)
    processed = preprocess_for_ocr(infobox_bgr)
    _save_debug_image("infobox_processed", processed)
    preprocess_time = time.perf_counter() - preprocess_start

    ocr_time = 0.0
    try:
        ocr_start = time.perf_counter()
        data = image_to_data(processed)
        ocr_time = time.perf_counter() - ocr_start
    except Exception as exc:
        print(
            f"[vision_ocr] ocr_backend image_to_data failed for full infobox; "
            f"falling back to empty OCR result. error={exc}",
            flush=True,
        )
        return InfoboxOcrResult(
            item_name="",
            raw_item_text="",
            sell_bbox=None,
            recycle_bbox=None,
            processed=processed,
            preprocess_time=preprocess_time,
            ocr_time=ocr_time,
            ocr_failed=True,
        )

    item_name, raw_item_text = _extract_title_from_data(data, processed.shape[0])
    sell_bbox = _extract_action_line_bbox(data, "sell")
    recycle_bbox = _extract_action_line_bbox(data, "recycle")
    return InfoboxOcrResult(
        item_name=item_name,
        raw_item_text=raw_item_text,
        sell_bbox=sell_bbox,
        recycle_bbox=recycle_bbox,
        processed=processed,
        preprocess_time=preprocess_time,
        ocr_time=ocr_time,
    )


def ocr_item_name(roi_bgr: np.ndarray) -> str:
    """
    OCR the item name from the pre-cropped title ROI.
    """
    if roi_bgr.size == 0:
        return ""

    processed = preprocess_for_ocr(roi_bgr)
    raw = image_to_string(processed)
    return clean_ocr_text(raw)


def ocr_inventory_count(roi_bgr: np.ndarray) -> Tuple[Optional[int], str]:
    """
    OCR the "items in stash" label and return (count, raw_text).
    """
    if roi_bgr.size == 0:
        return None, ""

    _save_debug_image("inventory_count_raw", roi_bgr)
    processed = preprocess_for_ocr(roi_bgr)
    _save_debug_image("inventory_count_processed", processed)

    try:
        raw = image_to_string(processed)
    except Exception as exc:
        print(
            f"[vision_ocr] ocr_backend image_to_string failed for inventory count: {exc}",
            flush=True,
        )
        return None, ""

    cleaned = (raw or "").replace("\n", " ").strip()
    digits = re.findall(r"\d+", cleaned)
    if not digits:
        return None, cleaned

    try:
        count = max(int(d) for d in digits)
    except Exception:
        count = None

    return count, cleaned
