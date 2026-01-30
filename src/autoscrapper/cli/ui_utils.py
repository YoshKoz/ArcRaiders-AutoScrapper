from __future__ import annotations

from typing import Tuple


def clamp(value: int, minimum: int, maximum: int) -> int:
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def window_for_cursor(total: int, cursor: int, max_rows: int) -> Tuple[int, int]:
    if max_rows <= 0:
        return 0, total
    if total <= max_rows:
        return 0, total

    cursor = clamp(cursor, 0, max(0, total - 1))
    half = max_rows // 2
    start = max(0, cursor - half)
    end = start + max_rows
    if end > total:
        end = total
        start = max(0, end - max_rows)
    return start, end
