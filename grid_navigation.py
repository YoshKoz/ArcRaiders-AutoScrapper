"""
Grid helpers for the 4x6 ARC Raiders inventory UI.

Coordinate system is window-relative pixels for a 1920x1080 window.
"""

from dataclasses import dataclass
from typing import Iterator, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GRID_COLS = 4   # 4 across
GRID_ROWS = 6   # 6 down

# Top-left of cell (row=0, col=0) in window-relative pixels
X0 = 153
Y0 = 273

# Distance from one cell's top-left to the next
STEP_X = 104
STEP_Y = 104

# Approximate cell size (used for rectangles + centers)
CELL_W = 96
CELL_H = 96

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    """Represents a single grid cell."""
    index: int          # 0..(GRID_ROWS*GRID_COLS-1), row-major
    row: int            # 0..GRID_ROWS-1 (top to bottom)
    col: int            # 0..GRID_COLS-1 (left to right)
    x: int              # top-left x in pixels
    y: int              # top-left y in pixels
    width: int = CELL_W
    height: int = CELL_H

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        """(x, y, w, h) rectangle in pixels."""
        return self.x, self.y, self.width, self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Center of the cell in pixels (cx, cy)."""
        cx = self.x + self.width / 2.0
        cy = self.y + self.height / 2.0
        return cx, cy


# ---------------------------------------------------------------------------
# Grid helper
# ---------------------------------------------------------------------------

class Grid:
    COLS = GRID_COLS
    ROWS = GRID_ROWS

    def __init__(self):
        # Precompute all cells in row-major order
        self._cells = []
        idx = 0
        for r in range(self.ROWS):
            for c in range(self.COLS):
                x = X0 + c * STEP_X
                y = Y0 + r * STEP_Y
                cell = Cell(
                    index=idx,
                    row=r,
                    col=c,
                    x=x,
                    y=y,
                )
                self._cells.append(cell)
                idx += 1

    # ---- Accessors --------------------------------------------------------

    def __len__(self) -> int:
        """Total number of cells."""
        return len(self._cells)

    def __iter__(self) -> Iterator[Cell]:
        """Iterate cells row-by-row, left-to-right, top-to-bottom."""
        return iter(self._cells)

    def cell_by_index(self, index: int) -> Cell:
        """
        Get cell by linear index (row-major).
        index: 0..len(grid)-1
        """
        return self._cells[index]

    def cell(self, row: int, col: int) -> Cell:
        """
        Get cell by (row, col).
        row: 0..ROWS-1
        col: 0..COLS-1
        """
        idx = row * self.COLS + col
        return self._cells[idx]

    def center_by_index(self, index: int) -> Tuple[float, float]:
        """Center (cx, cy) of cell with given row-major index."""
        return self.cell_by_index(index).center

    def center(self, row: int, col: int) -> Tuple[float, float]:
        """Center (cx, cy) of cell at (row, col)."""
        return self.cell(row, col).center
