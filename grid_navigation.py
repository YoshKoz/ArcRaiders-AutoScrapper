"""
inventory_grid.py

Utilities for working with the 4x6 item grid in the ARC Raiders inventory UI.

- Coordinate system is window-relative pixels for a 1920x1080 window.
- Grid is 4 columns (left->right) by 6 rows (top->bottom).
"""

from dataclasses import dataclass
from typing import Iterator, Tuple


# ---------------------------------------------------------------------------
# Configuration (derived from your sampled ROIs)
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

# Window size (for normalized coordinates)
WIN_W = 1920
WIN_H = 1080


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

    @property
    def center_norm(self) -> Tuple[float, float]:
        """Center of the cell normalized to [0,1] relative to window size."""
        cx, cy = self.center
        return cx / WIN_W, cy / WIN_H


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

    def center_norm(self, row: int, col: int) -> Tuple[float, float]:
        """Normalized center (cx_norm, cy_norm) of cell at (row, col)."""
        return self.cell(row, col).center_norm


# ---------------------------------------------------------------------------
# Example usage / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid = Grid()

    print(f"Grid has {len(grid)} cells.\n")

    # Example: iterate row-by-row, left-to-right
    for cell in grid:
        cx, cy = cell.center
        print(
            f"Cell idx={cell.index:2d} "
            f"(row={cell.row}, col={cell.col}) "
            f"rect={cell.rect} center=({cx:.1f}, {cy:.1f})"
        )

    # Example: directly access a specific cell
    print("\nCenter of top-left cell (row=0,col=0):", grid.center(0, 0))
    print("Center of bottom-right cell (row=5,col=3):", grid.center(5, 3))
