import math
import heapq
from typing import Iterable, List, Optional, Tuple

import numpy as np
from numpy import typing as npt

GridIndex = Tuple[int, int]


class DStarLite:
    """D* Lite planner for 2D grid maps (pure Python, numpy-friendly).

    This is an incremental planner implementing the D* Lite algorithm. It
    supports dynamic updates: change cell costs and call `update_costs(...)`
    followed by `compute_shortest_path()` to replan efficiently.

    The planner treats grid entries with value `np.inf` as impassable obstacles.

    Notes:
        - Grid indexing uses NumPy convention: (row, col).
        - Movement is 8-connected. The traversal cost of an edge between cells u->v
          uses the average of cell costs: cost = base_step_cost * 0.5*(cost[u]+cost[v]),
          where base_step_cost is 1 for orthogonal moves and sqrt(2) for diagonals.

    References:
        - Koenig, S., & Likhachev, M. (2002-2005). D* Lite algorithm papers.
    """

    def __init__(
        self,
        cost_grid: npt.NDArray[np.float32],
        start: GridIndex,
        goal: GridIndex,
    ) -> None:
        """
        Args:
            cost_grid: 2D array of non-negative costs (np.inf = obstacle).
            start: (row, col) start index in grid coordinates.
            goal: (row, col) goal index in grid coordinates.
        """
        if cost_grid.ndim != 2:
            raise ValueError("cost_grid must be 2D")
        self.cost_grid = cost_grid.astype(np.float64, copy=False)
        self.n_rows, self.n_cols = self.cost_grid.shape

        self.start = start
        self.goal = goal

        # D* Lite internal values
        # g: cost-to-come estimate
        # rhs: one-step lookahead value
        self.g = np.full(self.cost_grid.shape, np.inf, dtype=np.float64)
        self.rhs = np.full(self.cost_grid.shape, np.inf, dtype=np.float64)

        # priority queue (open list) implemented with heapq storing (key, counter, (r,c))
        self._open_heap: List[Tuple[Tuple[float, float], int, GridIndex]] = []
        self._entry_finder = {}  # maps (r,c) -> (key, counter)
        self._counter = 0  # monotonic counter to break ties in heap

        # km is used by D* Lite when start changes (for dynamic replanning)
        self.km = 0.0

        # heuristic function scaling factor (default 1.0)
        self.heuristic_scale = 1.0

        # initialize
        self._init_planner()

    # ---------------------------
    # Helper functions
    # ---------------------------
    @staticmethod
    def _heuristic(a: GridIndex, b: GridIndex) -> float:
        """Euclidean distance heuristic between two grid indices."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _in_bounds(self, idx: GridIndex) -> bool:
        r, c = idx
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def _is_obstacle(self, idx: GridIndex) -> bool:
        return not np.isfinite(self.cost_grid[idx])

    def _neighbors(self, idx: GridIndex) -> Iterable[GridIndex]:
        """8-connected neighbors (row, col)"""
        r, c = idx
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if self._in_bounds(nb) and not self._is_obstacle(nb):
                    yield nb

    def _edge_cost(self, a: GridIndex, b: GridIndex) -> float:
        """Cost to move from cell a to b.

        Uses average cell cost times geometric distance (1 or sqrt(2)).
        If either cell is obstacle (np.inf), returns np.inf.
        """
        if self._is_obstacle(a) or self._is_obstacle(b):
            return math.inf
        base = math.hypot(a[0] - b[0], a[1] - b[1])  # 1 or sqrt(2)
        cost = 0.5 * (self.cost_grid[a] + self.cost_grid[b]) * base
        # protect against numerical tiny negative values
        return float(max(cost, 0.0))

    # ---------------------------
    # Priority queue (open list)
    # ---------------------------
    def _calculate_key(self, idx: GridIndex) -> Tuple[float, float]:
        """Compute D* Lite key for a state idx."""
        g_val = self.g[idx]
        rhs_val = self.rhs[idx]
        min_val = min(g_val, rhs_val)
        return (
            min_val + self.heuristic_scale * self._heuristic(self.start, idx) + self.km,
            min_val,
        )

    def _push_open(self, idx: GridIndex) -> None:
        """Add or update state idx in the open list."""
        key = self._calculate_key(idx)
        # mark existing entry invalid by replacing in entry_finder
        self._counter += 1
        entry = (key, self._counter, idx)
        heapq.heappush(self._open_heap, entry)
        self._entry_finder[idx] = entry

    def _pop_open(self) -> Optional[GridIndex]:
        """Pop the top valid state from the heap (lazily skipping stale entries)."""
        while self._open_heap:
            key, count, idx = heapq.heappop(self._open_heap)
            # only accept if it matches current entry_finder (not stale)
            cur = self._entry_finder.get(idx)
            if cur is None:
                continue
            if cur[1] == count and cur[0] == key:
                return idx
            # else stale, skip
        return None

    def _top_key(self) -> Tuple[float, float]:
        """Return smallest key currently in open (or +inf if empty)."""
        while self._open_heap:
            key, count, idx = self._open_heap[0]
            cur = self._entry_finder.get(idx)
            if cur is None or cur[1] != count or cur[0] != key:
                heapq.heappop(self._open_heap)  # discard stale
                continue
            return key
        return (math.inf, math.inf)

    def _remove_from_open(self, idx: GridIndex) -> None:
        """Invalidate an entry by removing from entry_finder (lazy deletion)."""
        self._entry_finder.pop(idx, None)

    # ---------------------------
    # Core D* Lite procedures
    # ---------------------------
    def _init_planner(self) -> None:
        """Initialize rhs,g and open list according to D* Lite pseudocode."""
        self.g.fill(math.inf)
        self.rhs.fill(math.inf)
        self._open_heap.clear()
        self._entry_finder.clear()
        self._counter = 0
        self.km = 0.0

        # goal has rhs = 0
        self.rhs[self.goal] = 0.0
        self._push_open(self.goal)

    def update_vertex(self, u: GridIndex) -> None:
        """Update the vertex u when an edge cost to/from u changes."""
        if u != self.goal:
            # rhs(u) = min_{s' in Succ(u)} ( cost(u,s') + g(s') )
            best = math.inf
            for sprime in self._neighbors(u):
                candidate = self._edge_cost(u, sprime) + self.g[sprime]
                if candidate < best:
                    best = candidate
            self.rhs[u] = best

        # update open list membership
        if self.g[u] != self.rhs[u]:
            self._push_open(u)
        else:
            self._remove_from_open(u)

    def compute_shortest_path(self, max_iterations: int = 10_000_000) -> None:
        """Main D* Lite loop: drive g and rhs towards consistency.

        Args:
            max_iterations: safety cap to avoid pathological infinite loops.
        """
        iters = 0
        top_key = self._top_key()
        start_key = self._calculate_key(self.start)
        while (top_key < start_key) or (self.rhs[self.start] != self.g[self.start]):
            if iters > max_iterations:
                raise RuntimeError("D* Lite exceeded max iterations")
            u = self._pop_open()
            if u is None:
                break
            old_g = self.g[u]
            key_u = self._calculate_key(u)
            if old_g > self.rhs[u]:
                self.g[u] = self.rhs[u]
                # for all predecessors (here neighbors) v of u:
                for v in self._neighbors(u):
                    self.update_vertex(v)
            else:
                self.g[u] = math.inf
                # update u and all predecessors
                self.update_vertex(u)
                for v in self._neighbors(u):
                    self.update_vertex(v)
            top_key = self._top_key()
            iters += 1

    # ---------------------------
    # Public API
    # ---------------------------
    def plan(self) -> None:
        """Compute initial plan (calls compute_shortest_path)."""
        self._init_planner()
        self.compute_shortest_path()

    def query(self, start: Optional[GridIndex] = None) -> Tuple[np.ndarray, float]:
        """Return path from `start` (or self.start) to self.goal as an array shape (2, L).

        Also returns the path cost (sum of edge costs). If no path exists returns (empty array, inf).

        Args:
            start: optional start coordinate (row, col). If provided, the internal start will be updated.
        """
        if start is not None:
            # move start (D* Lite handles start changes by increasing km)
            self.km += self._heuristic(self.start, start)
            self.start = start

        # Ensure shortest path is computed to the (possibly new) start
        self.compute_shortest_path()

        if not np.isfinite(self.g[self.start]):
            return np.empty((2, 0), dtype=int), math.inf

        # Greedy follow from start to goal using locally optimal neighbor choices
        path_rows: List[int] = []
        path_cols: List[int] = []
        cur = self.start
        total_cost = 0.0
        max_steps = self.n_rows * self.n_cols * 4  # safety cap
        steps = 0
        while cur != self.goal and steps < max_steps:
            path_rows.append(cur[0])
            path_cols.append(cur[1])

            # pick neighbor minimizing cost(u,s') + g[s']
            best_succ = None
            best_value = math.inf
            for s in self._neighbors(cur):
                val = self._edge_cost(cur, s) + self.g[s]
                if val < best_value:
                    best_value = val
                    best_succ = s

            if best_succ is None:
                # no successor -> no path
                return np.empty((2, 0), dtype=int), math.inf

            total_cost += self._edge_cost(cur, best_succ)
            cur = best_succ
            steps += 1

        # append goal
        path_rows.append(self.goal[0])
        path_cols.append(self.goal[1])

        path = np.vstack(
            (np.array(path_rows, dtype=int), np.array(path_cols, dtype=int))
        )
        return path, total_cost

    def update_costs(self, changed_cells: Iterable[Tuple[GridIndex, float]]) -> None:
        """Apply cost changes and incrementally replan.

        Args:
            changed_cells: iterable of ( (row, col), new_cost ) pairs.
        """
        # apply changes
        affected = set()
        for (r, c), new_cost in changed_cells:
            if not self._in_bounds((r, c)):
                continue
            self.cost_grid[r, c] = new_cost
            affected.add((r, c))
            # neighbors of the changed cell also need update
            for nb in self._neighbors((r, c)):
                affected.add(nb)

        # update rhs for affected vertices and push to open
        for u in affected:
            self.update_vertex(u)
