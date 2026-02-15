"""
3-Tier Minesweeper Constraint Satisfaction Solver

Tier 1: Single-cell constraint propagation
Tier 2: Set-based coupled constraints (subset reduction)
Tier 3: Tank solver (backtracking enumeration with component partitioning)

Used for:
- Generating training data labels (optimal moves)
- Reward function (deducibility detection)
- Cell probability computation
"""

import time
from typing import List, Tuple, Set, Dict, Optional, FrozenSet
from collections import defaultdict
from math import exp, lgamma


def get_neighbors(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    """Get all valid 8-directional neighbors of a cell."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbors.append((nr, nc))
    return neighbors


class Constraint:
    """Represents: sum of mine indicators for `cells` == `count`"""

    __slots__ = ["cells", "count"]

    def __init__(self, cells: FrozenSet[Tuple[int, int]], count: int):
        self.cells = cells
        self.count = count

    def __repr__(self):
        return f"Constraint({set(self.cells)}, count={self.count})"

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __hash__(self):
        return hash((self.cells, self.count))


class MinesweeperSolver:
    """3-tier Minesweeper solver with probability computation."""

    def __init__(
        self,
        board: List[List[str]],
        rows: int,
        cols: int,
        total_mines: int,
        timeout: float = 1.0,
    ):
        """
        Args:
            board: 2D list of cell values ('.' unrevealed, 'F' flagged, '0'-'8' numbers, '*' mine)
            rows, cols: board dimensions
            total_mines: total mine count on the board
            timeout: max seconds per Tank solver component
        """
        self.board = board
        self.rows = rows
        self.cols = cols
        self.total_mines = total_mines
        self.timeout = timeout

        # Parse board state
        self.revealed: Set[Tuple[int, int]] = set()
        self.flagged: Set[Tuple[int, int]] = set()
        self.unrevealed: Set[Tuple[int, int]] = set()
        self.numbers: Dict[Tuple[int, int], int] = {}

        for r in range(rows):
            for c in range(cols):
                val = board[r][c]
                if val == ".":
                    self.unrevealed.add((r, c))
                elif val == "F":
                    self.flagged.add((r, c))
                elif val in "012345678":
                    self.revealed.add((r, c))
                    self.numbers[(r, c)] = int(val)
                elif val == "*":
                    self.revealed.add((r, c))

        self.remaining_mines = total_mines - len(self.flagged)

        # Results
        self.safe_cells: Set[Tuple[int, int]] = set()
        self.mine_cells: Set[Tuple[int, int]] = set()
        self.cell_probabilities: Dict[Tuple[int, int], float] = {}

    def solve(self) -> None:
        """Run all solver tiers."""
        self._tier1_propagation()
        self._tier2_coupled_constraints()
        self._tier3_tank_solver()

    def solve_fast(self) -> None:
        """Run only Tier 1 + 2 (fast, no enumeration)."""
        self._tier1_propagation()
        self._tier2_coupled_constraints()

    # ================================================================
    # TIER 1: Single-Cell Constraint Propagation
    # ================================================================

    def _tier1_propagation(self) -> None:
        """Iterate single-cell rules until fixed point."""
        changed = True
        while changed:
            changed = False
            for (r, c), num in list(self.numbers.items()):
                neighbors = get_neighbors(r, c, self.rows, self.cols)

                adj_flags = 0
                adj_unrevealed = []
                for nr, nc in neighbors:
                    if (nr, nc) in self.flagged or (nr, nc) in self.mine_cells:
                        adj_flags += 1
                    elif (nr, nc) in self.unrevealed and (
                        nr,
                        nc,
                    ) not in self.safe_cells:
                        adj_unrevealed.append((nr, nc))

                remaining = num - adj_flags

                # All mines accounted for → remaining unrevealed are safe
                if remaining == 0 and adj_unrevealed:
                    for cell in adj_unrevealed:
                        if cell not in self.safe_cells:
                            self.safe_cells.add(cell)
                            changed = True

                # Remaining mines == remaining unrevealed → all are mines
                if remaining > 0 and remaining == len(adj_unrevealed):
                    for cell in adj_unrevealed:
                        if cell not in self.mine_cells:
                            self.mine_cells.add(cell)
                            changed = True

    # ================================================================
    # TIER 2: Set-Based Coupled Constraints
    # ================================================================

    def _build_constraints(self) -> List[Constraint]:
        """Build constraint list from revealed numbers."""
        constraints = []
        for (r, c), num in self.numbers.items():
            neighbors = get_neighbors(r, c, self.rows, self.cols)

            adj_flags = 0
            adj_unknown = []
            for nr, nc in neighbors:
                if (nr, nc) in self.flagged or (nr, nc) in self.mine_cells:
                    adj_flags += 1
                elif (nr, nc) in self.unrevealed and (nr, nc) not in self.safe_cells:
                    adj_unknown.append((nr, nc))

            remaining = num - adj_flags
            if adj_unknown and remaining >= 0:
                constraints.append(Constraint(frozenset(adj_unknown), remaining))

        return constraints

    def _tier2_coupled_constraints(self) -> None:
        """Apply subset reduction between constraint pairs."""
        changed = True
        iterations = 0
        max_iterations = 10  # Prevent infinite loops

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            constraints = self._build_constraints()

            # Remove trivially resolved constraints
            constraints = [c for c in constraints if len(c.cells) > 0]

            new_constraints = []
            for i, c1 in enumerate(constraints):
                for j, c2 in enumerate(constraints):
                    if i == j:
                        continue

                    # Subset reduction: if c1.cells ⊂ c2.cells
                    if c1.cells < c2.cells:
                        diff_cells = c2.cells - c1.cells
                        diff_count = c2.count - c1.count

                        if diff_count >= 0 and diff_count <= len(diff_cells):
                            new_c = Constraint(diff_cells, diff_count)
                            if (
                                new_c not in constraints
                                and new_c not in new_constraints
                            ):
                                new_constraints.append(new_c)

            # Apply deductions from all constraints (original + new)
            all_constraints = constraints + new_constraints
            for c in all_constraints:
                if c.count == 0:
                    for cell in c.cells:
                        if cell not in self.safe_cells:
                            self.safe_cells.add(cell)
                            changed = True
                elif c.count == len(c.cells):
                    for cell in c.cells:
                        if cell not in self.mine_cells:
                            self.mine_cells.add(cell)
                            changed = True

            # Re-run Tier 1 if we found new info
            if changed:
                self._tier1_propagation()

    # ================================================================
    # TIER 3: Tank Solver (Backtracking Enumeration)
    # ================================================================

    def _get_frontier(self) -> Set[Tuple[int, int]]:
        """Get frontier cells: unrevealed cells adjacent to revealed numbers."""
        frontier = set()
        for r, c in self.numbers:
            for nr, nc in get_neighbors(r, c, self.rows, self.cols):
                if (
                    (nr, nc) in self.unrevealed
                    and (nr, nc) not in self.safe_cells
                    and (nr, nc) not in self.mine_cells
                ):
                    frontier.add((nr, nc))
        return frontier

    def _get_connected_components(
        self, frontier: Set[Tuple[int, int]]
    ) -> List[Set[Tuple[int, int]]]:
        """Partition frontier into connected components.
        Two frontier cells are connected if they share a constraining number."""
        if not frontier:
            return []

        # Build adjacency: which frontier cells share a constraint?
        cell_to_constraints: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        constraints = self._build_constraints()

        for i, c in enumerate(constraints):
            frontier_cells_in_c = c.cells & frontier
            for cell in frontier_cells_in_c:
                cell_to_constraints[cell].add(i)

        # Union-Find
        parent = {cell: cell for cell in frontier}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Connect cells that share a constraint
        constraint_to_cells: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for cell, cids in cell_to_constraints.items():
            for cid in cids:
                constraint_to_cells[cid].append(cell)

        for cid, cells in constraint_to_cells.items():
            for i in range(1, len(cells)):
                union(cells[0], cells[i])

        # Group by component
        components: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)
        for cell in frontier:
            root = find(cell)
            components[root].add(cell)

        return list(components.values())

    def _enumerate_component(
        self, component: Set[Tuple[int, int]], constraints: List[Constraint]
    ) -> Optional[Dict[int, List[Dict[Tuple[int, int], bool]]]]:
        """Enumerate all valid mine configurations for a connected component.

        Returns: {mine_count: [list of assignments]} where assignment is {cell: is_mine}
                 Or None if timeout.
        """
        cells = sorted(component)
        n = len(cells)

        if n > 35:  # Too large, skip
            return None

        # Filter constraints to those involving this component
        relevant_constraints = []
        for c in constraints:
            overlap = c.cells & component
            if overlap:
                # Adjust count for cells in this constraint that are outside the component
                # (they're already resolved as safe or mine)
                adj_count = c.count
                remaining_cells = []
                for cell in c.cells:
                    if cell in component:
                        remaining_cells.append(cell)
                    elif cell in self.mine_cells:
                        adj_count -= 1
                    # If cell is safe, doesn't affect count

                if remaining_cells:
                    relevant_constraints.append((frozenset(remaining_cells), adj_count))

        # Map cells to indices
        cell_to_idx = {cell: i for i, cell in enumerate(cells)}

        # Convert constraints to index-based
        idx_constraints = []
        for rc_cells, count in relevant_constraints:
            indices = frozenset(cell_to_idx[c] for c in rc_cells if c in cell_to_idx)
            if indices:
                idx_constraints.append((indices, count))

        # Backtracking enumeration
        results: Dict[int, List[Dict[Tuple[int, int], bool]]] = defaultdict(list)
        assignment = [None] * n  # None, True (mine), False (safe)
        start_time = time.time()

        def is_consistent(idx: int) -> bool:
            """Check if current partial assignment is consistent with constraints."""
            for indices, count in idx_constraints:
                if idx not in indices:
                    # Only check constraints involving the just-assigned cell
                    continue

                mines_so_far = sum(1 for i in indices if assignment[i] is True)
                unknown = sum(1 for i in indices if assignment[i] is None)

                # Too many mines already
                if mines_so_far > count:
                    return False
                # Not enough room for remaining mines
                if mines_so_far + unknown < count:
                    return False

            return True

        def backtrack(idx: int, mine_count: int):
            if time.time() - start_time > self.timeout:
                return False  # Timeout

            if idx == n:
                # Valid complete assignment
                config = {cells[i]: assignment[i] for i in range(n)}
                results[mine_count].append(config)
                return True

            # Check global mine count bounds
            remaining_cells_in_component = n - idx
            if mine_count > self.remaining_mines:
                return True  # Prune: too many mines
            if mine_count + remaining_cells_in_component < 0:
                return True  # Always possible

            # Try safe first (most cells are safe)
            assignment[idx] = False
            if is_consistent(idx):
                result = backtrack(idx + 1, mine_count)
                if result is False:
                    return False  # Timeout propagation

            # Try mine
            assignment[idx] = True
            if is_consistent(idx):
                result = backtrack(idx + 1, mine_count + 1)
                if result is False:
                    return False  # Timeout propagation

            assignment[idx] = None
            return True

        success = backtrack(0, 0)
        if not success:
            return None  # Timeout

        return dict(results)

    def _tier3_tank_solver(self) -> None:
        """Full Tank solver with component partitioning and probability computation."""
        frontier = self._get_frontier()
        if not frontier:
            # No frontier — either game is won or we need a random guess
            # Compute interior probabilities (all unrevealed are interior)
            interior = (
                self.unrevealed - self.safe_cells - self.mine_cells - self.flagged
            )
            if interior and self.remaining_mines > 0:
                p = self.remaining_mines / len(interior)
                p = max(0.0, min(1.0, p))
                for cell in interior:
                    self.cell_probabilities[cell] = p
            elif interior:
                for cell in interior:
                    self.cell_probabilities[cell] = 0.0
                    self.safe_cells.add(cell)
            return

        components = self._get_connected_components(frontier)
        constraints = self._build_constraints()

        # Enumerate each component
        component_results = []  # List of {mine_count: [configs]}
        failed_components = []

        for comp in components:
            result = self._enumerate_component(comp, constraints)
            if result is None:
                failed_components.append(comp)
            else:
                component_results.append((comp, result))

        if not component_results:
            return  # All components timed out, stick with Tier 1+2 results

        # Compute probabilities using component results + global mine count
        self._compute_probabilities(component_results, frontier, failed_components)

    def _compute_probabilities(self, component_results, frontier, failed_components):
        """Compute per-cell mine probabilities from enumeration results.

        Uses C(Y, M-m) weighting for interior cells and DP convolution across components.
        """
        # Interior cells: unrevealed, not on frontier, not already resolved
        resolved = self.safe_cells | self.mine_cells | self.flagged
        interior = self.unrevealed - frontier - resolved
        Y = len(interior)  # Interior cell count
        M = self.remaining_mines - len(
            self.mine_cells & self.unrevealed
        )  # Remaining after known mines

        # For failed components, we can't compute exact probabilities
        # Count mines that MUST be in failed components (from Tier 1+2)
        # Count of mines in failed components not needed for current logic

        if not component_results:
            # No successful components
            if interior and M > 0:
                p = M / (len(interior) + len(frontier))
                for cell in interior:
                    self.cell_probabilities[cell] = p
                for cell in frontier:
                    if cell not in self.mine_cells and cell not in self.safe_cells:
                        self.cell_probabilities[cell] = p
            return

        # Simple case: single component
        if len(component_results) == 1 and not failed_components:
            comp, results = component_results[0]
            self._compute_single_component_probs(comp, results, Y, M)
            return

        # Multi-component: DP convolution
        # For simplicity with multiple components, compute per-component independently
        # This is an approximation but good enough for training data
        for comp, results in component_results:
            total_configs = 0
            mine_counts_per_cell = defaultdict(int)

            for mine_count, configs in results.items():
                weight = 1  # Simplified: ignore global weighting for multi-component
                for config in configs:
                    total_configs += weight
                    for cell, is_mine in config.items():
                        if is_mine:
                            mine_counts_per_cell[cell] += weight

            if total_configs > 0:
                for cell in comp:
                    p = mine_counts_per_cell.get(cell, 0) / total_configs
                    self.cell_probabilities[cell] = p

                    if p == 0.0 and cell not in self.safe_cells:
                        self.safe_cells.add(cell)
                    elif p == 1.0 and cell not in self.mine_cells:
                        self.mine_cells.add(cell)

        # Interior cell probability
        if interior:
            frontier_expected_mines = sum(
                self.cell_probabilities.get(c, 0.5)
                for c in frontier
                if c not in self.mine_cells and c not in self.safe_cells
            )
            frontier_known_mines = len(self.mine_cells & frontier)
            remaining_for_interior = max(
                0, M - frontier_expected_mines - frontier_known_mines
            )

            if Y > 0:
                p_interior = remaining_for_interior / Y
                p_interior = max(0.0, min(1.0, p_interior))
                for cell in interior:
                    self.cell_probabilities[cell] = p_interior

    @staticmethod
    def _log_comb(n, k):
        """Log of C(n, k) using lgamma to avoid overflow."""
        if k < 0 or k > n:
            return float("-inf")
        if k == 0 or k == n:
            return 0.0
        return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

    def _compute_single_component_probs(self, comp, results, Y, M):
        """Compute probabilities for a single component with proper C(Y,M-m) weighting.
        Uses log-space to avoid overflow with large binomial coefficients."""
        # Collect (log_weight, config) pairs grouped by validity
        valid_entries = []  # (log_weight, config)

        for mine_count, configs in results.items():
            remaining = M - mine_count
            if remaining < 0 or remaining > Y:
                continue

            log_w = (
                self._log_comb(Y, remaining)
                if Y > 0
                else (0.0 if remaining == 0 else float("-inf"))
            )
            if log_w == float("-inf"):
                continue

            for config in configs:
                valid_entries.append((log_w, config))

        if not valid_entries:
            return

        # Normalize weights using log-sum-exp trick
        max_log_w = max(lw for lw, _ in valid_entries)
        total_weighted = sum(exp(lw - max_log_w) for lw, _ in valid_entries)
        mine_weighted = defaultdict(float)

        for log_w, config in valid_entries:
            w = exp(log_w - max_log_w)
            for cell, is_mine in config.items():
                if is_mine:
                    mine_weighted[cell] += w

        if total_weighted > 0:
            for cell in comp:
                p = mine_weighted.get(cell, 0) / total_weighted
                self.cell_probabilities[cell] = p

                if p == 0.0 and cell not in self.safe_cells:
                    self.safe_cells.add(cell)
                elif p == 1.0 and cell not in self.mine_cells:
                    self.mine_cells.add(cell)

            # Interior probability
            interior = (
                self.unrevealed
                - self._get_frontier()
                - self.safe_cells
                - self.mine_cells
                - self.flagged
            )
            if interior and Y > 0:
                # Weighted average of (M - m) / Y
                expected_interior = 0.0
                for mine_count, configs in results.items():
                    remaining = M - mine_count
                    if remaining < 0 or remaining > Y:
                        continue
                    log_w = self._log_comb(Y, remaining)
                    if log_w == float("-inf"):
                        continue
                    w = exp(log_w - max_log_w) * len(configs)
                    expected_interior += w * remaining / Y

                if total_weighted > 0:
                    p_interior = expected_interior / total_weighted
                    p_interior = max(0.0, min(1.0, p_interior))
                    for cell in interior:
                        self.cell_probabilities[cell] = p_interior

    # ================================================================
    # Public API
    # ================================================================

    def get_certain_moves(self) -> List[Tuple[str, int, int]]:
        """Return list of (action_type, row, col) for all certain moves."""
        moves = []
        for r, c in self.safe_cells:
            if (r, c) in self.unrevealed:
                moves.append(("reveal", r, c))
        for r, c in self.mine_cells:
            if (r, c) in self.unrevealed:
                moves.append(("flag", r, c))
        return moves

    def get_cell_probabilities(self) -> Dict[Tuple[int, int], float]:
        """Return {(r,c): mine_probability} for all unrevealed cells."""
        return dict(self.cell_probabilities)

    def get_best_move(self) -> Tuple[str, int, int, bool]:
        """Return (action_type, row, col, is_deducible) for the best move.

        Priority:
        1. Flag certain mines (zero risk, +15)
        2. Reveal certain safe cells (zero risk, +15)
        3. Reveal lowest-probability cell (guessing)
        """
        certain_moves = self.get_certain_moves()

        # Priority: flag certain mines first
        flag_moves = [(t, r, c) for t, r, c in certain_moves if t == "flag"]
        if flag_moves:
            t, r, c = flag_moves[0]
            return t, r, c, True

        # Then reveal certain safe cells
        reveal_moves = [(t, r, c) for t, r, c in certain_moves if t == "reveal"]
        if reveal_moves:
            t, r, c = reveal_moves[0]
            return t, r, c, True

        # Guessing: pick lowest probability cell
        probs = self.get_cell_probabilities()
        if probs:
            # Only consider unrevealed, unflagged cells
            candidates = {
                cell: p
                for cell, p in probs.items()
                if cell in self.unrevealed
                and cell not in self.safe_cells
                and cell not in self.mine_cells
            }
            if candidates:
                best_cell = min(candidates, key=candidates.get)
                return "reveal", best_cell[0], best_cell[1], False

        # Fallback: pick any unrevealed cell
        remaining = self.unrevealed - self.safe_cells - self.mine_cells
        if remaining:
            cell = min(remaining)  # Deterministic pick
            return "reveal", cell[0], cell[1], False

        # Nothing to do
        return "reveal", 0, 0, False

    def is_logically_deducible(self, action_type: str, row: int, col: int) -> bool:
        """Check if a specific action is logically deducible."""
        if action_type == "reveal" and (row, col) in self.safe_cells:
            return True
        if action_type == "flag" and (row, col) in self.mine_cells:
            return True
        return False


def solve_board(
    board: List[List[str]],
    rows: int,
    cols: int,
    total_mines: int,
    full: bool = True,
    timeout: float = 1.0,
) -> MinesweeperSolver:
    """Convenience function: create solver, run it, return it.

    Args:
        board: 2D visible board
        rows, cols: dimensions
        total_mines: total mine count
        full: if True, run all 3 tiers; if False, only Tier 1+2
        timeout: max seconds for Tank solver per component
    """
    solver = MinesweeperSolver(board, rows, cols, total_mines, timeout=timeout)
    if full:
        solver.solve()
    else:
        solver.solve_fast()
    return solver


# ================================================================
# Testing
# ================================================================

if __name__ == "__main__":
    # Test on a simple board
    board = [
        ["1", "1", ".", ".", "."],
        ["0", "1", ".", ".", "."],
        ["0", "1", "2", ".", "."],
        ["0", "0", "1", ".", "."],
        ["0", "0", "1", ".", "."],
    ]

    solver = solve_board(board, 5, 5, 3)
    print(f"Safe cells: {solver.safe_cells}")
    print(f"Mine cells: {solver.mine_cells}")
    print(f"Probabilities: {solver.cell_probabilities}")

    best = solver.get_best_move()
    print(f"Best move: {best}")

    certain = solver.get_certain_moves()
    print(f"Certain moves: {certain}")
