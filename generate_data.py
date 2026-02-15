#!/usr/bin/env python3
"""
Minesweeper Training Data Generator

Generates training examples by playing games forward with the solver.
Each step of the solver's playthrough becomes a training example.

Uses multiprocessing for parallel generation.
"""

import json
import random
import time
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from solver import solve_board


# ================================================================
# Game Engine (lightweight, no dependencies)
# ================================================================


class MineGame:
    """Lightweight Minesweeper game for data generation."""

    def __init__(self, rows: int, cols: int, mine_positions: List[Tuple[int, int]]):
        self.rows = rows
        self.cols = cols
        self.mine_set: Set[Tuple[int, int]] = set(mine_positions)
        self.num_mines = len(mine_positions)

        # Calculate numbers
        self.internal = [[0] * cols for _ in range(rows)]
        for r, c in self.mine_set:
            self.internal[r][c] = -1
        for r in range(rows):
            for c in range(cols):
                if self.internal[r][c] == -1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and self.internal[nr][nc] == -1
                        ):
                            count += 1
                self.internal[r][c] = count

        self.revealed: Set[Tuple[int, int]] = set()
        self.flagged: Set[Tuple[int, int]] = set()
        self.state = "ongoing"

    def reveal(self, r: int, c: int) -> bool:
        """Reveal a cell with flood fill. Returns False if mine hit."""
        if (r, c) in self.mine_set:
            self.state = "failed"
            return False

        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in self.revealed:
                continue
            self.revealed.add((cr, cc))
            if self.internal[cr][cc] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < self.rows
                            and 0 <= nc < self.cols
                            and (nr, nc) not in self.revealed
                            and (nr, nc) not in self.flagged
                        ):
                            stack.append((nr, nc))

        # Check win
        safe_total = self.rows * self.cols - self.num_mines
        if len(self.revealed) >= safe_total:
            self.state = "success"
        return True

    def flag(self, r: int, c: int):
        """Flag a cell."""
        self.flagged.add((r, c))

    def get_board(self) -> List[List[str]]:
        """Get visible board."""
        board = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for r, c in self.revealed:
            board[r][c] = str(self.internal[r][c])
        for r, c in self.flagged:
            board[r][c] = "F"
        return board

    def cells_revealed_before_action(self) -> int:
        """Count of cells revealed at this point."""
        return len(self.revealed)


# ================================================================
# Prompt Builders
# ================================================================


def build_compact_prompt(
    board: List[List[str]], rows: int, cols: int, num_mines: int, flags_placed: int
) -> str:
    """Build compact grid format prompt for boards <= 16x16."""
    mines_left = num_mines - flags_placed
    grid_lines = ["".join(row) for row in board]
    grid_str = "\n".join(grid_lines)

    return f"""MINESWEEPER {rows}x{cols} MINES:{num_mines} FLAGS:{flags_placed} LEFT:{mines_left}
{grid_str}
RULES: .=hidden F=flag 0-8=adjacent mines
- If number N has N flags around it, remaining hidden neighbors are SAFE->reveal
- If number N needs (N-flags) more mines and has exactly that many hidden neighbors, all are MINES->flag
- Flag certain mines FIRST, then reveal certain safe cells
- NEVER act on already revealed or flagged cells
Output ONLY: {{"type":"reveal"|"flag","row":R,"col":C}}"""


def build_frontier_prompt(
    board: List[List[str]], rows: int, cols: int, num_mines: int, flags_placed: int
) -> str:
    """Build frontier sparse format prompt for boards > 16x16."""
    mines_left = num_mines - flags_placed

    # Find frontier: numbered cells with hidden neighbors
    frontier_info = []
    all_hidden_near_numbers = set()

    for r in range(rows):
        for c in range(cols):
            if board[r][c] not in "012345678":
                continue
            num = int(board[r][c])
            flags = 0
            hidden = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if board[nr][nc] == "F":
                            flags += 1
                        elif board[nr][nc] == ".":
                            hidden.append((nr, nc))
                            all_hidden_near_numbers.add((nr, nc))

            if hidden:  # Only include cells that still have hidden neighbors
                hidden_str = "".join(f"({hr},{hc})" for hr, hc in hidden)
                frontier_info.append(
                    f"R{r}C{c}={num} flags:{flags} hidden:[{hidden_str}]"
                )

    # Count total hidden and interior
    total_hidden = sum(
        1 for r in range(rows) for c in range(cols) if board[r][c] == "."
    )
    interior_count = total_hidden - len(all_hidden_near_numbers)

    frontier_str = "\n".join(frontier_info[:200])  # Cap to prevent token explosion
    hidden_near_str = "".join(
        f"({r},{c})" for r, c in sorted(all_hidden_near_numbers)[:100]
    )

    return f"""MINESWEEPER {rows}x{cols} MINES:{num_mines} FLAGS:{flags_placed} LEFT:{mines_left}
FRONTIER (numbered cells with hidden neighbors):
{frontier_str}
HIDDEN NEAR NUMBERS: {hidden_near_str}
TOTAL HIDDEN: {total_hidden} INTERIOR(no adj number): {interior_count}
RULES: .=hidden F=flag 0-8=adjacent mines
- If number N has N flags around it, remaining hidden neighbors are SAFE->reveal
- If number N needs (N-flags) more mines and has exactly that many hidden neighbors, all are MINES->flag
- Flag certain mines FIRST, then reveal certain safe cells
- NEVER act on already revealed or flagged cells
- Choose ONLY from HIDDEN NEAR NUMBERS cells listed above
Output ONLY: {{"type":"reveal"|"flag","row":R,"col":C}}"""


def build_prompt(
    board: List[List[str]],
    rows: int,
    cols: int,
    num_mines: int,
    flags_placed: int,
    frontier_threshold: int = 16,
) -> str:
    """Build size-adaptive prompt. Compact for small, frontier for large."""
    if rows <= frontier_threshold and cols <= frontier_threshold:
        return build_compact_prompt(board, rows, cols, num_mines, flags_placed)
    else:
        return build_frontier_prompt(board, rows, cols, num_mines, flags_placed)


SYSTEM_PROMPT = (
    "You are an expert Minesweeper AI in a competitive tournament. "
    "Maximize points: +15 safe reveal, +15 correct flag, -25 mine hit, -10 wrong flag, -12 redundant move. "
    "Flag ONLY confirmed mines. Reveal safe cells first. Never target already revealed/flagged cells. "
    'Output ONLY: {"type":"reveal"|"flag","row":R,"col":C}'
)


# ================================================================
# Single Game Generator
# ================================================================


def generate_single_game(args) -> List[Dict]:
    """
    Play one game forward with the solver, capturing training examples.

    Returns list of training example dicts.
    """
    rows, cols, num_mines, game_seed, frontier_threshold = args
    rng = random.Random(game_seed)

    # Generate mine positions
    positions = [(r, c) for r in range(rows) for c in range(cols)]
    mine_positions = rng.sample(positions, num_mines)

    game = MineGame(rows, cols, mine_positions)

    # Random first safe reveal (simulates controller-provided opening)
    safe_cells = [
        (r, c) for r in range(rows) for c in range(cols) if (r, c) not in game.mine_set
    ]
    first_cell = rng.choice(safe_cells)
    game.reveal(*first_cell)

    examples = []
    max_moves = rows * cols * 2  # Safety limit
    move_count = 0

    while game.state == "ongoing" and move_count < max_moves:
        board = game.get_board()
        flags_placed = len(game.flagged)

        # Run solver
        solver = solve_board(board, rows, cols, num_mines, full=True, timeout=1.0)

        # Get best move and all certain moves
        action_type, r, c, is_deducible = solver.get_best_move()
        certain_moves = solver.get_certain_moves()
        all_deducible = [(t, r2, c2) for t, r2, c2 in certain_moves]

        # Build the prompt
        prompt_text = build_prompt(
            board, rows, cols, num_mines, flags_placed, frontier_threshold
        )

        # Build target action
        best_action = {"type": action_type, "row": r, "col": c}
        target_response = json.dumps(best_action)

        # Compute game stage
        total_safe = rows * cols - num_mines
        revealed_count = len(game.revealed)
        reveal_pct = revealed_count / total_safe if total_safe > 0 else 0

        # Build SFT messages format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": target_response},
        ]

        # Build GRPO prompt format
        grpo_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        example = {
            "messages": json.dumps(messages),
            "prompt": json.dumps(grpo_prompt),
            "mine_positions": json.dumps(mine_positions),
            "rows": rows,
            "cols": cols,
            "num_mines": num_mines,
            "flagged_positions": json.dumps(list(game.flagged)),
            "revealed_positions": json.dumps([(r2, c2) for r2, c2 in game.revealed]),
            "board_state": json.dumps(board),
            "deducible_moves": json.dumps(all_deducible),
            "best_move": target_response,
            "is_deducible": is_deducible,
            "game_stage": (
                "opening"
                if reveal_pct < 0.05
                else "early"
                if reveal_pct < 0.15
                else "mid"
                if reveal_pct < 0.50
                else "late"
                if reveal_pct < 0.80
                else "endgame"
            ),
            "board_size": f"{rows}x{cols}",
        }

        examples.append(example)

        # Execute the action
        if action_type == "flag":
            game.flag(r, c)
        else:
            ok = game.reveal(r, c)
            if not ok:
                break  # Hit a mine (shouldn't happen with solver but just in case)

        move_count += 1

        # Balance stage distribution by probabilistic subsampling
        # Without this, late/endgame dominates (~70% of examples)
        keep_prob = 1.0
        if reveal_pct > 0.80:  # Endgame: heavily subsample
            keep_prob = 0.15
        elif reveal_pct > 0.50:  # Late: moderately subsample
            keep_prob = 0.35
        elif reveal_pct > 0.15:  # Mid: slight subsample
            keep_prob = 0.65
        # Opening and early: always keep (keep_prob = 1.0)

        if rng.random() > keep_prob:
            if examples:
                examples.pop()

    return examples


def generate_near_failure_examples(args) -> List[Dict]:
    """
    Generate examples where flags are close to mine count.
    These teach the model NOT to over-flag.
    """
    rows, cols, num_mines, game_seed, frontier_threshold = args
    rng = random.Random(game_seed + 1000000)

    positions = [(r, c) for r in range(rows) for c in range(cols)]
    mine_positions = rng.sample(positions, num_mines)

    game = MineGame(rows, cols, mine_positions)

    safe_cells = [
        (r, c) for r in range(rows) for c in range(cols) if (r, c) not in game.mine_set
    ]
    first_cell = rng.choice(safe_cells)
    game.reveal(*first_cell)

    # Play game forward until flags are close to mine count
    max_moves = rows * cols * 2
    move_count = 0

    while game.state == "ongoing" and move_count < max_moves:
        board = game.get_board()
        solver = solve_board(board, rows, cols, num_mines, full=True, timeout=1.0)
        action_type, r, c, is_deducible = solver.get_best_move()

        if action_type == "flag":
            game.flag(r, c)
        else:
            ok = game.reveal(r, c)
            if not ok:
                break

        move_count += 1

        # Once flags reach 80%+ of mines, capture near-failure examples
        if len(game.flagged) >= int(num_mines * 0.8):
            break

    # Now generate examples from this high-flag state
    examples = []
    if game.state == "ongoing":
        board = game.get_board()
        flags_placed = len(game.flagged)

        solver = solve_board(board, rows, cols, num_mines, full=True, timeout=1.0)
        certain_moves = solver.get_certain_moves()

        # Find reveal moves (not flag moves) - these are the correct actions
        reveal_moves = [(t, r, c) for t, r, c in certain_moves if t == "reveal"]

        if reveal_moves:
            # Pick one
            t, r, c = reveal_moves[0]
            prompt_text = build_prompt(
                board, rows, cols, num_mines, flags_placed, frontier_threshold
            )
            best_action = {"type": "reveal", "row": r, "col": c}
            target_response = json.dumps(best_action)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": target_response},
            ]
            grpo_prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]

            example = {
                "messages": json.dumps(messages),
                "prompt": json.dumps(grpo_prompt),
                "mine_positions": json.dumps(mine_positions),
                "rows": rows,
                "cols": cols,
                "num_mines": num_mines,
                "flagged_positions": json.dumps(list(game.flagged)),
                "revealed_positions": json.dumps(
                    [(r2, c2) for r2, c2 in game.revealed]
                ),
                "board_state": json.dumps(board),
                "deducible_moves": json.dumps(
                    [(t2, r2, c2) for t2, r2, c2 in certain_moves]
                ),
                "best_move": target_response,
                "is_deducible": True,
                "game_stage": "near_failure",
                "board_size": f"{rows}x{cols}",
            }
            examples.append(example)

    return examples


# ================================================================
# Main Generator
# ================================================================


def generate_dataset(
    target_count: int = 50000,
    num_workers: int = None,
    frontier_threshold: int = 16,
    output_file: str = "minesweeper_training_data.jsonl",
    seed: int = 42,
):
    """
    Generate the full training dataset.

    Board size distribution (from plan):
    | Size   | %   | Mine Density |
    | 6x6    | 10% | 10-15%       |
    | 8x8    | 10% | 10-18%       |
    | 10x10  | 15% | 10-18%       |
    | 16x16  | 20% | 12-20%       |
    | 20x20  | 20% | 12-20%       |
    | 30x30  | 15% | 10-20%       |
    | 50x50  | 10% | 10-20%       |
    """
    if num_workers is None:
        num_workers = min(cpu_count(), 32)

    print(f"Generating dataset with {num_workers} workers...")
    print(f"Target: ~{target_count} examples")

    rng = random.Random(seed)

    # Define board configs: (rows, cols, target_game_count, min_density, max_density)
    # Competition: NxM up to 50x50, 10-20% mines. All frontier format.
    board_configs = [
        (6, 6, 1000, 0.10, 0.20),
        (8, 8, 800, 0.10, 0.20),
        (10, 10, 600, 0.10, 0.20),
        (16, 16, 300, 0.10, 0.20),
        (20, 20, 200, 0.10, 0.20),
        (30, 30, 100, 0.10, 0.20),
        (50, 50, 80, 0.10, 0.20),
        # Rectangular boards for NxM coverage
        (8, 12, 300, 0.10, 0.20),
        (10, 16, 200, 0.10, 0.20),
        (12, 20, 150, 0.10, 0.20),
        (16, 30, 100, 0.10, 0.20),
        (20, 40, 60, 0.10, 0.20),
        (30, 50, 40, 0.10, 0.20),
    ]

    # Generate task arguments
    all_args = []
    near_failure_args = []

    for rows, cols, num_games, min_d, max_d in board_configs:
        total_cells = rows * cols
        for i in range(num_games):
            density = rng.uniform(min_d, max_d)
            num_mines = max(1, min(int(total_cells * density), total_cells - 2))
            game_seed = rng.randint(0, 10_000_000)
            all_args.append((rows, cols, num_mines, game_seed, frontier_threshold))

            # 10% near-failure examples
            if rng.random() < 0.1:
                near_failure_args.append(
                    (rows, cols, num_mines, game_seed, frontier_threshold)
                )

    print(
        f"Total games to play: {len(all_args)} + {len(near_failure_args)} near-failure"
    )

    # Process with multiprocessing
    all_examples = []

    print("Generating main examples...")
    t0 = time.time()
    with Pool(num_workers) as pool:
        results = pool.map(generate_single_game, all_args, chunksize=4)
    for game_examples in results:
        all_examples.extend(game_examples)
    t1 = time.time()
    print(f"  Main examples: {len(all_examples)} in {t1 - t0:.1f}s")

    # Near-failure examples
    print("Generating near-failure examples...")
    t0 = time.time()
    with Pool(num_workers) as pool:
        nf_results = pool.map(
            generate_near_failure_examples, near_failure_args, chunksize=4
        )
    nf_count = 0
    for game_examples in nf_results:
        all_examples.extend(game_examples)
        nf_count += len(game_examples)
    t1 = time.time()
    print(f"  Near-failure examples: {nf_count} in {t1 - t0:.1f}s")

    # Shuffle
    rng.shuffle(all_examples)

    # Truncate to target count if needed
    if len(all_examples) > target_count:
        all_examples = all_examples[:target_count]

    print(f"\nFinal dataset: {len(all_examples)} examples")

    # Statistics
    stage_counts = defaultdict(int)
    size_counts = defaultdict(int)
    deducible_count = sum(1 for e in all_examples if e["is_deducible"])

    for e in all_examples:
        stage_counts[e["game_stage"]] += 1
        size_counts[e["board_size"]] += 1

    print("\nBoard size distribution:")
    for size in sorted(
        size_counts.keys(), key=lambda x: (int(x.split("x")[0]), int(x.split("x")[1]))
    ):
        cnt = size_counts[size]
        print(f"  {size}: {cnt} ({cnt / len(all_examples) * 100:.1f}%)")

    print("\nGame stage distribution:")
    for stage in ["opening", "early", "mid", "late", "endgame", "near_failure"]:
        cnt = stage_counts.get(stage, 0)
        print(f"  {stage}: {cnt} ({cnt / len(all_examples) * 100:.1f}%)")

    print(
        f"\nDeducible: {deducible_count} ({deducible_count / len(all_examples) * 100:.1f}%)"
    )

    # Save as JSONL
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w") as f:
        for example in all_examples:
            f.write(json.dumps(example) + "\n")

    print(f"Done! {len(all_examples)} examples saved.")
    return all_examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Minesweeper training data")
    parser.add_argument(
        "--target", type=int, default=50000, help="Target number of examples"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="minesweeper_training_data.jsonl",
        help="Output file",
    )
    parser.add_argument(
        "--frontier-threshold",
        type=int,
        default=16,
        help="Board size threshold for frontier format",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_dataset(
        target_count=args.target,
        num_workers=args.workers,
        frontier_threshold=args.frontier_threshold,
        output_file=args.output,
        seed=args.seed,
    )
