#!/usr/bin/python3
"""
Minesweeper Agent
Size-adaptive prompt builder with compact grid (<=16x16) and frontier sparse (>16x16)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from .minesweeper_model import MinesweeperAgent

# Safety net flag - set True to enable post-LLM validation
SAFETY_NET_ENABLED = False


class MinesweeperPlayer:
    """Agent responsible for playing Minesweeper"""

    FRONTIER_THRESHOLD = 0  # Use frontier format for ALL boards (validated: 100% valid moves vs 10% with compact)

    def __init__(self, **kwargs):
        self.agent = MinesweeperAgent(**kwargs)

    def build_prompt(self, game_state: Dict[str, Any]) -> tuple[str, str]:
        """
        Generate size-adaptive prompt from game state.
        Compact grid for small boards, frontier sparse for large boards.

        Args:
            game_state: Dictionary containing board, rows, cols, mines, etc.

        Returns:
            (prompt, system_prompt)
        """
        sys_prompt = (
            "You are a Minesweeper AI. "
            'Output ONLY valid JSON: {"type":"reveal"|"flag","row":R,"col":C}'
        )

        # Strip internal metadata keys (prefixed with _)
        board = game_state["board"]
        rows = game_state["rows"]
        cols = game_state["cols"]
        num_mines = game_state.get("mines", game_state.get("num_mines", 0))
        flags_placed = game_state.get("flags_placed", 0)

        # Count flags from board if not provided
        if flags_placed == 0:
            flags_placed = sum(1 for r in board for c in r if c == "F")

        mines_left = num_mines - flags_placed

        if rows <= self.FRONTIER_THRESHOLD and cols <= self.FRONTIER_THRESHOLD:
            prompt = self._build_compact_prompt(
                board, rows, cols, num_mines, flags_placed, mines_left
            )
        else:
            prompt = self._build_frontier_prompt(
                board, rows, cols, num_mines, flags_placed, mines_left
            )

        return prompt, sys_prompt

    def _build_compact_prompt(
        self, board, rows, cols, num_mines, flags_placed, mines_left
    ):
        """Compact grid format for boards <= 16x16."""
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

    def _build_frontier_prompt(
        self, board, rows, cols, num_mines, flags_placed, mines_left
    ):
        """Frontier sparse format for boards > 16x16."""
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

                if hidden:
                    hidden_str = "".join(f"({hr},{hc})" for hr, hc in hidden)
                    frontier_info.append(
                        f"R{r}C{c}={num} flags:{flags} hidden:[{hidden_str}]"
                    )

        total_hidden = sum(
            1 for r in range(rows) for c in range(cols) if board[r][c] == "."
        )
        interior_count = total_hidden - len(all_hidden_near_numbers)

        # Cap frontier info to prevent token explosion
        frontier_str = "\n".join(frontier_info[:200])
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

    def play_action(
        self, game_state: Dict[str, Any], **gen_kwargs
    ) -> tuple[Optional[Dict], Optional[int], Optional[float]]:
        """
        Generate a single action for the given game state.

        Returns:
            (action_dict, token_count, generation_time)
        """
        prompt, sys_prompt = self.build_prompt(game_state)
        response, tl, gt = self.agent.generate_response(
            prompt, sys_prompt, **gen_kwargs
        )

        action = self.parse_action(response)

        # Optional safety net validation
        if SAFETY_NET_ENABLED and action is not None:
            action = self._validate_action(action, game_state)

        return action, tl, gt

    def _validate_action(self, action: Dict, game_state: Dict) -> Dict:
        """Safety net: validate action against game state."""
        board = game_state["board"]
        rows = game_state["rows"]
        cols = game_state["cols"]
        num_mines = game_state.get("mines", game_state.get("num_mines", 0))
        row, col = action["row"], action["col"]

        # Clamp out of bounds
        row = max(0, min(row, rows - 1))
        col = max(0, min(col, cols - 1))
        action["row"] = row
        action["col"] = col

        # If targeting already revealed/flagged cell, find nearest unrevealed
        cell_val = board[row][col]
        if cell_val != ".":
            # Find nearest unrevealed cell
            best_dist = float("inf")
            best_cell = None
            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == ".":
                        dist = abs(r - row) + abs(c - col)
                        if dist < best_dist:
                            best_dist = dist
                            best_cell = (r, c)
            if best_cell:
                action["row"], action["col"] = best_cell
                action["type"] = "reveal"  # Safe default

        # Prevent over-flagging
        if action["type"] == "flag":
            flags_count = sum(1 for r in board for c in r if c == "F")
            if flags_count >= num_mines:
                action["type"] = "reveal"

        return action

    def parse_action(self, response: str) -> Optional[Dict]:
        """
        Extract JSON action from LLM response.
        Finds first valid action JSON object.
        """
        try:
            potential_jsons = []
            i = 0
            while i < len(response):
                start = response.find("{", i)
                if start == -1:
                    break

                brace_count = 0
                end = start
                while end < len(response):
                    if response[end] == "{":
                        brace_count += 1
                    elif response[end] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = response[start : end + 1]
                            try:
                                obj = json.loads(json_str)
                                potential_jsons.append(obj)
                            except (json.JSONDecodeError, ValueError):
                                pass
                            break
                    end += 1

                i = end + 1 if end < len(response) else len(response)

            for obj in potential_jsons:
                if (
                    isinstance(obj, dict)
                    and "type" in obj
                    and "row" in obj
                    and "col" in obj
                    and obj["type"] in ["reveal", "flag"]
                ):
                    obj["row"] = int(obj["row"])
                    obj["col"] = int(obj["col"])
                    return obj

        except Exception as e:
            print(f"Failed to parse action: {e}")
            return None

        return None

    @staticmethod
    def save_action(action: Dict, file_path: str | Path) -> None:
        """Save action to JSON file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(action, f, indent=2)


# Example usage
if __name__ == "__main__":
    import argparse
    import yaml

    argparser = argparse.ArgumentParser(
        description="Play Minesweeper using fine-tuned LLM."
    )
    argparser.add_argument(
        "--game_state_file",
        type=str,
        required=True,
        help="Input JSON file containing game state",
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="outputs/action.json",
        help="Output file to save the action",
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for debugging."
    )
    args = argparser.parse_args()

    with open(args.game_state_file, "r") as f:
        game_state = json.load(f)

    player = MinesweeperPlayer()

    gen_kwargs = {"tgps_show": args.verbose}
    config_file = Path("minesweeper_config.yaml")
    if config_file.exists():
        with open(config_file, "r") as f:
            gen_kwargs.update(yaml.safe_load(f))

    action, tl, gt = player.play_action(game_state, **gen_kwargs)

    if args.verbose:
        print("Game State:")
        print(json.dumps(game_state, indent=2))
        print("\nGenerated Action:")
        print(json.dumps(action, indent=2))
        if tl and gt:
            print(f"\nStats: Tokens={tl}, Time={gt:.2f}s, TGPS={tl / gt:.2f}")

    if action:
        player.save_action(action, args.output_file)
        print(f"Action saved to {args.output_file}")
    else:
        print("ERROR: Failed to generate valid action!")
        player.save_action({"error": "parse_failed"}, args.output_file)
