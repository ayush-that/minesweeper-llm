#!/usr/bin/env python3
"""Evaluation with improved prompts, max_new_tokens=128, 50x50 boards.
Tests both v1 and v2 models if available."""

import os

os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

import json
import re
import random
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================================================================
# Game Engine
# ================================================================


class MinesweeperGame:
    def __init__(self, rows, cols, mine_positions):
        self.rows = rows
        self.cols = cols
        self.mine_set = set(tuple(p) for p in mine_positions)
        self.num_mines = len(self.mine_set)
        self._board = [[0] * cols for _ in range(rows)]
        for r, c in self.mine_set:
            self._board[r][c] = -1
        for r in range(rows):
            for c in range(cols):
                if self._board[r][c] == -1:
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
                            and self._board[nr][nc] == -1
                        ):
                            count += 1
                self._board[r][c] = count
        self.revealed = set()
        self.flagged = set()
        self._state = "ongoing"

    def reveal(self, r, c):
        if (r, c) in self.mine_set:
            self._state = "failed"
            return "mine"
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in self.revealed:
                continue
            self.revealed.add((cr, cc))
            if self._board[cr][cc] == 0:
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
        safe_total = self.rows * self.cols - self.num_mines
        if len(self.revealed) >= safe_total:
            self._state = "success"
            return "win"
        return "ok"

    def flag(self, r, c):
        self.flagged.add((r, c))

    def get_board(self):
        board = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for r, c in self.revealed:
            board[r][c] = str(self._board[r][c])
        for r, c in self.flagged:
            board[r][c] = "F"
        return board

    @property
    def state(self):
        return self._state


def parse_llm_action(response):
    best = None
    for match in re.finditer(r"\{[^{}]*\}", response):
        try:
            action = json.loads(match.group())
            if (
                "type" in action
                and "row" in action
                and "col" in action
                and action["type"] in ["reveal", "flag"]
            ):
                action["row"] = int(action["row"])
                action["col"] = int(action["col"])
                best = action
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return best


def build_frontier_prompt(board, rows, cols, mines, flags):
    mines_left = mines - flags
    frontier_info = []
    all_hidden_near_numbers = set()
    for r in range(rows):
        for c in range(cols):
            if board[r][c] not in "012345678":
                continue
            num = int(board[r][c])
            fl = sum(
                1
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if not (dr == 0 and dc == 0)
                and 0 <= r + dr < rows
                and 0 <= c + dc < cols
                and board[r + dr][c + dc] == "F"
            )
            hidden = [
                (r + dr, c + dc)
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if not (dr == 0 and dc == 0)
                and 0 <= r + dr < rows
                and 0 <= c + dc < cols
                and board[r + dr][c + dc] == "."
            ]
            if hidden:
                for h in hidden:
                    all_hidden_near_numbers.add(h)
                hs = "".join(f"({hr},{hc})" for hr, hc in hidden)
                frontier_info.append(f"R{r}C{c}={num} flags:{fl} hidden:[{hs}]")
    total_hidden = sum(
        1 for r in range(rows) for c in range(cols) if board[r][c] == "."
    )
    interior_count = total_hidden - len(all_hidden_near_numbers)
    frontier_str = "\n".join(frontier_info[:200])
    hidden_near_str = "".join(
        f"({r},{c})" for r, c in sorted(all_hidden_near_numbers)[:100]
    )
    return (
        f"MINESWEEPER {rows}x{cols} MINES:{mines} FLAGS:{flags} LEFT:{mines_left}\n"
        f"FRONTIER (numbered cells with hidden neighbors):\n{frontier_str}\n"
        f"HIDDEN NEAR NUMBERS: {hidden_near_str}\n"
        f"TOTAL HIDDEN: {total_hidden} INTERIOR(no adj number): {interior_count}\n"
        f"RULES: .=hidden F=flag 0-8=adjacent mines\n"
        f"- If number N has N flags around it, remaining hidden neighbors are SAFE->reveal\n"
        f"- If number N needs (N-flags) more mines and has exactly that many hidden neighbors, all are MINES->flag\n"
        f"- Flag certain mines FIRST, then reveal certain safe cells\n"
        f"- NEVER act on already revealed or flagged cells\n"
        f"- Choose ONLY from HIDDEN NEAR NUMBERS cells listed above\n"
        f'Output ONLY: {{"type":"reveal"|"flag","row":R,"col":C}}'
    )


# System prompts to test
SYS_PROMPTS = {
    "original": "You are an expert Minesweeper AI. Analyze constraints and output ONLY a valid JSON action. No explanation.",
    "tournament": (
        "You are an expert Minesweeper AI in a competitive tournament. "
        "Maximize points: +15 safe reveal, +15 correct flag, -25 mine hit, -10 wrong flag, -12 redundant move. "
        "Flag ONLY confirmed mines. Reveal safe cells first. Never target already revealed/flagged cells. "
        'Output ONLY: {"type":"reveal"|"flag","row":R,"col":C}'
    ),
}
SYS_PROMPT = SYS_PROMPTS.get(
    os.environ.get("EVAL_PROMPT", "tournament"), SYS_PROMPTS["tournament"]
)


def evaluate_model(
    model_path, model_name_label, board_configs=None, max_new_tokens=128
):
    """Run full evaluation on a model."""
    print(f"\n{'=' * 60}")
    print(f"EVALUATING: {model_name_label}")
    print(f"Model: {model_path}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"{'=' * 60}\n")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if board_configs is None:
        board_configs = [
            (6, 6, 5, 15),
            (8, 8, 10, 15),
            (10, 10, 15, 15),
            (16, 16, 40, 8),
            (20, 20, 60, 8),
            (30, 30, 120, 3),
            (50, 50, 400, 2),
        ]

    overall_score = 0
    overall_games = 0
    overall_moves = 0
    overall_valid = 0
    overall_wins = 0

    for rows, cols, mines, n_games in board_configs:
        valid_json = 0
        valid_moves = 0
        total_moves = 0
        wins = 0
        total_score = 0
        reveal_safe = 0
        reveal_mine = 0
        flag_correct = 0
        flag_wrong = 0
        redundant = 0
        oob = 0
        invalid_json = 0
        verbose_count = 0

        for seed in range(n_games):
            rng = random.Random(seed + 30000)
            positions = [(r, c) for r in range(rows) for c in range(cols)]
            mine_pos = rng.sample(positions, mines)
            game = MinesweeperGame(rows, cols, mine_pos)
            safe = [
                (r, c)
                for r in range(rows)
                for c in range(cols)
                if (r, c) not in game.mine_set
            ]
            first = rng.choice(safe)
            game.reveal(*first)

            # Competition: max_moves = 2*N*M
            max_moves = min(2 * rows * cols, 500)
            game_score = 0
            last_action = None
            repeat_count = 0

            for move_i in range(max_moves):
                if game.state != "ongoing":
                    if game.state == "success":
                        wins += 1
                        game_score += 50
                    break

                board = game.get_board()
                flags = len(game.flagged)
                prompt = build_frontier_prompt(board, rows, cols, mines, flags)
                messages = [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                response = tokenizer.decode(
                    output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                resp_tokens = output[0].shape[0] - inputs.input_ids.shape[1]

                # Track verbosity
                if resp_tokens > 40:
                    verbose_count += 1

                action = parse_llm_action(response)
                total_moves += 1

                # Loop detection
                if action is not None:
                    action_tuple = (action["type"], action["row"], action["col"])
                    if action_tuple == last_action:
                        repeat_count += 1
                        if repeat_count >= 3:
                            break
                    else:
                        repeat_count = 1
                        last_action = action_tuple

                if action is not None:
                    valid_json += 1
                    r_act, c_act = action["row"], action["col"]
                    if 0 <= r_act < rows and 0 <= c_act < cols:
                        cell_val = board[r_act][c_act]
                        if cell_val == ".":
                            valid_moves += 1
                            if action["type"] == "reveal":
                                result = game.reveal(r_act, c_act)
                                if result == "mine":
                                    game_score -= 25
                                    reveal_mine += 1
                                else:
                                    game_score += 15
                                    reveal_safe += 1
                            elif action["type"] == "flag":
                                game.flag(r_act, c_act)
                                if (r_act, c_act) in game.mine_set:
                                    game_score += 15
                                    flag_correct += 1
                                else:
                                    game_score -= 10
                                    flag_wrong += 1
                        else:
                            game_score -= 12
                            redundant += 1
                    else:
                        game_score -= 15
                        oob += 1
                else:
                    game_score -= 10
                    invalid_json += 1

            total_score += game_score

        json_rate = valid_json / max(total_moves, 1) * 100
        move_rate = valid_moves / max(total_moves, 1) * 100
        avg_score = total_score / max(n_games, 1)

        print(f"{rows}x{cols} ({n_games} games, {total_moves} moves):")
        print(
            f"  JSON={json_rate:.0f}% ValidMove={move_rate:.0f}% Wins={wins}/{n_games} AvgScore={avg_score:.1f}"
        )
        print(
            f"  RevealSafe={reveal_safe} RevealMine={reveal_mine} FlagCorrect={flag_correct} FlagWrong={flag_wrong}"
        )
        print(
            f"  Redundant={redundant} OOB={oob} InvalidJSON={invalid_json} Verbose={verbose_count}"
        )

        overall_score += total_score
        overall_games += n_games
        overall_moves += total_moves
        overall_valid += valid_moves
        overall_wins += wins

    print(f"\n--- OVERALL ({model_name_label}) ---")
    print(f"Games: {overall_games}, Moves: {overall_moves}, Wins: {overall_wins}")
    print(
        f"Valid moves: {overall_valid}/{overall_moves} ({overall_valid / max(overall_moves, 1) * 100:.0f}%)"
    )
    print(
        f"Total score: {overall_score}, Avg per game: {overall_score / max(overall_games, 1):.1f}"
    )

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return overall_score, overall_games, overall_wins


if __name__ == "__main__":
    v1_path = "/workspace/your_finetuned_model"
    v2_path = "/workspace/your_finetuned_model_v2"
    v3_path = "/workspace/your_finetuned_model_v3"

    target = sys.argv[1] if len(sys.argv) > 1 else "v1"

    if target == "v3":
        evaluate_model(v3_path, "SFT-v3 (targeted, same prompt)")
    elif target == "v2":
        evaluate_model(v2_path, "SFT-v2 (continued, all-frontier)")
    elif target == "both":
        evaluate_model(v1_path, "SFT-v1 (original)")
        if os.path.exists(v3_path):
            evaluate_model(v3_path, "SFT-v3 (targeted, same prompt)")
    else:
        evaluate_model(v1_path, "SFT-v1 (original)")
