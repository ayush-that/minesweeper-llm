#!/usr/bin/env python3
"""Compare SFT vs GRPO models with different FRONTIER_THRESHOLDs."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

import json
import re
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def build_prompt(board, rows, cols, mines, flags, frontier_threshold):
    mines_left = mines - flags
    if rows <= frontier_threshold and cols <= frontier_threshold:
        grid = "\n".join("".join(r) for r in board)
        return f'MINESWEEPER {rows}x{cols} MINES:{mines} FLAGS:{flags} LEFT:{mines_left}\n{grid}\nRULES: .=hidden F=flag 0-8=adjacent mines\n- If number N has N flags around it, remaining hidden neighbors are SAFE->reveal\n- If number N needs (N-flags) more mines and has exactly that many hidden neighbors, all are MINES->flag\n- Flag certain mines FIRST, then reveal certain safe cells\n- NEVER act on already revealed or flagged cells\nOutput ONLY: {{"type":"reveal"|"flag","row":R,"col":C}}'
    else:
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
        return f'MINESWEEPER {rows}x{cols} MINES:{mines} FLAGS:{flags} LEFT:{mines_left}\nFRONTIER (numbered cells with hidden neighbors):\n{frontier_str}\nHIDDEN NEAR NUMBERS: {hidden_near_str}\nTOTAL HIDDEN: {total_hidden} INTERIOR(no adj number): {interior_count}\nRULES: .=hidden F=flag 0-8=adjacent mines\n- If number N has N flags around it, remaining hidden neighbors are SAFE->reveal\n- If number N needs (N-flags) more mines and has exactly that many hidden neighbors, all are MINES->flag\n- Flag certain mines FIRST, then reveal certain safe cells\n- NEVER act on already revealed or flagged cells\nOutput ONLY: {{"type":"reveal"|"flag","row":R,"col":C}}'


def eval_model(
    model, tokenizer, board_configs, frontier_threshold, max_moves_per_game=15
):
    sys_prompt = "You are an expert Minesweeper AI. Analyze constraints and output ONLY a valid JSON action. No explanation."
    results = {}
    for rows, cols, mines, n_games in board_configs:
        valid_json = 0
        valid_moves = 0
        total_moves = 0
        wins = 0
        total_score = 0
        for seed in range(n_games):
            rng = random.Random(seed + 20000)
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
            game_score = 0
            for move_i in range(max_moves_per_game):
                if game.state != "ongoing":
                    if game.state == "success":
                        wins += 1
                        game_score += 50
                    break
                board = game.get_board()
                flags = len(game.flagged)
                prompt = build_prompt(
                    board, rows, cols, mines, flags, frontier_threshold
                )
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                response = tokenizer.decode(
                    output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                action = parse_llm_action(response)
                total_moves += 1
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
                                else:
                                    game_score += 15
                            elif action["type"] == "flag":
                                game.flag(r_act, c_act)
                                if (r_act, c_act) in game.mine_set:
                                    game_score += 15
                                else:
                                    game_score -= 10
                        elif cell_val == "F":
                            game_score -= 8
                        else:
                            game_score -= 12  # already revealed
                    else:
                        game_score -= 15  # OOB
                else:
                    game_score -= 10  # invalid JSON
            total_score += game_score
        json_rate = valid_json / max(total_moves, 1) * 100
        move_rate = valid_moves / max(total_moves, 1) * 100
        avg_score = total_score / max(n_games, 1)
        key = f"{rows}x{cols}"
        results[key] = {
            "json_rate": json_rate,
            "move_rate": move_rate,
            "wins": wins,
            "n_games": n_games,
            "avg_score": avg_score,
            "total_moves": total_moves,
        }
        print(
            f"  {key}: JSON={json_rate:.0f}% ValidMove={move_rate:.0f}% Wins={wins}/{n_games} AvgScore={avg_score:.1f} ({total_moves} moves)"
        )
    return results


board_configs = [
    (6, 6, 5, 10),
    (8, 8, 10, 10),
    (10, 10, 15, 10),
    (16, 16, 40, 5),
    (20, 20, 60, 5),
]

# Test both models with frontier_threshold=0 (all frontier format)
for model_name, model_path in [
    ("SFT-only", "/workspace/sft_merged_model"),
    ("GRPO-400", "/workspace/your_finetuned_model"),
]:
    print(f"\n{'=' * 60}")
    print(f"Model: {model_name} ({model_path})")
    print(f"{'=' * 60}")

    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test with frontier threshold = 0 (all frontier)
    print("\n--- Frontier for ALL boards (threshold=0) ---")
    eval_model(model, tokenizer, board_configs, frontier_threshold=0)

    # Test with frontier threshold = 16 (original)
    print("\n--- Compact ≤16, Frontier >16 (threshold=16) ---")
    eval_model(model, tokenizer, board_configs, frontier_threshold=16)

    del model
    torch.cuda.empty_cache()
    import gc

    gc.collect()

print("\nComparison complete!")
