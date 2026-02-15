#!/usr/bin/env python3
"""Prompt Battle: Test multiple system prompts to find the best performer."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

import json
import re
import random
import time
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


def build_frontier_prompt(board, rows, cols, mines, flags, extra_rules=""):
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

    base = f"""MINESWEEPER {rows}x{cols} MINES:{mines} FLAGS:{flags} LEFT:{mines_left}
FRONTIER (numbered cells with hidden neighbors):
{frontier_str}
HIDDEN NEAR NUMBERS: {hidden_near_str}
TOTAL HIDDEN: {total_hidden} INTERIOR(no adj number): {interior_count}
RULES: .=hidden F=flag 0-8=adjacent mines
- If number N has N flags around it, remaining hidden neighbors are SAFE->reveal
- If number N needs (N-flags) more mines and has exactly that many hidden neighbors, all are MINES->flag
- Flag certain mines FIRST, then reveal certain safe cells
- NEVER act on already revealed or flagged cells"""

    if extra_rules:
        base += "\n" + extra_rules

    base += '\nOutput ONLY: {"type":"reveal"|"flag","row":R,"col":C}'
    return base


# === PROMPT STRATEGIES ===

PROMPTS = {
    "baseline": {
        "sys": 'You are a Minesweeper AI. Output ONLY valid JSON: {"type":"reveal"|"flag","row":R,"col":C}',
        "extra": "",
    },
    "risk_averse": {
        "sys": 'You are a cautious Minesweeper AI. NEVER guess. Only act on cells you are 100% CERTAIN about. Output ONLY valid JSON: {"type":"reveal"|"flag","row":R,"col":C}',
        "extra": "- CRITICAL: Only reveal a cell if you can PROVE it is safe. If unsure, flag instead.",
    },
    "step_by_step": {
        "sys": 'You are a Minesweeper AI. For each numbered cell, count: needed_mines = number - flags. If needed_mines == hidden_count, flag all hidden. If needed_mines == 0, reveal all hidden. Output ONLY valid JSON: {"type":"reveal"|"flag","row":R,"col":C}',
        "extra": "",
    },
    "flag_first": {
        "sys": 'You are a Minesweeper AI that prioritizes flagging mines. Always look for cells that MUST be mines first. Output ONLY valid JSON: {"type":"reveal"|"flag","row":R,"col":C}',
        "extra": "- PRIORITY: Find and flag definite mines before revealing safe cells\n- A cell MUST be a mine if: needed_mines == hidden_count for any adjacent number",
    },
    "reveal_zero": {
        "sys": 'You are a Minesweeper AI. Prioritize revealing neighbors of 0-cells and cells adjacent to low numbers with all mines found. Output ONLY valid JSON: {"type":"reveal"|"flag","row":R,"col":C}',
        "extra": "- PRIORITY: Reveal cells next to numbers whose mines are ALL flagged (needed=0) for maximum safe reveals",
    },
    "deduction": {
        "sys": "You are a logical Minesweeper solver. Apply constraint deduction: for each number, remaining_mines = number - adjacent_flags. If remaining_mines equals adjacent_hidden_count, all hidden are mines. If remaining_mines is 0, all hidden are safe. Output ONLY valid JSON.",
        "extra": "- Apply deduction to EVERY frontier cell before choosing\n- Choose ONLY cells where the logic is conclusive",
    },
}

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "/workspace/your_finetuned_model_v2", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/workspace/your_finetuned_model_v2")

# Test on same games for fair comparison
board_configs = [
    (8, 8, 10, 15),
    (10, 10, 15, 10),
]

print(f"\n{'=' * 70}")
print(f"  PROMPT BATTLE ROYALE — {len(PROMPTS)} contenders")
print(f"{'=' * 70}\n")

results = {}

for prompt_name, prompt_cfg in PROMPTS.items():
    sys_prompt = prompt_cfg["sys"]
    extra_rules = prompt_cfg["extra"]

    total_score = 0
    total_games = 0
    total_moves = 0
    total_valid = 0
    total_wins = 0
    total_mine_hits = 0
    total_safe_reveals = 0
    total_correct_flags = 0
    total_wrong_flags = 0

    t0 = time.time()

    for rows, cols, mines, n_games in board_configs:
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

            max_moves = min(30, rows * cols)
            game_score = 0

            for move_i in range(max_moves):
                if game.state != "ongoing":
                    if game.state == "success":
                        total_wins += 1
                        game_score += 50
                    break

                board = game.get_board()
                flags = len(game.flagged)
                prompt = build_frontier_prompt(
                    board, rows, cols, mines, flags, extra_rules
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
                        max_new_tokens=128,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                response = tokenizer.decode(
                    output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )

                action = parse_llm_action(response)
                total_moves += 1

                if action is not None:
                    r_act, c_act = action["row"], action["col"]
                    if 0 <= r_act < rows and 0 <= c_act < cols:
                        cell_val = board[r_act][c_act]
                        if cell_val == ".":
                            total_valid += 1
                            if action["type"] == "reveal":
                                result = game.reveal(r_act, c_act)
                                if result == "mine":
                                    game_score -= 25
                                    total_mine_hits += 1
                                else:
                                    game_score += 15
                                    total_safe_reveals += 1
                            elif action["type"] == "flag":
                                game.flag(r_act, c_act)
                                if (r_act, c_act) in game.mine_set:
                                    game_score += 15
                                    total_correct_flags += 1
                                else:
                                    game_score -= 10
                                    total_wrong_flags += 1
                        else:
                            game_score -= 12
                    else:
                        game_score -= 15
                else:
                    game_score -= 10

            total_score += game_score
            total_games += 1

    elapsed = time.time() - t0
    avg_score = total_score / max(total_games, 1)
    valid_rate = total_valid / max(total_moves, 1) * 100
    mine_rate = total_mine_hits / max(total_safe_reveals + total_mine_hits, 1) * 100
    flag_acc = (
        total_correct_flags / max(total_correct_flags + total_wrong_flags, 1) * 100
    )

    results[prompt_name] = {
        "avg_score": avg_score,
        "wins": total_wins,
        "games": total_games,
        "valid_rate": valid_rate,
        "mine_rate": mine_rate,
        "flag_acc": flag_acc,
        "safe_reveals": total_safe_reveals,
        "mine_hits": total_mine_hits,
        "correct_flags": total_correct_flags,
        "wrong_flags": total_wrong_flags,
        "moves": total_moves,
        "time": elapsed,
    }

    print(
        f"[{prompt_name}] AvgScore={avg_score:+.1f} | Wins={total_wins}/{total_games} | "
        f"Valid={valid_rate:.0f}% | MineHitRate={mine_rate:.0f}% | FlagAcc={flag_acc:.0f}% | "
        f"Reveals={total_safe_reveals}ok/{total_mine_hits}mine | Flags={total_correct_flags}ok/{total_wrong_flags}bad | "
        f"{elapsed:.0f}s"
    )

# === LEADERBOARD ===
print(f"\n{'=' * 70}")
print("  LEADERBOARD (sorted by avg score)")
print(f"{'=' * 70}")
ranked = sorted(results.items(), key=lambda x: x[1]["avg_score"], reverse=True)
for i, (name, r) in enumerate(ranked):
    medal = ["🥇", "🥈", "🥉", "4.", "5.", "6."][i]
    print(
        f"  {medal} {name:20s} | AvgScore={r['avg_score']:+6.1f} | MineHit={r['mine_rate']:.0f}% | FlagAcc={r['flag_acc']:.0f}% | Wins={r['wins']}"
    )

# Show best vs baseline improvement
baseline_score = results["baseline"]["avg_score"]
best_name, best_r = ranked[0]
diff = best_r["avg_score"] - baseline_score
print(f"\n  Best vs Baseline: {diff:+.1f} points ({best_name})")
print("\nDone!")
