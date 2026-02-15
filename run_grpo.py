#!/usr/bin/env python3
"""
Run GRPO training starting from SFT checkpoint.
SFT is already complete - this script loads the SFT model and runs GRPO + eval + save.
"""

import os

os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

import json
import re
import random
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

# ================================================================
# Step 1: Load SFT checkpoint directly via Unsloth
# ================================================================
print("Loading SFT checkpoint via Unsloth...")
max_seq_length = 8192

# Unsloth can load a PEFT adapter checkpoint directly
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/workspace/sft_checkpoint",
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
)

print(f"Model loaded. Device: {model.device}")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(
    f"Trainable: {trainable / 1e6:.1f}M / {total / 1e9:.1f}B ({trainable / total * 100:.2f}%)"
)

# ================================================================
# Step 2: Game Engine + Reward Functions
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


def reconstruct_game(mine_positions, rows, cols, revealed_positions, flagged_positions):
    game = MinesweeperGame(rows, cols, mine_positions)
    for r, c in revealed_positions:
        if (r, c) not in game.revealed:
            game.reveal(r, c)
    for r, c in flagged_positions:
        game.flag(r, c)
    game._state = "ongoing"
    return game


def format_reward(completions, **kwargs):
    scores = []
    for completion in completions:
        response = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )
        action = parse_llm_action(response)
        scores.append(1.0 if action is not None else -3.0)
    return scores


def gameplay_reward(completions, **kwargs):
    mine_positions_list = kwargs.get("mine_positions", [])
    rows_list = kwargs.get("rows", [])
    cols_list = kwargs.get("cols", [])
    num_mines_list = kwargs.get("num_mines", [])
    flagged_positions_list = kwargs.get("flagged_positions", [])
    _revealed_positions_list = kwargs.get("revealed_positions", [])
    deducible_moves_list = kwargs.get("deducible_moves", [])

    scores = []
    for idx, completion in enumerate(completions):
        response = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )
        action = parse_llm_action(response)
        if action is None:
            scores.append(-10.0 / 25.0)
            continue
        try:
            mine_pos = (
                json.loads(mine_positions_list[idx])
                if isinstance(mine_positions_list[idx], str)
                else mine_positions_list[idx]
            )
            rows = int(rows_list[idx])
            cols = int(cols_list[idx])
            num_mines = int(num_mines_list[idx])
            flagged_pos = (
                json.loads(flagged_positions_list[idx])
                if isinstance(flagged_positions_list[idx], str)
                else flagged_positions_list[idx]
            )
            revealed_pos = (
                json.loads(_revealed_positions_list[idx])
                if isinstance(_revealed_positions_list[idx], str)
                else _revealed_positions_list[idx]
            )
            deducible_raw = (
                json.loads(deducible_moves_list[idx])
                if isinstance(deducible_moves_list[idx], str)
                else deducible_moves_list[idx]
            )
            mine_set = set(tuple(p) for p in mine_pos)
            flagged_set = set(tuple(p) for p in flagged_pos)
            revealed_set = set(tuple(p) for p in revealed_pos)
            deducible_set = set((m[0], m[1], m[2]) for m in deducible_raw)
            row, col = action["row"], action["col"]
            action_type = action["type"]
            if not (0 <= row < rows and 0 <= col < cols):
                scores.append(-15.0 / 25.0)
                continue
            if (row, col) in revealed_set:
                scores.append(-12.0 / 25.0)
                continue
            if (row, col) in flagged_set:
                scores.append(-8.0 / 25.0)
                continue
            if action_type == "flag":
                if len(flagged_set) >= num_mines:
                    scores.append(-10.0 / 25.0)
                    continue
                if (row, col) in mine_set:
                    game = reconstruct_game(
                        mine_pos, rows, cols, revealed_pos, flagged_pos
                    )
                    game.flag(row, col)
                    safe_total = rows * cols - num_mines
                    if (
                        len(game.flagged) == num_mines
                        and len(game.revealed) >= safe_total
                    ):
                        scores.append(37.5 / 25.0)
                    else:
                        scores.append(15.0 / 25.0)
                else:
                    scores.append(-10.0 / 25.0)
            elif action_type == "reveal":
                if (row, col) in mine_set:
                    scores.append(-25.0 / 25.0)
                else:
                    is_deducible = ("reveal", row, col) in deducible_set
                    game = reconstruct_game(
                        mine_pos, rows, cols, revealed_pos, flagged_pos
                    )
                    game.reveal(row, col)
                    safe_total = rows * cols - num_mines
                    if (
                        len(game.revealed) >= safe_total
                        and len(game.flagged) == num_mines
                    ):
                        scores.append(37.5 / 25.0)
                    elif is_deducible:
                        scores.append(15.0 / 25.0)
                    else:
                        scores.append(10.0 / 25.0)
        except Exception:
            scores.append(0.0)
    return scores


def strategic_reward(completions, **kwargs):
    deducible_moves_list = kwargs.get("deducible_moves", [])
    mine_positions_list = kwargs.get("mine_positions", [])
    rows_list = kwargs.get("rows", [])
    cols_list = kwargs.get("cols", [])
    num_mines_list = kwargs.get("num_mines", [])
    flagged_positions_list = kwargs.get("flagged_positions", [])
    board_state_list = kwargs.get("board_state", [])

    scores = []
    for idx, completion in enumerate(completions):
        response = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )
        action = parse_llm_action(response)
        if action is None:
            scores.append(0.0)
            continue
        try:
            rows = int(rows_list[idx])
            cols = int(cols_list[idx])
            num_mines = int(num_mines_list[idx])
            deducible_raw = (
                json.loads(deducible_moves_list[idx])
                if isinstance(deducible_moves_list[idx], str)
                else deducible_moves_list[idx]
            )
            flagged_pos = (
                json.loads(flagged_positions_list[idx])
                if isinstance(flagged_positions_list[idx], str)
                else flagged_positions_list[idx]
            )
            board = (
                json.loads(board_state_list[idx])
                if isinstance(board_state_list[idx], str)
                else board_state_list[idx]
            )
            mine_pos = (
                json.loads(mine_positions_list[idx])
                if isinstance(mine_positions_list[idx], str)
                else mine_positions_list[idx]
            )
            mine_set = set(tuple(p) for p in mine_pos)
            flagged_set = set(tuple(p) for p in flagged_pos)
            row, col = action["row"], action["col"]
            action_type = action["type"]
            score = 0.0
            if not (0 <= row < rows and 0 <= col < cols):
                scores.append(0.0)
                continue
            has_deducible = len(deducible_raw) > 0
            action_is_deducible = any(
                m[0] == action_type and m[1] == row and m[2] == col
                for m in deducible_raw
            )
            if has_deducible and not action_is_deducible:
                score -= 0.3
            adjacent_to_number = any(
                0 <= row + dr < rows
                and 0 <= col + dc < cols
                and board[row + dr][col + dc] in "012345678"
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
            )
            if adjacent_to_number:
                score += 0.2
            if action_type == "flag" and (row, col) in mine_set and action_is_deducible:
                score += 0.15
            if action_type == "flag" and len(flagged_set) >= num_mines:
                score -= 0.4
            if action_type == "reveal" and (row, col) not in mine_set:
                adj_mine_count = sum(
                    1
                    for dr in [-1, 0, 1]
                    for dc in [-1, 0, 1]
                    if not (dr == 0 and dc == 0)
                    and 0 <= row + dr < rows
                    and 0 <= col + dc < cols
                    and (row + dr, col + dc) in mine_set
                )
                if adj_mine_count == 0:
                    score += 0.15
            scores.append(score)
        except Exception:
            scores.append(0.0)
    return scores


print("Reward functions loaded.")

# ================================================================
# Step 3: Load GRPO Dataset
# ================================================================
print("Loading training data...")
data_file = "/workspace/minesweeper_training_data.jsonl"
raw_data = []
skipped_50x50 = 0
with open(data_file, "r") as f:
    for line in f:
        ex = json.loads(line.strip())
        if ex.get("board_size") == "50x50":
            skipped_50x50 += 1
            continue
        raw_data.append(ex)
print(f"Loaded {len(raw_data)} examples (filtered {skipped_50x50} 50x50)")

grpo_items = []
skipped_long = 0
MAX_PROMPT_TOKENS = 4000

for ex in raw_data:
    prompt_msgs = json.loads(ex["prompt"])
    text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    token_len = len(tokenizer(text, add_special_tokens=False).input_ids)
    if token_len > MAX_PROMPT_TOKENS:
        skipped_long += 1
        continue
    item = {
        "prompt": prompt_msgs,
        "mine_positions": ex["mine_positions"],
        "rows": ex["rows"],
        "cols": ex["cols"],
        "num_mines": ex["num_mines"],
        "flagged_positions": ex["flagged_positions"],
        "revealed_positions": ex["revealed_positions"],
        "board_state": ex["board_state"],
        "deducible_moves": ex["deducible_moves"],
        "best_move": ex["best_move"],
        "is_deducible": ex["is_deducible"],
    }
    grpo_items.append(item)

grpo_dataset = Dataset.from_list(grpo_items)
print(
    f"GRPO dataset: {len(grpo_dataset)} examples (filtered {skipped_long} long prompts)"
)

# ================================================================
# Step 4: GRPO Training
# ================================================================
FastLanguageModel.for_training(model)


class CheckpointCallback(TrainerCallback):
    def __init__(self, save_steps_list):
        self.save_steps_list = set(save_steps_list)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in self.save_steps_list and model is not None:
            ckpt_dir = f"grpo_{state.global_step}"
            model.save_pretrained(ckpt_dir)
            print(f"\nSaved ablation checkpoint: {ckpt_dir}")


checkpoint_cb = CheckpointCallback(save_steps_list=[100, 200, 300])

grpo_config = GRPOConfig(
    output_dir="grpo_outputs",
    loss_type="dapo",
    epsilon=0.2,
    epsilon_high=0.28,
    beta=0.0,
    num_generations=4,
    max_prompt_length=4096,
    max_completion_length=128,
    temperature=1.0,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=400,
    optim="adamw_8bit",
    bf16=True,
    disable_dropout=True,
    scale_rewards="batch",
    reward_weights=[1.0, 2.0, 0.5],
    mask_truncated_completions=True,
    logging_steps=5,
    save_steps=200,
    save_total_limit=3,
    report_to="none",
    use_vllm=False,
)

print(
    f"\nGRPO config: {grpo_config.max_steps} steps, {grpo_config.num_generations} gens, no vLLM"
)

grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward, gameplay_reward, strategic_reward],
    args=grpo_config,
    train_dataset=grpo_dataset,
    callbacks=[checkpoint_cb],
)

print("Starting GRPO training...")
grpo_trainer.train()
print("GRPO training complete!")

# ================================================================
# Step 5: Eval + Save
# ================================================================
FastLanguageModel.for_inference(model)

FRONTIER_THRESHOLD = 16


def build_eval_prompt(board, rows, cols, mines, flags):
    mines_left = mines - flags
    if rows <= FRONTIER_THRESHOLD and cols <= FRONTIER_THRESHOLD:
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


print("\nPost-GRPO evaluation:")
board_configs = [
    (6, 6, 5, 10),
    (10, 10, 15, 10),
    (16, 16, 40, 5),
    (20, 20, 60, 5),
    (30, 30, 120, 3),
]

for rows, cols, mines, n_games in board_configs:
    valid_json = 0
    valid_moves = 0
    total_moves = 0
    wins = 0
    for seed in range(n_games):
        rng = random.Random(seed + 10000)
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
        for move_i in range(min(10, rows * cols)):
            if game.state != "ongoing":
                if game.state == "success":
                    wins += 1
                break
            board = game.get_board()
            flags = len(game.flagged)
            prompt = build_eval_prompt(board, rows, cols, mines, flags)
            sys_prompt = "You are an expert Minesweeper AI. Analyze constraints and output ONLY a valid JSON action. No explanation."
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
                            game.reveal(r_act, c_act)
                        elif action["type"] == "flag":
                            game.flag(r_act, c_act)
    json_rate = valid_json / max(total_moves, 1) * 100
    move_rate = valid_moves / max(total_moves, 1) * 100
    print(
        f"  {rows}x{cols}: JSON={json_rate:.0f}% ValidMove={move_rate:.0f}% Wins={wins}/{n_games} ({total_moves} moves)"
    )

# Save merged model
print("\nSaving merged model...")
output_path = "/workspace/your_finetuned_model"
model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")
print(f"Model saved to: {output_path}")

model_files = os.listdir(output_path)
total_size = sum(
    os.path.getsize(os.path.join(output_path, f))
    for f in model_files
    if os.path.isfile(os.path.join(output_path, f))
)
print(f"Files: {len(model_files)}, Total: {total_size / 1024**3:.1f} GB")
print("\nDone! Pipeline complete.")
