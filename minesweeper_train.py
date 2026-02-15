#!/usr/bin/env python
# coding: utf-8

# # Minesweeper LLM Competition - SFT + GRPO Training Pipeline
#
# ## Model: Qwen2.5-14B-Instruct
# ## Strategy: 3-Tier Solver -> 50K SFT Dataset -> GRPO Refinement
#
# **Pipeline:**
# 1. Load pre-generated training data (50K examples from forward-gameplay solver)
# 2. SFT warmup: 1 epoch on solver-labeled optimal moves
# 3. GRPO refinement: 1200 steps with 3 reward functions (format, gameplay, strategic)
# 4. Save merged model for evaluation

# # Step 1: Environment Setup & Model Loading

# In[ ]:


import os

os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"  # ROCm fix for Qwen2.5 SWA
os.environ["HF_HUB_OFFLINE"] = "1"  # Don't try to download, use local cache

from unsloth import FastLanguageModel
import torch

max_seq_length = (
    8192  # Handle large frontier format prompts (50x50 boards can reach ~6K tokens)
)
lora_rank = 64  # High rank for complex reasoning task

# Use local cache path directly (HF cache is read-only)
model_name = "/root/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
)

print(f"Model loaded: {model.config._name_or_path}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
print(f"Device: {model.device}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# # Step 2: Add LoRA Adapters

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank * 2,  # alpha = 2 * rank
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Print trainable parameter count
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(
    f"Trainable: {trainable / 1e6:.1f}M / {total / 1e9:.1f}B ({trainable / total * 100:.2f}%)"
)


# # Step 3: Load Training Data & Game Engine

# In[ ]:


import json
import re
import random
from datasets import Dataset
from collections import defaultdict

# ================================================================
# Game Engine (needed for reward functions)
# ================================================================


class MinesweeperGame:
    """Minesweeper game reconstructed from stored mine positions."""

    def __init__(self, rows, cols, mine_positions):
        self.rows = rows
        self.cols = cols
        self.mine_set = set(tuple(p) for p in mine_positions)
        self.num_mines = len(self.mine_set)

        # Calculate numbers
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
        """Reveal with flood fill. Returns 'mine', 'ok', or 'win'."""
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
        # Check win
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
    """Extract JSON action from LLM response. Returns last valid match."""
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
    """Reconstruct game state from stored data."""
    game = MinesweeperGame(rows, cols, mine_positions)
    # Re-reveal all cells (with flood fill)
    for r, c in revealed_positions:
        if (r, c) not in game.revealed:
            game.reveal(r, c)
    # Re-flag
    for r, c in flagged_positions:
        game.flag(r, c)
    # Reset state to ongoing (we're reconstructing mid-game)
    game._state = "ongoing"
    return game


print("Game engine loaded.")


# In[ ]:


# Verify external dependencies: solver.py and generate_data.py
import importlib.util
import os

for module_name, filepath in [
    ("solver", "/workspace/solver.py"),
    ("generate_data", "/workspace/generate_data.py"),
]:
    assert os.path.exists(filepath), f"MISSING: {filepath}"
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"  {module_name}: OK ({os.path.getsize(filepath) / 1024:.1f} KB)")

# Quick solver sanity check
from solver import solve_board

board = [["1", "1", "1"], ["1", ".", "1"], ["1", "1", "1"]]
solver = solve_board(board, 3, 3, 1, full=True)
moves = solver.get_certain_moves()
print(
    f"  Solver test: {len(moves)} certain moves on 3x3 with 1 mine -> {'PASS' if len(moves) == 1 else 'FAIL'}"
)
print("\nAll external files verified!")


# # Step 4: Load Pre-Generated Dataset

# In[ ]:


# Load pre-generated data from generate_data.py
# FILTER: Skip 50x50 boards — competition max is < 50x50
data_file = "minesweeper_training_data.jsonl"

raw_data = []
skipped_50x50 = 0
with open(data_file, "r") as f:
    for line in f:
        ex = json.loads(line.strip())
        if ex.get("board_size") == "50x50":
            skipped_50x50 += 1
            continue
        raw_data.append(ex)

print(f"Loaded {len(raw_data)} examples (filtered {skipped_50x50} x 50x50 boards)")

# Show distribution
stage_counts = defaultdict(int)
size_counts = defaultdict(int)
deducible_count = 0
for e in raw_data:
    stage_counts[e["game_stage"]] += 1
    size_counts[e["board_size"]] += 1
    if e["is_deducible"]:
        deducible_count += 1

print("\nBoard size distribution:")
for size in sorted(size_counts.keys(), key=lambda x: int(x.split("x")[0])):
    cnt = size_counts[size]
    print(f"  {size}: {cnt} ({cnt / len(raw_data) * 100:.1f}%)")

print("\nGame stage distribution:")
for stage in ["opening", "early", "mid", "late", "endgame", "near_failure"]:
    cnt = stage_counts.get(stage, 0)
    print(f"  {stage}: {cnt} ({cnt / len(raw_data) * 100:.1f}%)")

print(
    f"\nDeducible: {deducible_count}/{len(raw_data)} ({deducible_count / len(raw_data) * 100:.1f}%)"
)

# Show example
ex = raw_data[0]
msgs = json.loads(ex["messages"])
print("\nExample prompt (first 300 chars):")
print(msgs[1]["content"][:300])
print(f"\nExample response: {msgs[2]['content']}")


# # Step 5: Prepare SFT Dataset

# In[ ]:


# Prepare SFT dataset: parse messages from JSON strings to lists
sft_items = []
for ex in raw_data:
    messages = json.loads(ex["messages"])  # Parse JSON string -> list of dicts
    sft_items.append({"messages": messages})

sft_dataset = Dataset.from_list(sft_items)
print(f"SFT dataset: {len(sft_dataset)} examples")

# Verify format
assert isinstance(sft_dataset[0]["messages"], list), "Messages must be a list!"
assert isinstance(sft_dataset[0]["messages"][0], dict), "Each message must be a dict!"
assert "role" in sft_dataset[0]["messages"][0], "Messages must have 'role' key!"
print("Format check passed: messages is list of dicts with role/content")

# Verify tokenization works
test_text = tokenizer.apply_chat_template(
    sft_dataset[0]["messages"],
    tokenize=False,
    add_generation_prompt=False,
)
test_tokens = tokenizer(test_text, return_tensors="pt")
print(f"Example token length: {test_tokens.input_ids.shape[1]}")
print("First 200 chars of formatted text:")
print(test_text[:200])


# # Step 6: SFT Training (Phase 1)

# In[ ]:


from trl import SFTConfig, SFTTrainer


# Formatting function required by Unsloth's SFTTrainer
# Must always return a list of strings
def formatting_func(examples):
    """Apply chat template to convert messages to training text."""
    messages = examples["messages"]
    # Single example: messages is a list of dicts [{role:..., content:...}, ...]
    # Batch: messages is a list of lists [[{role:..., content:...}, ...], ...]
    if (
        isinstance(messages, list)
        and len(messages) > 0
        and isinstance(messages[0], dict)
    ):
        # Single example - wrap in list
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return [text]
    else:
        # Batch
        return [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in messages
        ]


sft_config = SFTConfig(
    output_dir="sft_checkpoint",
    per_device_train_batch_size=2,  # Reduced for 8192 seq length
    gradient_accumulation_steps=8,  # Effective batch = 16
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=1,  # 1 epoch to avoid memorization
    optim="adamw_8bit",
    bf16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    max_seq_length=max_seq_length,
    warmup_ratio=0.05,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset,
    args=sft_config,
    formatting_func=formatting_func,
)

print("SFT config:")
print(f"  Epochs: {sft_config.num_train_epochs}")
print(
    f"  Batch: {sft_config.per_device_train_batch_size} x {sft_config.gradient_accumulation_steps} = {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}"
)
print(f"  LR: {sft_config.learning_rate}")
print(f"  Max seq length: {sft_config.max_seq_length}")
print(
    f"  Steps: ~{len(sft_dataset) // (sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps)}"
)

print("\nStarting SFT training...")
sft_trainer.train()
print("SFT training complete!")


# # Step 7: Save SFT Checkpoint & Quick Eval

# In[ ]:


# Save SFT checkpoint
model.save_pretrained("sft_checkpoint")
tokenizer.save_pretrained("sft_checkpoint")
print("SFT checkpoint saved!")

# Quick evaluation after SFT
FastLanguageModel.for_inference(model)

FRONTIER_THRESHOLD = 16  # Must match generate_data.py and agents/minesweeper_agent.py


def build_eval_prompt(board, rows, cols, mines, flags):
    """Build eval prompt matching training data format exactly."""
    mines_left = mines - flags
    if rows <= FRONTIER_THRESHOLD and cols <= FRONTIER_THRESHOLD:
        grid = "\n".join("".join(r) for r in board)
        return f'MINESWEEPER {rows}x{cols} MINES:{mines} FLAGS:{flags} LEFT:{mines_left}\n{grid}\nRULES: .=hidden F=flag 0-8=adjacent mines\n- If number N has N flags around it, remaining hidden neighbors are SAFE->reveal\n- If number N needs (N-flags) more mines and has exactly that many hidden neighbors, all are MINES->flag\n- Flag certain mines FIRST, then reveal certain safe cells\n- NEVER act on already revealed or flagged cells\nOutput ONLY: {{"type":"reveal"|"flag","row":R,"col":C}}'
    else:
        # Frontier format - matches generate_data.py and agent exactly
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
        frontier_str = "\n".join(frontier_info[:200])  # Match training data: 200 cap
        hidden_near_str = "".join(
            f"({r},{c})" for r, c in sorted(all_hidden_near_numbers)[:100]
        )

        return f'MINESWEEPER {rows}x{cols} MINES:{mines} FLAGS:{flags} LEFT:{mines_left}\nFRONTIER (numbered cells with hidden neighbors):\n{frontier_str}\nHIDDEN NEAR NUMBERS: {hidden_near_str}\nTOTAL HIDDEN: {total_hidden} INTERIOR(no adj number): {interior_count}\nRULES: .=hidden F=flag 0-8=adjacent mines\n- If number N has N flags around it, remaining hidden neighbors are SAFE->reveal\n- If number N needs (N-flags) more mines and has exactly that many hidden neighbors, all are MINES->flag\n- Flag certain mines FIRST, then reveal certain safe cells\n- NEVER act on already revealed or flagged cells\nOutput ONLY: {{"type":"reveal"|"flag","row":R,"col":C}}'


def quick_eval(model, tokenizer, num_games=20, board_configs=None):
    """Quick evaluation across board sizes. Continues after invalid moves (like competition)."""
    if board_configs is None:
        board_configs = [
            (6, 6, 5, 5),
            (10, 10, 15, 5),
            (16, 16, 40, 5),
            (20, 20, 60, 5),
        ]

    results = {}
    for rows, cols, mines, n_games in board_configs:
        valid_json = 0
        valid_moves = 0
        invalid_moves = 0
        total_moves = 0
        wins = 0

        for seed in range(n_games):
            rng = random.Random(seed + 10000)
            positions = [(r, c) for r in range(rows) for c in range(cols)]
            mine_pos = rng.sample(positions, mines)
            game = MinesweeperGame(rows, cols, mine_pos)

            # Random first reveal
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
                        temperature=1.0,
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
                        else:
                            # Invalid target (already revealed/flagged) - count penalty but continue
                            invalid_moves += 1
                    else:
                        # Out of bounds - count penalty but continue
                        invalid_moves += 1
                else:
                    # Invalid JSON - count penalty but continue
                    invalid_moves += 1

        json_rate = valid_json / max(total_moves, 1) * 100
        move_rate = valid_moves / max(total_moves, 1) * 100
        results[f"{rows}x{cols}"] = (json_rate, move_rate, total_moves, wins)
        print(
            f"  {rows}x{cols}: JSON={json_rate:.0f}% ValidMove={move_rate:.0f}% Wins={wins}/{n_games} Invalid={invalid_moves} ({total_moves} moves)"
        )

    return results


print("Post-SFT evaluation:")
sft_results = quick_eval(model, tokenizer)


# # Step 8: GRPO Reward Functions

# In[ ]:


# ================================================================
# Reward Function 1: Format Reward (weight: 1.0)
# ================================================================


def format_reward(completions, **kwargs):
    """Reward valid JSON action format.
    Valid JSON with correct keys -> +1.0
    Invalid -> -3.0
    """
    scores = []
    for completion in completions:
        response = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )
        action = parse_llm_action(response)
        if action is not None:
            scores.append(1.0)
        else:
            scores.append(-3.0)
    return scores


# ================================================================
# Reward Function 2: Gameplay Reward (weight: 2.0)
# ================================================================


def gameplay_reward(completions, **kwargs):
    """Score gameplay quality by reconstructing game and simulating the action.

    Uses reconstruct_game() + actual reveal/flag for accurate win detection
    including 0-cell flood-fill cascades.

    Win requires BOTH: all mines flagged AND all safe cells revealed.
    This matches the competition specification exactly.

    Scoring (raw, normalized by /25):
    - Out of bounds:        -15 -> -0.60
    - Already revealed:     -12 -> -0.48
    - Already flagged:       -8 -> -0.32
    - Flag non-mine:        -10 -> -0.40
    - Total flags > mines:  -10 -> -0.40
    - Reveal mine:          -25 -> -1.00
    - Flag correct mine:    +15 -> +0.60
    - Reveal safe (random): +10 -> +0.40
    - Reveal safe (deducible): +15 -> +0.60
    - Win game:            +37.5 -> +1.50 (capped)
    """
    mine_positions_list = kwargs.get("mine_positions", [])
    rows_list = kwargs.get("rows", [])
    cols_list = kwargs.get("cols", [])
    num_mines_list = kwargs.get("num_mines", [])
    flagged_positions_list = kwargs.get("flagged_positions", [])
    __revealed_positions_list = kwargs.get("revealed_positions", [])
    deducible_moves_list = kwargs.get("deducible_moves", [])

    scores = []
    for idx, completion in enumerate(completions):
        response = (
            completion[0]["content"] if isinstance(completion, list) else completion
        )
        action = parse_llm_action(response)

        if action is None:
            scores.append(-10.0 / 25.0)  # Invalid format penalty
            continue

        try:
            # Get stored game data
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
                json.loads(__revealed_positions_list[idx])
                if isinstance(__revealed_positions_list[idx], str)
                else __revealed_positions_list[idx]
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

            # Out of bounds
            if not (0 <= row < rows and 0 <= col < cols):
                scores.append(-15.0 / 25.0)
                continue

            # Already revealed
            if (row, col) in revealed_set:
                scores.append(-12.0 / 25.0)
                continue

            # Already flagged
            if (row, col) in flagged_set:
                scores.append(-8.0 / 25.0)
                continue

            if action_type == "flag":
                # Total flags > total mines check
                if len(flagged_set) >= num_mines:
                    scores.append(-10.0 / 25.0)
                    continue

                if (row, col) in mine_set:
                    # Flag correct mine - simulate to check win
                    game = reconstruct_game(
                        mine_pos, rows, cols, revealed_pos, flagged_pos
                    )
                    game.flag(row, col)
                    # Win: all mines flagged AND all safe revealed
                    safe_total = rows * cols - num_mines
                    if (
                        len(game.flagged) == num_mines
                        and len(game.revealed) >= safe_total
                    ):
                        scores.append(37.5 / 25.0)  # Capped win reward
                    else:
                        scores.append(15.0 / 25.0)
                else:
                    scores.append(-10.0 / 25.0)  # Flag non-mine

            elif action_type == "reveal":
                if (row, col) in mine_set:
                    scores.append(-25.0 / 25.0)  # Hit mine
                else:
                    is_deducible = ("reveal", row, col) in deducible_set

                    # Simulate the reveal with flood fill to detect cascade wins
                    game = reconstruct_game(
                        mine_pos, rows, cols, revealed_pos, flagged_pos
                    )
                    game.reveal(row, col)

                    # Win: all safe revealed (checked by game.reveal) AND all mines flagged
                    safe_total = rows * cols - num_mines
                    if (
                        len(game.revealed) >= safe_total
                        and len(game.flagged) == num_mines
                    ):
                        scores.append(37.5 / 25.0)  # Capped win reward
                    elif is_deducible:
                        scores.append(15.0 / 25.0)
                    else:
                        scores.append(10.0 / 25.0)

        except Exception:
            scores.append(0.0)

    return scores


# ================================================================
# Reward Function 3: Strategic Reward (weight: 0.5)
# ================================================================


def strategic_reward(completions, **kwargs):
    """Reward strategic play quality.

    - Guessed when deducible move existed: -0.3
    - Move adjacent to revealed numbers:  +0.2
    - Flagged certain mine with safe reveals available: +0.15
    - Over-flagged (flags >= mines and chose flag): -0.4
    - Reveal triggers 0-cell cascade: +0.15
    """
    deducible_moves_list = kwargs.get("deducible_moves", [])
    __is_deducible_list = kwargs.get("is_deducible", [])
    mine_positions_list = kwargs.get("mine_positions", [])
    rows_list = kwargs.get("rows", [])
    cols_list = kwargs.get("cols", [])
    num_mines_list = kwargs.get("num_mines", [])
    flagged_positions_list = kwargs.get("flagged_positions", [])
    __revealed_positions_list = kwargs.get("revealed_positions", [])
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

            # Check bounds
            if not (0 <= row < rows and 0 <= col < cols):
                scores.append(0.0)
                continue

            # Check if there were deducible moves
            has_deducible = len(deducible_raw) > 0
            action_is_deducible = False
            for m in deducible_raw:
                if m[0] == action_type and m[1] == row and m[2] == col:
                    action_is_deducible = True
                    break

            # Penalty: guessed when deducible move existed
            if has_deducible and not action_is_deducible:
                score -= 0.3

            # Reward: move adjacent to revealed numbers (info-gathering)
            adjacent_to_number = False
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if board[nr][nc] in "012345678":
                            adjacent_to_number = True
                            break
                if adjacent_to_number:
                    break
            if adjacent_to_number:
                score += 0.2

            # Reward: flagged certain mine (flag-first strategy)
            if action_type == "flag" and (row, col) in mine_set and action_is_deducible:
                score += 0.15

            # Penalty: over-flagging
            if action_type == "flag" and len(flagged_set) >= num_mines:
                score -= 0.4

            # Reward: reveal that triggers 0-cell cascade
            if action_type == "reveal" and (row, col) not in mine_set:
                adj_mine_count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in mine_set:
                            adj_mine_count += 1
                if adj_mine_count == 0:
                    score += 0.15  # Cell is 0 -> triggers flood-fill cascade

            scores.append(score)

        except Exception:
            scores.append(0.0)

    return scores


print("Reward functions defined:")
print("  1. format_reward (weight=1.0): JSON validity")
print(
    "  2. gameplay_reward (weight=2.0): Game rules + flood-fill win detection (requires all flags)"
)
print("  3. strategic_reward (weight=0.5): Strategic play + cascade detection")


# # Step 9: Prepare GRPO Dataset & Configure Training

# In[ ]:


# Prepare GRPO dataset with all metadata columns
# IMPORTANT: "prompt" must be a list of dicts (not JSON string) for TRL
grpo_items = []
skipped_long = 0
MAX_PROMPT_TOKENS = (
    7500  # Leave headroom: 8192 - 128 (completion) - 564 (padding/overhead)
)

for ex in raw_data:
    prompt_msgs = json.loads(ex["prompt"])  # Parse JSON string -> list of dicts

    # Filter out prompts that would be truncated by TRL's max_prompt_length
    # This prevents silent reward poisoning where model sees truncated board
    # but reward function scores against full game state
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
    f"GRPO dataset: {len(grpo_dataset)} examples (filtered {skipped_long} long prompts > {MAX_PROMPT_TOKENS} tokens)"
)
print(f"Columns: {grpo_dataset.column_names}")

# Verify prompt format
assert isinstance(grpo_dataset[0]["prompt"], list), "Prompt must be a list!"
assert isinstance(grpo_dataset[0]["prompt"][0], dict), "Each prompt msg must be a dict!"
print("Prompt format check passed")

# Show token length distribution of remaining data
if len(grpo_items) > 0:
    sample_lens = []
    for item in grpo_items[:1000]:
        text = tokenizer.apply_chat_template(
            item["prompt"], tokenize=False, add_generation_prompt=True
        )
        sample_lens.append(len(tokenizer(text, add_special_tokens=False).input_ids))
    print(
        f"Token length stats (sample of {len(sample_lens)}): min={min(sample_lens)} max={max(sample_lens)} mean={sum(sample_lens) / len(sample_lens):.0f}"
    )


# # Step 10: GRPO Training

# In[ ]:


# Switch model back to training mode
FastLanguageModel.for_training(model)


# In[ ]:


from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback


class CheckpointCallback(TrainerCallback):
    """Save ablation checkpoints at specific steps."""

    def __init__(self, save_steps_list):
        self.save_steps_list = set(save_steps_list)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in self.save_steps_list and model is not None:
            ckpt_dir = f"grpo_{state.global_step}"
            model.save_pretrained(ckpt_dir)
            print(f"\nSaved ablation checkpoint: {ckpt_dir}")


checkpoint_cb = CheckpointCallback(save_steps_list=[400, 800])

grpo_config = GRPOConfig(
    output_dir="grpo_outputs",
    # DAPO loss with asymmetric clipping
    loss_type="dapo",
    epsilon=0.2,
    epsilon_high=0.28,
    beta=0.0,  # No KL, no ref model
    # Generation
    num_generations=8,
    max_prompt_length=7500,  # Matches token filter in cell 18 - no silent truncation
    max_completion_length=128,
    temperature=1.0,
    # Training
    per_device_train_batch_size=1,  # Reduced for 8K prompts + 8 generations
    gradient_accumulation_steps=8,  # Effective batch = 8
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=1200,
    # Optimization
    optim="adamw_8bit",
    bf16=True,
    # Reward
    scale_rewards="batch",
    reward_weights=[1.0, 2.0, 0.5],
    # Generation masking
    mask_truncated_completions=True,
    # Logging
    logging_steps=5,
    save_steps=400,
    save_total_limit=3,
    report_to="none",
    # vLLM for faster generation
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.3,
)

print("GRPO config:")
print(
    f"  Loss: {grpo_config.loss_type}, epsilon={grpo_config.epsilon}/{grpo_config.epsilon_high}"
)
print(f"  Generations: {grpo_config.num_generations}")
print(
    f"  Batch: {grpo_config.per_device_train_batch_size} x {grpo_config.gradient_accumulation_steps}"
)
print(f"  Steps: {grpo_config.max_steps}")
print(f"  LR: {grpo_config.learning_rate}")
print(f"  Reward weights: {grpo_config.reward_weights}")
print(f"  vLLM: {grpo_config.use_vllm} mode={grpo_config.vllm_mode}")
print(f"  Max prompt length: {grpo_config.max_prompt_length}")
print(f"  Max completion length: {grpo_config.max_completion_length}")

# Create trainer
grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward, gameplay_reward, strategic_reward],
    args=grpo_config,
    train_dataset=grpo_dataset,
    callbacks=[checkpoint_cb],
)

print("\nStarting GRPO training...")
try:
    grpo_trainer.train()
    print("GRPO training complete!")
except torch.cuda.OutOfMemoryError:
    print("\nOOM with vLLM colocate! Falling back to num_generations=4...")
    torch.cuda.empty_cache()
    # Rebuild with reduced generations
    grpo_config_fallback = GRPOConfig(
        output_dir="grpo_outputs",
        loss_type="dapo",
        epsilon=0.2,
        epsilon_high=0.28,
        beta=0.0,
        num_generations=4,  # Reduced from 8
        max_prompt_length=7500,
        max_completion_length=128,
        temperature=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_steps=1200,
        optim="adamw_8bit",
        bf16=True,
        scale_rewards="batch",
        reward_weights=[1.0, 2.0, 0.5],
        mask_truncated_completions=True,
        logging_steps=5,
        save_steps=400,
        save_total_limit=3,
        report_to="none",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.2,
    )
    FastLanguageModel.for_training(model)
    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, gameplay_reward, strategic_reward],
        args=grpo_config_fallback,
        train_dataset=grpo_dataset,
        callbacks=[checkpoint_cb],
    )
    grpo_trainer.train()
    print("GRPO training complete (with fallback config)!")


# In[ ]:


# Evaluate GRPO model
FastLanguageModel.for_inference(model)

print("Post-GRPO evaluation:")
grpo_results = quick_eval(
    model,
    tokenizer,
    board_configs=[
        (6, 6, 5, 10),
        (10, 10, 15, 10),
        (16, 16, 40, 5),
        (20, 20, 60, 5),
        (30, 30, 120, 3),
    ],
)

# Compare with SFT results
print("\nComparison (JSON% / ValidMove% / Wins):")
for size in sorted(
    set(list(sft_results.keys()) + list(grpo_results.keys())),
    key=lambda x: int(x.split("x")[0]),
):
    sft_r = sft_results.get(size, (0, 0, 0, 0))
    grpo_r = grpo_results.get(size, (0, 0, 0, 0))
    print(
        f"  {size}: SFT={sft_r[0]:.0f}%/{sft_r[1]:.0f}%/W{sft_r[3]}  GRPO={grpo_r[0]:.0f}%/{grpo_r[1]:.0f}%/W{grpo_r[3]}"
    )


# # Step 12: Save Final Merged Model

# In[ ]:


# Save merged model in 16-bit for evaluation
output_path = "/workspace/your_finetuned_model"

model.save_pretrained_merged(
    output_path,
    tokenizer,
    save_method="merged_16bit",
)

print(f"Model saved to: {output_path}")
print("This path is referenced in agents/minesweeper_model.py")

# Verify the saved model
import os

model_files = os.listdir(output_path)
print(f"\nFiles: {model_files}")
total_size = sum(
    os.path.getsize(os.path.join(output_path, f))
    for f in model_files
    if os.path.isfile(os.path.join(output_path, f))
)
print(f"Total size: {total_size / 1024**3:.1f} GB")


# In[ ]:


# Comprehensive final evaluation with full game playouts
# Use higher move limit for meaningful win rate data (unlike quick_eval's 10-move sanity check)
print("=" * 60)
print("FINAL COMPREHENSIVE EVALUATION")
print("=" * 60)


def full_eval(model, tokenizer, board_configs, max_moves_per_game=500):
    """Full evaluation with high move limit for actual win rate measurement.

    Includes:
    - Infinite loop detection (break after 3 identical consecutive actions)
    - Accurate mine_hit tracking (not counted as valid_moves)
    - Continues after invalid moves (like competition)
    """
    results = {}
    for rows, cols, mines, n_games in board_configs:
        valid_json = 0
        valid_moves = 0
        invalid_moves = 0
        total_moves = 0
        wins = 0
        mine_hits = 0
        loops_broken = 0

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

            last_action = None
            repeat_count = 0

            for move_i in range(max_moves_per_game):
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
                        temperature=1.0,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                response = tokenizer.decode(
                    output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
                )
                action = parse_llm_action(response)

                # Infinite loop detection: if same action 3 times in a row, break
                if action is not None:
                    action_tuple = (action["type"], action["row"], action["col"])
                    if action_tuple == last_action:
                        repeat_count += 1
                        if repeat_count >= 3:
                            loops_broken += 1
                            break
                    else:
                        repeat_count = 1
                        last_action = action_tuple

                total_moves += 1
                if action is not None:
                    valid_json += 1
                    r_act, c_act = action["row"], action["col"]
                    if 0 <= r_act < rows and 0 <= c_act < cols:
                        cell_val = board[r_act][c_act]
                        if cell_val == ".":
                            if action["type"] == "reveal":
                                result = game.reveal(r_act, c_act)
                                if result == "mine":
                                    mine_hits += 1
                                    # Mine hit is NOT a "valid move" for metrics
                                else:
                                    valid_moves += 1
                            elif action["type"] == "flag":
                                game.flag(r_act, c_act)
                                valid_moves += 1
                        else:
                            invalid_moves += 1
                    else:
                        invalid_moves += 1
                else:
                    invalid_moves += 1

        json_rate = valid_json / max(total_moves, 1) * 100
        move_rate = valid_moves / max(total_moves, 1) * 100
        results[f"{rows}x{cols}"] = (
            json_rate,
            move_rate,
            total_moves,
            wins,
            mine_hits,
            n_games,
        )
        loop_str = f" Loops={loops_broken}" if loops_broken > 0 else ""
        print(
            f"  {rows}x{cols}: JSON={json_rate:.0f}% ValidMove={move_rate:.0f}% Wins={wins}/{n_games} MineHits={mine_hits} Invalid={invalid_moves}{loop_str} ({total_moves} moves)"
        )

    return results


final_results = full_eval(
    model,
    tokenizer,
    board_configs=[
        (6, 6, 5, 20),
        (8, 8, 10, 20),
        (10, 10, 15, 20),
        (16, 16, 40, 10),
        (20, 20, 60, 10),
        (30, 30, 120, 5),
    ],
)

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
for size, (json_r, move_r, total, wins, mine_hits, n_games) in sorted(
    final_results.items(), key=lambda x: int(x[0].split("x")[0])
):
    status = "PASS" if json_r >= 90 and move_r >= 50 else "CHECK"
    print(
        f"  [{status}] {size}: JSON={json_r:.0f}% ValidMove={move_r:.0f}% Wins={wins}/{n_games} MineHits={mine_hits} ({total} moves)"
    )

print("\nTraining pipeline complete! Model saved to /workspace/your_finetuned_model")


# # Done! Summary of what was trained:
# #
# # Model: Qwen2.5-14B-Instruct with LoRA (rank=64, alpha=128)
# # Phase 1: SFT on 50K solver-generated examples (1 epoch)
# # Phase 2: GRPO with 3 reward functions (1200 steps, DAPO loss)
# #
# # Reward functions:
# #   1. format_reward: Valid JSON output (+1.0 / -3.0)
# #   2. gameplay_reward: Game rules scoring (normalized /25)
# #   3. strategic_reward: Strategic play quality (deducible moves, flag-first)
# #
# # Output: /workspace/your_finetuned_model (merged 16-bit)
# # Agent: /workspace/agents/ (minesweeper_model.py points to model)
# #
# # Key decisions:
# # - Compact grid prompt for <=16x16, frontier sparse for >16x16
# # - Greedy decoding at eval (temperature=0, do_sample=false)
# # - Flag-first strategy (flag certain mines before revealing safe cells)
# # - Win reward capped at +1.5 normalized to prevent gradient spikes
