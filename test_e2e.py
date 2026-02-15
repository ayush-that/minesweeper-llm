#!/usr/bin/env python3
"""Quick E2E test: load model via agent code path, generate a single action."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

import sys

sys.path.insert(0, "/workspace")

import json
from agents.minesweeper_agent import MinesweeperPlayer

# Test game state (10x10 board, mid-game)
game_state = {
    "board": [
        ["1", "1", "0", "0", "0", "0", "1", ".", ".", "."],
        [".", "2", "1", "0", "0", "0", "1", "2", ".", "."],
        [".", ".", "1", "0", "0", "0", "0", "1", ".", "."],
        [".", ".", "1", "0", "0", "0", "1", "2", ".", "."],
        [".", ".", "2", "1", "1", "0", "1", ".", ".", "."],
        [".", ".", ".", ".", "1", "0", "1", ".", ".", "."],
        [".", ".", ".", ".", "1", "0", "1", "1", ".", "."],
        [".", ".", ".", ".", "1", "0", "0", "1", ".", "."],
        [".", ".", ".", ".", "1", "1", "1", "1", ".", "."],
        [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    ],
    "rows": 10,
    "cols": 10,
    "mines": 15,
    "flags_placed": 0,
    "_round": 1,
    "_sequence": 0,
}

print("Initializing player...")
player = MinesweeperPlayer()
print("Player initialized!")

# Build prompt
prompt, sys_prompt = player.build_prompt(game_state)
print(f"\nPrompt type: {'frontier' if 'FRONTIER' in prompt else 'compact'}")
print(f"Prompt length: {len(prompt)} chars")
print(f"Prompt preview:\n{prompt[:500]}...")

# Generate action
print("\nGenerating action...")
action, tl, gt = player.play_action(game_state, max_new_tokens=64, do_sample=False)

if action:
    print(f"\nAction: {json.dumps(action)}")
    board = game_state["board"]
    r, c = action["row"], action["col"]
    if 0 <= r < 10 and 0 <= c < 10:
        cell = board[r][c]
        print(
            f"Target cell [{r},{c}] = '{cell}' ({'hidden/valid' if cell == '.' else 'INVALID - already revealed/flagged'})"
        )
    else:
        print(f"INVALID: out of bounds [{r},{c}]")
else:
    print("ERROR: No valid action generated!")

print("\nE2E test complete!")
