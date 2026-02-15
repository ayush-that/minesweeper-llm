#!/usr/bin/env python
"""
Targeted continued SFT: Add 50x50 + rectangular + small-frontier capability
WITHOUT changing the system prompt (key lesson from v2 failure).

Strategy:
1. 50x50 frontier from ORIGINAL data (same system prompt!)
2. Rectangular boards from v2 data (swap system prompt to original)
3. Small-board frontier from v2 data (swap system prompt to original)
4. Mix in original frontier reinforcement examples
5. Very conservative LR
"""

import os

os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface"

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
import torch
import json
import random
from collections import defaultdict
from datasets import Dataset

# ================================================================
# Config
# ================================================================
ORIGINAL_SYSTEM_PROMPT = "You are an expert Minesweeper AI. Analyze constraints and output ONLY a valid JSON action. No explanation."
max_seq_length = 8192
lora_rank = 64

# ================================================================
# Step 1: Load existing finetuned model
# ================================================================
model_name = "/workspace/your_finetuned_model"
print(f"Loading finetuned model from {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

# ================================================================
# Step 2: Add LoRA
# ================================================================
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
    lora_alpha=lora_rank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(
    f"Trainable: {trainable / 1e6:.1f}M / {total / 1e9:.1f}B ({trainable / total * 100:.2f}%)"
)

# ================================================================
# Step 3: Build targeted dataset
# ================================================================
rng = random.Random(42)

# Source 1: 50x50 frontier from original data (same system prompt!)
original_50x50 = []
original_frontier_other = []
with open("/workspace/minesweeper_training_data.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        msgs = (
            json.loads(ex["messages"])
            if isinstance(ex["messages"], str)
            else ex["messages"]
        )
        user_msg = msgs[1]["content"]
        if "FRONTIER" not in user_msg:
            continue
        if ex["board_size"] == "50x50":
            original_50x50.append(msgs)
        else:
            original_frontier_other.append(msgs)

print(f"Original 50x50 frontier: {len(original_50x50)}")
print(f"Original other frontier (20x20, 30x30): {len(original_frontier_other)}")

# Source 2: New board types from v2 data (swap system prompt)
v2_novel = defaultdict(list)
with open("/workspace/minesweeper_v2_data.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        msgs = (
            json.loads(ex["messages"])
            if isinstance(ex["messages"], str)
            else ex["messages"]
        )
        user_msg = msgs[1]["content"]
        if "FRONTIER" not in user_msg:
            continue
        size = ex["board_size"]
        rows, cols = size.split("x")
        # Novel: rectangular boards OR small boards that were compact in original
        if rows != cols or int(rows) <= 16:
            # Swap system prompt to original
            msgs[0]["content"] = ORIGINAL_SYSTEM_PROMPT
            v2_novel[size].append(msgs)

print("V2 novel boards (system prompt swapped):")
for size in sorted(
    v2_novel, key=lambda x: (int(x.split("x")[0]), int(x.split("x")[1]))
):
    print(f"  {size}: {len(v2_novel[size])}")

# Sample balanced dataset
focused = []

# 50x50 from original (already correct prompt)
n_50x50 = min(800, len(original_50x50))
focused.extend(rng.sample(original_50x50, n_50x50))
print(f"\nSampled 50x50 from original: {n_50x50}")

# Rectangular from v2 (prompt swapped)
rect_target = {
    "8x12": 200,
    "10x16": 200,
    "12x20": 200,
    "16x30": 150,
    "20x40": 100,
    "30x50": 100,
}
for size, target in rect_target.items():
    avail = v2_novel.get(size, [])
    n = min(target, len(avail))
    if n > 0:
        focused.extend(rng.sample(avail, n))
        print(f"Sampled {size} rectangular: {n}")

# Small board frontier from v2 (prompt swapped) - these were compact in original
small_target = {"6x6": 200, "8x8": 200, "10x10": 200, "16x16": 200}
for size, target in small_target.items():
    avail = v2_novel.get(size, [])
    n = min(target, len(avail))
    if n > 0:
        focused.extend(rng.sample(avail, n))
        print(f"Sampled {size} small frontier: {n}")

# Reinforcement: original frontier (20x20, 30x30) to prevent forgetting
n_reinforce = min(400, len(original_frontier_other))
focused.extend(rng.sample(original_frontier_other, n_reinforce))
print(f"Reinforcement from original (20x20, 30x30): {n_reinforce}")

rng.shuffle(focused)
print(f"\nTotal targeted dataset: {len(focused)} examples")

# Verify prompts
sys_prompts = set(msgs[0]["content"] for msgs in focused)
print(f"System prompts used: {sys_prompts}")

# ================================================================
# Step 4: Prepare SFT Dataset
# ================================================================
sft_items = [{"messages": msgs} for msgs in focused]
sft_dataset = Dataset.from_list(sft_items)
print(f"SFT dataset: {len(sft_dataset)} examples")

# ================================================================
# Step 5: Train (very conservative)
# ================================================================


def formatting_func(examples):
    messages = examples["messages"]
    if (
        isinstance(messages, list)
        and len(messages) > 0
        and isinstance(messages[0], dict)
    ):
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return [text]
    else:
        return [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in messages
        ]


sft_config = SFTConfig(
    output_dir="sft_targeted_checkpoint",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16
    learning_rate=2e-6,  # Very conservative (10x lower than original)
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    optim="adamw_8bit",
    bf16=True,
    logging_steps=10,
    save_steps=9999,  # Don't save intermediate (short run)
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

effective_batch = (
    sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps
)
est_steps = len(sft_dataset) // effective_batch
print("\nTargeted SFT config:")
print(f"  LR: {sft_config.learning_rate} (very conservative)")
print(f"  Batch: {effective_batch}")
print(f"  Est steps: ~{est_steps}")
print(f"  Est time: ~{est_steps * 10 / 60:.0f} min")

print("\nStarting targeted SFT...")
sft_trainer.train()
print("Targeted SFT complete!")

# ================================================================
# Step 6: Save & Merge
# ================================================================
model.save_pretrained("sft_targeted_checkpoint")
tokenizer.save_pretrained("sft_targeted_checkpoint")

print("\nMerging model...")
try:
    model.save_pretrained_merged(
        "/workspace/your_finetuned_model_v3",
        tokenizer,
        save_method="merged_16bit",
    )
    print("Merged via Unsloth!")
except Exception as e:
    print(f"Unsloth merge failed ({e}), using PEFT fallback...")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        "/workspace/your_finetuned_model",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, "sft_targeted_checkpoint")
    merged = merged.merge_and_unload()
    merged.save_pretrained("/workspace/your_finetuned_model_v3")
    tok = AutoTokenizer.from_pretrained("/workspace/your_finetuned_model")
    tok.save_pretrained("/workspace/your_finetuned_model_v3")
    print("Merged via PEFT fallback!")

model_files = os.listdir("/workspace/your_finetuned_model_v3")
total_size = sum(
    os.path.getsize(os.path.join("/workspace/your_finetuned_model_v3", f))
    for f in model_files
    if os.path.isfile(os.path.join("/workspace/your_finetuned_model_v3", f))
)
print(f"Saved {len(model_files)} files, total size: {total_size / 1024**3:.1f} GB")
print("\nDone! Model saved to /workspace/your_finetuned_model_v3")
