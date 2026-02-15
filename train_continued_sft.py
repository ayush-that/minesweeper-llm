#!/usr/bin/env python
"""
Continued SFT on existing finetuned model with corrected data:
- ALL frontier format (fixes 51.4% training/inference mismatch)
- Includes 50x50 and rectangular boards
- Focused 5K dataset for fast training (~50 min)
- Lower LR since model already knows Minesweeper
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
# Step 1: Load EXISTING finetuned model (not base!)
# ================================================================

max_seq_length = 8192
lora_rank = 64

# Use existing finetuned model as starting point
model_name = "/workspace/your_finetuned_model"

print(f"Loading finetuned model from {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
)

print(f"Model loaded: {model.config._name_or_path}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ================================================================
# Step 2: Add LoRA Adapters (fresh adapters on top of finetuned)
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
# Step 3: Create focused dataset from full v2 data
# ================================================================

data_file = "minesweeper_v2_data.jsonl"

all_data = []
with open(data_file, "r") as f:
    for line in f:
        ex = json.loads(line.strip())
        all_data.append(ex)

print(f"Full dataset: {len(all_data)} examples")

# Group by board size
by_size = defaultdict(list)
for ex in all_data:
    by_size[ex["board_size"]].append(ex)

print(f"Board sizes: {sorted(by_size.keys())}")

# Sample balanced focused dataset (~5K examples)
# Prioritize sizes that were trained wrong (small boards were compact format)
# and new sizes (50x50, rectangular)
rng = random.Random(42)
focused = []

target_per_size = {
    "6x6": 500,  # Critical: was trained on compact, now frontier
    "8x8": 500,  # Critical: was trained on compact, now frontier
    "8x12": 300,  # NEW: rectangular
    "10x10": 400,  # Critical: was trained on compact
    "10x16": 300,  # NEW: rectangular
    "12x20": 300,  # NEW: rectangular
    "16x16": 400,  # Critical: was trained on compact
    "16x30": 300,  # NEW: rectangular
    "20x20": 400,  # Reinforce with new prompt
    "20x40": 300,  # NEW: rectangular
    "30x30": 400,  # Reinforce with new prompt
    "30x50": 300,  # NEW: rectangular
    "50x50": 600,  # NEW: was filtered out before
}

for size, target in target_per_size.items():
    available = by_size.get(size, [])
    if not available:
        print(f"  WARNING: No examples for {size}")
        continue
    n = min(target, len(available))
    sampled = rng.sample(available, n)
    focused.extend(sampled)
    print(f"  {size}: {n}/{len(available)} examples")

rng.shuffle(focused)
print(f"\nFocused dataset: {len(focused)} examples")

# Show stats
size_counts = defaultdict(int)
deducible_count = 0
for e in focused:
    size_counts[e["board_size"]] += 1
    if e["is_deducible"]:
        deducible_count += 1

print(
    f"Deducible: {deducible_count}/{len(focused)} ({deducible_count / len(focused) * 100:.1f}%)"
)

# ================================================================
# Step 4: Prepare SFT Dataset
# ================================================================

sft_items = []
for ex in focused:
    messages = json.loads(ex["messages"])
    sft_items.append({"messages": messages})

sft_dataset = Dataset.from_list(sft_items)
print(f"\nSFT dataset: {len(sft_dataset)} examples")

# Verify format
msgs = sft_dataset[0]["messages"]
print(f"System: {msgs[0]['content'][:80]}")
print(f"Assistant: {msgs[2]['content']}")

# ================================================================
# Step 5: SFT Training (continued, lower LR)
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
    output_dir="sft_continued_checkpoint",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16
    learning_rate=5e-6,  # Lower LR for continued SFT
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    optim="adamw_8bit",
    bf16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
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
print("\nContinued SFT config:")
print("  Starting from: finetuned model (already knows Minesweeper)")
print(f"  Epochs: {sft_config.num_train_epochs}")
print(
    f"  Batch: {sft_config.per_device_train_batch_size} x {sft_config.gradient_accumulation_steps} = {effective_batch}"
)
print(f"  LR: {sft_config.learning_rate} (lower for continued SFT)")
print(f"  Estimated steps: ~{est_steps}")

print("\n" + "=" * 60)
print("Starting continued SFT training...")
print("=" * 60)
sft_trainer.train()
print("Continued SFT training complete!")

# ================================================================
# Step 6: Save & Merge
# ================================================================

model.save_pretrained("sft_continued_checkpoint")
tokenizer.save_pretrained("sft_continued_checkpoint")
print("Checkpoint saved!")

print("\nMerging model...")
try:
    model.save_pretrained_merged(
        "/workspace/your_finetuned_model_v2",
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
    merged = PeftModel.from_pretrained(base, "sft_continued_checkpoint")
    merged = merged.merge_and_unload()
    merged.save_pretrained("/workspace/your_finetuned_model_v2")
    tok = AutoTokenizer.from_pretrained("/workspace/your_finetuned_model")
    tok.save_pretrained("/workspace/your_finetuned_model_v2")
    print("Merged via PEFT fallback!")

# Verify
model_files = os.listdir("/workspace/your_finetuned_model_v2")
total_size = sum(
    os.path.getsize(os.path.join("/workspace/your_finetuned_model_v2", f))
    for f in model_files
    if os.path.isfile(os.path.join("/workspace/your_finetuned_model_v2", f))
)
print(f"Saved {len(model_files)} files, total size: {total_size / 1024**3:.1f} GB")
print("\nDone! Model saved to /workspace/your_finetuned_model_v2")
