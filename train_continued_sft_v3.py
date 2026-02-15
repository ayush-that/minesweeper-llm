#!/usr/bin/env python
"""
Continued SFT v3: Train on v2 model with 10K examples
- More data coverage than v2's 5K
- Boost 6x6 (was weak in v2) with more examples
- 2 epochs for better convergence
- Uses v2 as base model
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
# Step 1: Load v2 finetuned model
# ================================================================

max_seq_length = 8192
lora_rank = 64

model_name = "/workspace/your_finetuned_model_v2"

print(f"Loading v2 model from {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=False,
    max_seq_length=max_seq_length,
    torch_dtype=torch.bfloat16,
)

print(f"Model loaded: {model.config._name_or_path}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

# ================================================================
# Step 2: Add LoRA Adapters
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
# Step 3: Create 10K dataset - boost small boards
# ================================================================

data_file = "minesweeper_v2_data.jsonl"

all_data = []
with open(data_file, "r") as f:
    for line in f:
        ex = json.loads(line.strip())
        all_data.append(ex)

print(f"Full dataset: {len(all_data)} examples")

by_size = defaultdict(list)
for ex in all_data:
    by_size[ex["board_size"]].append(ex)

# Use a different seed than v2 training (42) to get different examples
rng = random.Random(123)
focused = []

# Boost small boards (6x6 was weak in v2)
target_per_size = {
    "6x6": 1000,  # Doubled! Was weak (-1.9) in v2
    "8x8": 800,  # Boost
    "8x12": 500,  # Rectangular
    "10x10": 700,  # Boost
    "10x16": 500,  # Rectangular
    "12x20": 500,  # Rectangular
    "16x16": 700,  # Strong, reinforce
    "16x30": 500,  # Rectangular
    "20x20": 700,  # Reinforce
    "20x40": 500,  # Rectangular
    "30x30": 700,  # Reinforce
    "30x50": 500,  # Rectangular
    "50x50": 1000,  # Large boards
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

deducible_count = sum(1 for e in focused if e["is_deducible"])
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

msgs = sft_dataset[0]["messages"]
print(f"System: {msgs[0]['content'][:80]}")
print(f"Assistant: {msgs[2]['content']}")

# ================================================================
# Step 5: SFT Training
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
    output_dir="sft_v3_checkpoint",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16
    learning_rate=3e-6,  # Even lower LR for v3 (was 5e-6 for v2)
    lr_scheduler_type="cosine",
    num_train_epochs=2,  # 2 epochs for better convergence
    optim="adamw_8bit",
    bf16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    max_seq_length=max_seq_length,
    warmup_ratio=0.03,
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
est_steps = len(sft_dataset) * sft_config.num_train_epochs // effective_batch
print("\nv3 SFT config:")
print("  Starting from: v2 finetuned model")
print(f"  Epochs: {sft_config.num_train_epochs}")
print(
    f"  Batch: {sft_config.per_device_train_batch_size} x {sft_config.gradient_accumulation_steps} = {effective_batch}"
)
print(f"  LR: {sft_config.learning_rate}")
print(f"  Estimated steps: ~{est_steps}")

print("\n" + "=" * 60)
print("Starting v3 SFT training...")
print("=" * 60)
sft_trainer.train()
print("v3 SFT training complete!")

# ================================================================
# Step 6: Save & Merge
# ================================================================

model.save_pretrained("sft_v3_checkpoint")
tokenizer.save_pretrained("sft_v3_checkpoint")
print("Checkpoint saved!")

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
        "/workspace/your_finetuned_model_v2",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, "sft_v3_checkpoint")
    merged = merged.merge_and_unload()
    merged.save_pretrained("/workspace/your_finetuned_model_v3")
    tok = AutoTokenizer.from_pretrained("/workspace/your_finetuned_model_v2")
    tok.save_pretrained("/workspace/your_finetuned_model_v3")
    print("Merged via PEFT fallback!")

# Verify
model_files = os.listdir("/workspace/your_finetuned_model_v3")
total_size = sum(
    os.path.getsize(os.path.join("/workspace/your_finetuned_model_v3", f))
    for f in model_files
    if os.path.isfile(os.path.join("/workspace/your_finetuned_model_v3", f))
)
print(f"Saved {len(model_files)} files, total size: {total_size / 1024**3:.1f} GB")
print("\nDone! Model saved to /workspace/your_finetuned_model_v3")
