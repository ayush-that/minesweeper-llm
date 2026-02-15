#!/usr/bin/env python
"""
Minesweeper SFT Training v2 - Streamlined
Key fixes:
1. All frontier format (no compact) - matches inference
2. Includes 50x50 boards
3. Concise JSON-only output
4. Loads base model from /workspace/qwen_base/
"""

import os

os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"
os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface"

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
import torch
import json
from collections import defaultdict
from datasets import Dataset

# ================================================================
# Step 1: Load Model
# ================================================================

max_seq_length = 8192
lora_rank = 64

model_name = "/workspace/qwen_base"

print(f"Loading model from {model_name}...")
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
# Step 3: Load Training Data (NO 50x50 filter!)
# ================================================================

data_file = "minesweeper_v2_data.jsonl"

raw_data = []
with open(data_file, "r") as f:
    for line in f:
        ex = json.loads(line.strip())
        raw_data.append(ex)

print(f"Loaded {len(raw_data)} examples (no filtering)")

# Show distribution
size_counts = defaultdict(int)
stage_counts = defaultdict(int)
deducible_count = 0
for e in raw_data:
    size_counts[e["board_size"]] += 1
    stage_counts[e["game_stage"]] += 1
    if e["is_deducible"]:
        deducible_count += 1

print("\nBoard size distribution:")
for size in sorted(
    size_counts.keys(), key=lambda x: (int(x.split("x")[0]), int(x.split("x")[1]))
):
    cnt = size_counts[size]
    print(f"  {size}: {cnt} ({cnt / len(raw_data) * 100:.1f}%)")

print("\nStage distribution:")
for stage in ["opening", "early", "mid", "late", "endgame", "near_failure"]:
    cnt = stage_counts.get(stage, 0)
    print(f"  {stage}: {cnt} ({cnt / len(raw_data) * 100:.1f}%)")

print(
    f"\nDeducible: {deducible_count}/{len(raw_data)} ({deducible_count / len(raw_data) * 100:.1f}%)"
)

# ================================================================
# Step 4: Prepare SFT Dataset
# ================================================================

sft_items = []
for ex in raw_data:
    messages = json.loads(ex["messages"])
    sft_items.append({"messages": messages})

sft_dataset = Dataset.from_list(sft_items)
print(f"\nSFT dataset: {len(sft_dataset)} examples")

# Verify format
assert isinstance(sft_dataset[0]["messages"], list)
assert isinstance(sft_dataset[0]["messages"][0], dict)
assert "role" in sft_dataset[0]["messages"][0]
print("Format check passed")

# Show example
msgs = sft_dataset[0]["messages"]
print(f"\nSystem: {msgs[0]['content']}")
print(f"User (first 200): {msgs[1]['content'][:200]}")
print(f"Assistant: {msgs[2]['content']}")

# Check token lengths
test_text = tokenizer.apply_chat_template(
    msgs, tokenize=False, add_generation_prompt=False
)
test_tokens = tokenizer(test_text, return_tensors="pt")
print(f"Example token length: {test_tokens.input_ids.shape[1]}")

# ================================================================
# Step 5: SFT Training
# ================================================================


def formatting_func(examples):
    """Apply chat template to convert messages to training text."""
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
    output_dir="sft_v2_checkpoint",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 16
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    optim="adamw_8bit",
    bf16=True,
    logging_steps=10,
    save_steps=200,
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
print("\nSFT config:")
print(f"  Epochs: {sft_config.num_train_epochs}")
print(
    f"  Batch: {sft_config.per_device_train_batch_size} x {sft_config.gradient_accumulation_steps} = {effective_batch}"
)
print(f"  LR: {sft_config.learning_rate}")
print(f"  Max seq length: {sft_config.max_seq_length}")
print(f"  Estimated steps: ~{est_steps}")
print(f"  Checkpoints every {sft_config.save_steps} steps")

print("\n" + "=" * 60)
print("Starting SFT training...")
print("=" * 60)
sft_trainer.train()
print("SFT training complete!")

# ================================================================
# Step 6: Save Checkpoint
# ================================================================

model.save_pretrained("sft_v2_checkpoint")
tokenizer.save_pretrained("sft_v2_checkpoint")
print("SFT checkpoint saved to sft_v2_checkpoint/")

# ================================================================
# Step 7: Merge and Save Final Model
# ================================================================

print("\nMerging model...")
try:
    model.save_pretrained_merged(
        "/workspace/your_finetuned_model_v2",
        tokenizer,
        save_method="merged_16bit",
    )
    print("Model merged and saved via Unsloth!")
except Exception as e:
    print(f"Unsloth merge failed ({e}), using PEFT fallback...")
    # Fallback: manual PEFT merge
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load base model on CPU to save VRAM
    base = AutoModelForCausalLM.from_pretrained(
        "/workspace/qwen_base",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, "sft_v2_checkpoint")
    merged = merged.merge_and_unload()
    merged.save_pretrained("/workspace/your_finetuned_model_v2")
    tok = AutoTokenizer.from_pretrained("/workspace/qwen_base")
    tok.save_pretrained("/workspace/your_finetuned_model_v2")
    print("Model merged and saved via PEFT fallback!")

# Verify
model_files = os.listdir("/workspace/your_finetuned_model_v2")
total_size = sum(
    os.path.getsize(os.path.join("/workspace/your_finetuned_model_v2", f))
    for f in model_files
    if os.path.isfile(os.path.join("/workspace/your_finetuned_model_v2", f))
)
print(f"Saved {len(model_files)} files, total size: {total_size / 1024**3:.1f} GB")
print("\nDone! Model saved to /workspace/your_finetuned_model_v2")
