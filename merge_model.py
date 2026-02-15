#!/usr/bin/env python3
"""Merge GRPO LoRA adapter into base model and save."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_path = "/root/.cache/huggingface/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
adapter_path = "/workspace/grpo_outputs/checkpoint-400"
output_path = "/workspace/your_finetuned_model"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # merge on CPU to avoid GPU memory issues
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print(f"Loading adapter from {adapter_path}...")
model = PeftModel.from_pretrained(model, adapter_path)

print("Merging adapter into base model...")
model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)

total_size = sum(
    os.path.getsize(os.path.join(output_path, f))
    for f in os.listdir(output_path)
    if os.path.isfile(os.path.join(output_path, f))
)
print(
    f"Done! Files: {len(os.listdir(output_path))}, Total: {total_size / 1024**3:.1f} GB"
)
