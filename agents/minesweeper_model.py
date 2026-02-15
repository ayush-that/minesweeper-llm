"""
Minesweeper Model - Loads fine-tuned Qwen2.5-14B-Instruct
"""
import os
import time
from typing import Optional, List

# ROCm fix for Qwen2.5 sliding window attention
os.environ["VLLM_USE_TRITON_FLASH_ATTN"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer


class MinesweeperAgent(object):
    def __init__(self, **kwargs):
        model_name = "/workspace/your_finetuned_model_v2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> tuple:
        """
        Generate LLM response for Minesweeper action.

        Returns:
            (response, token_count, generation_time)
        """
        if system_prompt is None:
            system_prompt = (
                'You are a Minesweeper AI that prioritizes flagging mines. '
                'Always look for cells that MUST be mines first. '
                'Output ONLY valid JSON: {"type":"reveal"|"flag","row":R,"col":C}'
            )

        if isinstance(message, str):
            message = [message]

        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            all_messages.append(messages)

        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        tgps_show_var = kwargs.get("tgps_show", False)

        if tgps_show_var:
            start_time = time.time()

        gen_kwargs = dict(
            max_new_tokens=kwargs.get("max_new_tokens", 64),
            do_sample=kwargs.get("do_sample", False),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Only pass sampling params when do_sample=True to avoid HF warnings
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = kwargs.get("temperature", 1.0)
            gen_kwargs["top_p"] = kwargs.get("top_p", 1.0)
        if "repetition_penalty" in kwargs:
            gen_kwargs["repetition_penalty"] = kwargs["repetition_penalty"]

        generated_ids = self.model.generate(**model_inputs, **gen_kwargs)

        if tgps_show_var:
            generation_time = time.time() - start_time

        batch_outs = self.tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        batch_outs = [output.strip() for output in batch_outs]

        if tgps_show_var:
            token_len = sum(len(generated_ids[i]) - model_inputs.input_ids.shape[1]
                          for i in range(len(generated_ids)))
            return (
                batch_outs[0] if len(batch_outs) == 1 else batch_outs,
                token_len,
                generation_time,
            )

        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None


if __name__ == "__main__":
    agent = MinesweeperAgent()
    test_prompt = 'MINESWEEPER 4x4 MINES:3 FLAGS:0 LEFT:3\n....\n....\n....\n....\nRULES: .=hidden F=flag 0-8=adjacent mines\nOutput ONLY: {"type":"reveal"|"flag","row":R,"col":C}'
    response, tl, tm = agent.generate_response(test_prompt, tgps_show=True, max_new_tokens=64)
    print(f"Response: {response}")
    if tl and tm:
        print(f"Tokens: {tl}, Time: {tm:.2f}s")
