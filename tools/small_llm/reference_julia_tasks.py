"""Independent CPU-native control for the Julia coding diagnostic.

Consumes prompts only, not tests/references. Never executes model-generated code.
Run with an existing Torch/Transformers environment; no Radeon is used.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--tasks", nargs="+", default=["stable_groups", "makie_band"])
    parser.add_argument("--max-tokens", type=int, default=320)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.checkpoint, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True,
    ).to("cpu").eval()
    for task in args.tasks:
        path = args.results / f"{task}_native_cpu.json"
        if path.exists():
            continue
        baseline = json.loads((args.results / f"{task}_baseline.json").read_text())
        messages = [dict(role=role, content=content)
                    for message in baseline["messages"] for role, content in message.items()]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt", return_dict=True,
        )
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(
                **encoded, do_sample=False, max_new_tokens=args.max_tokens,
                repetition_penalty=1.0, use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1]:].tolist()
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        row = dict(task=task, arm="native_cpu", precision="FP32 over native BF16 checkpoint",
                   model=str(args.checkpoint), torch_version=torch.__version__,
                   transformers_version=transformers.__version__,
                   config_sha256=hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest(),
                   input_ids=encoded["input_ids"][0].tolist(), generated_ids=generated,
                   answer=answer, budget_tokens=args.max_tokens,
                   elapsed_seconds=time.monotonic()-started)
        path.write_text(json.dumps(row, indent=2))
        print(task, len(generated), row["elapsed_seconds"], answer[:100], flush=True)


if __name__ == "__main__":
    main()
