"""Independent native CPU reference for exact prompts and long prefill.

No remote code, model-generated code execution, or Radeon operations. Reuse the
existing Torch/Transformers installation; output logits are FP32 safetensors.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--only-case")
    parser.add_argument("--generation-tokens", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.generation_tokens <= 512:
        parser.error("generation token budget must be 1..512")
    data = json.loads(args.inputs.read_text())
    checkpoint = Path(data["checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    rows = []
    for case in data["cases"]:
        if args.only_case and case["id"] != args.only_case:
            continue
        prompt = tokenizer.apply_chat_template(case["messages"], tokenize=False,
                    add_generation_prompt=True, enable_thinking=case["thinking"])
        ids = tokenizer.apply_chat_template(case["messages"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=case["thinking"], return_dict=False)
        julia_ids = case["julia_ids"]
        mismatch = next((i for i, (a, b) in enumerate(zip(ids, julia_ids)) if a != b), None)
        if mismatch is None and len(ids) != len(julia_ids):
            mismatch = min(len(ids), len(julia_ids))
        row = dict(id=case["id"], native_prompt=prompt, native_ids=ids,
                   julia_ids=julia_ids, identical=ids == julia_ids,
                   first_mismatch_zero_based=mismatch, validate_logits=case["validate_logits"])
        rows.append(row)
        print(case["id"], "prompt parity", row["identical"], len(ids), len(julia_ids), flush=True)
    result = dict(scope="native template parity; FP32 model consumes EXACT Julia IDs to isolate inference from rendering",
                  torch=torch.__version__, transformers=transformers.__version__, device="cpu",
                  reference_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
                  generation_budget=args.generation_tokens, rows=rows)
    if not rows:
        parser.error("no matching case")
    args.output.write_text(json.dumps(result, indent=2))
    if args.tokenizer_only:
        return
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation="eager", local_files_only=True,
    ).to("cpu").eval()
    tensors = {}
    with torch.inference_mode():
        for row in rows:
            if not row["validate_logits"]:
                continue
            started = time.monotonic()
            ids = torch.tensor([row["julia_ids"]], dtype=torch.long, device="cpu")
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=True, logits_to_keep=1)
            logits = output.logits[0, -1].float().contiguous()
            tensors[row["id"]] = logits
            greedy = []
            eos = model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            for step in range(args.generation_tokens):
                token = int(logits.argmax())
                greedy.append(token)
                # The original eight-token parity fixtures retain fixed length.
                if step == args.generation_tokens - 1 or (args.generation_tokens > 8 and token in eos):
                    break
                next_ids = torch.tensor([[token]], dtype=torch.long)
                output = model(input_ids=next_ids, past_key_values=output.past_key_values,
                    attention_mask=torch.ones((1, ids.shape[1] + step + 1), dtype=torch.long),
                    use_cache=True, logits_to_keep=1)
                logits = output.logits[0, -1].float().contiguous()
            row.update(reference_greedy_ids=greedy, elapsed_seconds=time.monotonic() - started,
                       reference_greedy_text=tokenizer.decode(greedy, skip_special_tokens=False))
            save_file(tensors, str(args.output.with_suffix(".safetensors")))
            args.output.write_text(json.dumps(result, indent=2))
            print(row["id"], "FP32 reference ready", row["elapsed_seconds"], row["reference_greedy_text"], flush=True)


if __name__ == "__main__":
    main()
