"""Posthoc ALL8 direct-prompt FP32/native diagnostic, CPU ONLY.

Different precision AND greedy decoding: not an isolated quantization ablation,
not a replacement for the frozen sampled primary scores. No gold/reference
answers are accepted in the request packet or loaded by this program.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time


def validate_requests(packet):
    jobs = packet["jobs"]
    if len(jobs) != 8 or len({j["id"] for j in jobs}) != 8:
        raise ValueError("ALL8 unique original direct prompts required")
    for job in jobs:
        if set(job) != {"id", "task", "messages", "input_ids"}:
            raise ValueError("Public prompts/IDs ONLY, no answer fields")
        if not re.fullmatch(r"bayes_extract_0[1-8]_direct_probability", job["id"]):
            raise ValueError("Original direct cohort only")
        if not 0 < len(job["input_ids"]) <= 7136 or not all(
            type(x) is int and 0 <= x < 1_000_000 for x in job["input_ids"]
        ):
            raise ValueError("Bounded token IDs required")
        if len(job["messages"]) != 2 or [m["role"] for m in job["messages"]] != ["system", "user"]:
            raise ValueError("Original task-only conversation required")
        if not all(set(m) == {"role", "content"} and isinstance(m["content"], str) for m in job["messages"]):
            raise ValueError("Public role/content strings only")
    return jobs


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Preserve first diagnostic; no regeneration/resume")
    args.output.mkdir(parents=True, exist_ok=True)
    # This process must never initialize the blocked Bosgame Radeon or any GPU.
    for variable in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[variable] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    limit = 64 * 1024**3
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    packet = json.loads(args.requests.read_text())
    jobs = validate_requests(packet)
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    matches = []
    for job in jobs:
        native = tokenizer.apply_chat_template(job["messages"], tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=False)
        matches.append({"id": job["id"], "matches": native == job["input_ids"], "input_tokens": len(native)})
    save(args.output / "tokenizer_validation.json", {"rows": matches, "all_match": all(r["matches"] for r in matches)})
    if not all(r["matches"] for r in matches):
        raise ValueError("Native chat IDs differ; STOP before model generation")
    checkpoint_files = [p for p in args.checkpoint.iterdir() if p.name in (
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"
    ) or p.suffix == ".safetensors" and p.name.startswith("model.")]
    registration = {
        "scope": "POSTHOC ALL8 unchanged direct prompts. FP32/native Transformers GREEDY diagnostic, NOT same-sampler/Q4-only comparison or replacement primary score.",
        "requests_sha256": sha_file(args.requests), "script_sha256": sha_file(Path(__file__)),
        "checkpoint_sha256": {p.name: sha_file(p) for p in checkpoint_files},
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": "cpu", "dtype": "float32 over native checkpoint values", "do_sample": False,
        "max_new_tokens": 1024, "per_case_soft_time_seconds": 90, "private_answers_loaded": False,
        "memory_address_space_limit_bytes": limit,
    }
    save(args.output / "conditions.json", registration)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.checkpoint, dtype=torch.float32,
        attn_implementation="eager", local_files_only=True).eval()
    if any(p.device.type != "cpu" or p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("CPU/FP32-only diagnostic required")
    for job in jobs:
        save(args.output / (job["id"] + "_attempt.json"), {"id": job["id"], "started_unix": time.time()})
        started = time.monotonic()
        try:
            ids = torch.tensor([job["input_ids"]], dtype=torch.long, device="cpu")
            with torch.inference_mode():
                generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=1024, do_sample=False, use_cache=True, max_time=90)
            tokens = generated[0, ids.shape[1]:].tolist()
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            ended = bool(tokens and tokens[-1] in eos)
            result = {"id": job["id"], "answer": tokenizer.decode(tokens, skip_special_tokens=True),
                "raw_answer": tokenizer.decode(tokens, skip_special_tokens=False), "generated_ids": tokens,
                "eos_completed": ended, "truncated_or_time_limited": not ended,
                "elapsed_seconds": time.monotonic() - started, "input_tokens": len(job["input_ids"])}
        except Exception as exc:
            result = {"id": job["id"], "inference_error": str(exc), "elapsed_seconds": time.monotonic() - started}
            save(args.output / (job["id"] + ".json"), result)
            raise
        save(args.output / (job["id"] + ".json"), result)
        print(json.dumps({"id": job["id"], "generated_tokens": len(tokens), "eos_completed": ended}), flush=True)
    save(args.output / "completion.json", {"complete": True, "jobs": len(jobs), "device": "cpu"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("requests", type=Path)
    parser.add_argument("output", type=Path)
    run(parser.parse_args())
