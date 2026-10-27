"""Independent CPU/BF16 native35B reference. No GPU or quantized runner code.

Requires the complete pinned native download. Preserves the first output and
refuses insufficient RAM instead of widening the entire checkpoint to FP32.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main(checkpoint, output):
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[name] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if output.exists():
        raise RuntimeError("Preserve first native reference attempt")
    from download_qwen36 import REPO, REVISION
    manifest = json.loads((checkpoint / "publisher-manifest.json").read_text())
    if (manifest["repo"], manifest["revision"]) != (REPO, REVISION):
        raise RuntimeError("Wrong native checkpoint")
    weights = [f for f in manifest["files"] if f["name"].endswith(".safetensors")]
    for f in weights:
        path = checkpoint / f["name"]
        if path.stat().st_size != f["size"] or digest(path) != f["sha256"]:
            raise RuntimeError(f"Native source mismatch: {path}")
    available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))) * 1024
    if available < sum(f["size"] for f in weights) + 12 * 1024**3:
        raise RuntimeError("Need checkpoint size plus12 GiB available RAM for CPU/BF16 reference")
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration
    torch.set_num_threads(8)
    output.mkdir(parents=True)
    report = dict(complete=False, scope="Native35B CPU/BF16 numerical reference, NOT a quality benchmark",
                  repo=REPO, revision=REVISION, dtype="bfloat16", device="cpu",
                  torch=torch.__version__, transformers=transformers.__version__,
                  generator_sha256=digest(Path(__file__)),
                  model_source_sha256=hashlib.sha256(inspect.getsource(Qwen3_5MoeForConditionalGeneration).encode()).hexdigest(),
                  source_sha256={f["name"]: f["sha256"] for f in weights})
    def save_report():
        (output / "reference.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save_report()
    started = time.monotonic()
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True, dtype=torch.bfloat16,
        device_map="cpu", low_cpu_mem_usage=True, attn_implementation="eager").eval()
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("Reference must be CPU-only")
    print(f"Native CPU/BF16 model loaded in{time.monotonic()-started:.1f}s", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    # Fixed engineering prompt; not an intelligence/held-out task result.
    tokens = tokenizer.encode("Julia arrays use column-major storage.", add_special_tokens=False)[:12]
    report["input_tokens"] = tokens
    captured = {}
    routes = {}
    handles = []
    def capture_layer(i):
        def hook(module, args, result):
            x = result[0] if isinstance(result, tuple) else result
            captured[i] = x.detach().float().flatten().cpu()
        return hook
    def capture_router(i):
        def hook(module, args, result):
            routes[i] = dict(selected=result[2].detach().flatten().tolist(),
                             weights=result[1].detach().float().flatten().tolist())
        return hook
    layers = model.model.language_model.layers
    selected_layers = [0, 1, 3, len(layers)-1]
    for i, layer in enumerate(layers):
        if i in selected_layers:
            handles.append(layer.register_forward_hook(capture_layer(i)))
        handles.append(layer.mlp.gate.register_forward_hook(capture_router(i)))
    tensors = {}
    rows = []
    cache = None
    with torch.inference_mode():
        for position, token in enumerate(tokens):
            result = model(input_ids=torch.tensor([[token]], dtype=torch.long),
                           past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            logits = result.logits[0, 0].float().cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite native reference")
            tensors[f"step.{position}.logits"] = logits.contiguous()
            for i in selected_layers:
                tensors[f"step.{position}.layer.{i}"] = captured[i].contiguous()
            rows.append(dict(token=token, next_token=int(logits.argmax()),
                             routing=[routes[i] for i in range(len(layers))]))
            print(f"Reference step{position+1}/{len(tokens)} next={int(logits.argmax())}", flush=True)
    for handle in handles:
        handle.remove()
    save_file(tensors, str(output / "reference.safetensors"))
    report.update(complete=True, rows=rows, selected_layers=selected_layers,
                  tensor_sha256=digest(output / "reference.safetensors"),
                  elapsed_seconds=time.monotonic()-started)
    save_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.checkpoint, args.output)
