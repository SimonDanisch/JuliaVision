"""Generate CPU Transformers reference logits from a local pinned checkpoint."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--tokens", type=int, default=8)
args = parser.parse_args()
torch.set_num_threads(16)
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    args.checkpoint, dtype=torch.float32, attn_implementation="eager", local_files_only=True
).eval()
ids = tokenizer.encode("The capital of France is Paris. The capital of Germany is")[:args.tokens]
reference = {}
handles = []
for i, layer in enumerate(model.model.language_model.layers):
    def hook(module, inputs, output, i=i):
        reference[f"layer.{i}"] = output[0, -1].detach().contiguous()
    handles.append(layer.register_forward_hook(hook))

cache = None
rows = []
with torch.inference_mode():
    for token in ids:
        result = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        rows.append(result.logits[0, -1].detach())
reference["logits"] = torch.stack(rows).contiguous()
save_file(reference, args.checkpoint / "reference.safetensors")
chat = tokenizer.apply_chat_template(
    [{"role": "user", "content": "In three sentences, explain why Vulkan compute dispatches need memory barriers."}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False,
)
(args.checkpoint / "reference.json").write_text(json.dumps({
    "input_ids": ids, "greedy_ids": [int(x.argmax()) for x in rows],
    "chat_ids": chat, "torch_version": torch.__version__,
    "precision": "FP32 arithmetic over native BF16 checkpoint values",
}, indent=2))
print(json.dumps({"input_ids": ids, "greedy_ids": [int(x.argmax()) for x in rows]}))
