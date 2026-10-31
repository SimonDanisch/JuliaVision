"""Export a dynamic-batch graph for a local throughput/parity experiment.

Weights and tokenizer remain the existing Qwen3EmbeddingRunner artifacts. This
exports only the graph and any additional constants, not a new weight checkpoint.
"""
import json
from pathlib import Path
import torch
from torch.export import Dim
from safetensors import safe_open
from safetensors.torch import save_file
import export_qwen3embedding as Q


def main():
    torch.set_num_threads(8)
    model = Q.load(Q.DT)
    model.rotary_emb = Q.StaticRotary(model.rotary_emb, Q.MAX_TOKENS + 1, Q.DT)
    wrapper = Q.StaticQwen3Encoder(model).eval()
    example = torch.zeros(2, 16, model.config.hidden_size, dtype=Q.DT)
    shapes = ({0: Dim.DYNAMIC(min=2, max=8), 1: Dim.DYNAMIC(min=2, max=Q.MAX_TOKENS)},)
    with torch.no_grad():
        program = torch.export.export(wrapper, (example,), dynamic_shapes=shapes,
                                     strict=False).run_decompositions()
    graph = Q.EG.convert(program, ({0: "b", 1: "t"},), "qwen3embedding_batch")
    available = {**dict(wrapper.state_dict()), **Q.EG.save_constants(program)}
    tensors = {}
    Q.prune(graph, available, tensors)
    # Assert that shared names really are the same weights/rotary constants.
    # Only additional lifted constants may need a separate small artifact.
    extra = {}
    with safe_open(str(Q.OUT / "weights.safetensors"), framework="pt") as old:
        for name, tensor in tensors.items():
            if name in old.keys():
                if not torch.equal(tensor, old.get_tensor(name)):
                    raise RuntimeError(f"Existing tensor changed: {name}")
            else:
                extra[name] = tensor.detach().contiguous()
    out = Q.ROOT / "gen" / "qwen3embedding_batch_bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / "qwen3embedding_batch.json").write_text(json.dumps(graph, indent=1))
    if extra:
        save_file(extra, str(out / "constants.safetensors"))
    print(f"Exported {len(graph['ops'])} ops; reused {len(tensors)-len(extra)} tensors; {len(extra)} new constants", flush=True)


if __name__ == "__main__":
    main()
