"""CPU-only fixtures from the native Transformers router, not a copied oracle."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
import transformers
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter, Qwen3_5MoeSparseMoeBlock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(3525)
    cases = []
    with torch.no_grad():
        for experts, topk in ((8, 2), (256, 8)):
            config = Qwen3_5MoeTextConfig(hidden_size=16, num_experts=experts, num_experts_per_tok=topk)
            router = Qwen3_5MoeTopKRouter(config).to(device="cpu", dtype=torch.float32)
            router.weight.copy_(torch.randn(experts, 16, device="cpu") / 4)
            x = torch.randn(3, 16, device="cpu")
            logits, weights, selected = router(x)
            # Random non-tied probabilities avoid torch.topk's unspecified tie order.
            cases.append(dict(experts=experts, topk=topk, logits=logits.tolist(),
                              weights=weights.tolist(), selected=selected.tolist()))
        config = Qwen3_5MoeTextConfig(hidden_size=64, num_experts=8, num_experts_per_tok=2,
                                     moe_intermediate_size=24, shared_expert_intermediate_size=32)
        block = Qwen3_5MoeSparseMoeBlock(config).to(device="cpu", dtype=torch.float32)
        for parameter in block.parameters():
            parameter.normal_(0, 0.15)
        x = torch.randn(1, 3, 64, device="cpu")
        w = dict(router=block.gate.weight.tolist(), gate_up=block.experts.gate_up_proj.tolist(),
                 down=block.experts.down_proj.tolist(),
                 shared_gate_up=torch.cat((block.shared_expert.gate_proj.weight,
                                           block.shared_expert.up_proj.weight), dim=0).tolist(),
                 shared_down=block.shared_expert.down_proj.weight.tolist(),
                 shared_gate=block.shared_expert_gate.weight.tolist())
        block_case = dict(weights=w, topk=2, inputs=[x[0].tolist(), x[0].flip(0).tolist()],
                          outputs=[block(x)[0].tolist(), block(x.flip(1))[0].tolist()])
    result = dict(transformers=transformers.__version__, torch=torch.__version__, device="cpu",
                  reference_class="Qwen3_5MoeTopKRouter",
                  reference_source_sha256=hashlib.sha256(inspect.getsource(Qwen3_5MoeTopKRouter).encode()).hexdigest(),
                  cases=cases, block=block_case)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(cases)} native-router cases to {args.output}")


if __name__ == "__main__":
    main()
