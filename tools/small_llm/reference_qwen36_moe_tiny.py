"""Independent CPU/FP32 Transformers FULL hybrid-MoE fixture, not35B parity."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path


def main(output):
    for key in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"):
        os.environ[key]=""
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    if output.exists():
        raise FileExistsError("Preserve first reference fixture")
    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    torch.set_num_threads(4);torch.manual_seed(3635)
    config=Qwen3_5MoeTextConfig(vocab_size=256,hidden_size=256,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=1,head_dim=256,
        linear_key_head_dim=128,linear_value_head_dim=128,linear_num_key_heads=2,linear_num_value_heads=4,
        linear_conv_kernel_dim=4,layer_types=["linear_attention"]*3+["full_attention"],
        full_attention_interval=4,num_experts=4,num_experts_per_tok=2,moe_intermediate_size=256,
        shared_expert_intermediate_size=256,tie_word_embeddings=False,max_position_embeddings=256,
        rope_parameters={"rope_type":"default","rope_theta":10000000.,"partial_rotary_factor":0.25},
        attn_implementation="eager")
    model=Qwen3_5MoeForCausalLM(config).to(device="cpu",dtype=torch.float32).eval()
    if sum(p.numel() for p in model.parameters())>30_000_000:
        raise ValueError("Tiny fixture allocation bound")
    states=model.state_dict()
    if any(p.device.type!="cpu" or p.dtype!=torch.float32 for p in states.values()):
        raise ValueError("CPU/FP32 fixture only")
    output.mkdir(parents=True)
    save_file({k:v.contiguous() for k,v in states.items()},str(output/"model.safetensors"))
    (output/"config.json").write_text(json.dumps(dict(text_config=config.to_dict()),indent=2)+"\n")
    layer_values={};routes={};handles=[]
    def capture_layer(i):
        def hook(module,args,result):
            x=result[0] if isinstance(result,tuple) else result
            layer_values[i]=x.detach().flatten().tolist()
        return hook
    def capture_router(i):
        def hook(module,args,result):
            routes[i]=dict(selected=result[2].detach().flatten().tolist(),weights=result[1].detach().flatten().tolist())
        return hook
    for i,layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(capture_layer(i)))
        handles.append(layer.mlp.gate.register_forward_hook(capture_router(i)))
    tokens=[3,19,8,42,2,17,89,9];cache=None;rows=[]
    with torch.inference_mode():
        for token in tokens:
            result=model(input_ids=torch.tensor([[token]],dtype=torch.long),past_key_values=cache,use_cache=True)
            cache=result.past_key_values
            rows.append(dict(token=token,logits=result.logits[0,0].tolist(),
                             layers=[layer_values[i] for i in range(4)],routing=[routes[i] for i in range(4)]))
    for handle in handles: handle.remove()
    report=dict(scope="Random tiny FULL4-layer hybrid MoE native Transformers fixture; NOT35B checkpoint or throughput/quality proof",
        device="cpu",dtype="float32",torch=torch.__version__,transformers=transformers.__version__,rows=rows,
        model_source_sha256=hashlib.sha256(inspect.getsource(Qwen3_5MoeForCausalLM).encode()).hexdigest(),
        fixture_sha256={f:hashlib.sha256((output/f).read_bytes()).hexdigest() for f in ("config.json","model.safetensors")},
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/"reference.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(complete=True,steps=len(rows),parameters=sum(p.numel() for p in model.parameters()))))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output",type=Path);main(p.parse_args().output)
