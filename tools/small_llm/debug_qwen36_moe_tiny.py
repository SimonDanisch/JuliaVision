"""Stage diagnostics for the SAME saved tiny fixture; never replaces reference."""
import json
import os
import sys
from pathlib import Path
for key in ("CUDA_VISIBLE_DEVICES","HIP_VISIBLE_DEVICES","ROCR_VISIBLE_DEVICES"):
    os.environ[key]=""
import torch
from safetensors.torch import load_file
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
root=Path(sys.argv[1]);output=root/"first_stage_diagnostics.json"
if output.exists(): raise FileExistsError(output)
torch.set_num_threads(4)
c=Qwen3_5MoeTextConfig(**json.loads((root/"config.json").read_text())["text_config"])
c._attn_implementation="eager"
model=Qwen3_5MoeForCausalLM(c).float().cpu().eval()
model.load_state_dict(load_file(str(root/"model.safetensors")))
data={};handles=[]
def hook(name):
    def capture(module,args,result):
        if isinstance(result,torch.Tensor): data[name]=result.detach().flatten().tolist()
    return capture
for name,module in model.named_modules():
    if name.startswith("model.layers.0."):
        handles.append(module.register_forward_hook(hook(name)))
with torch.inference_mode():
    result=model(input_ids=torch.tensor([[3]],dtype=torch.long),use_cache=True)
    data["logits"]=result.logits.flatten().tolist()
for handle in handles: handle.remove()
output.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
print("Captured same-fixture first-layer diagnostics")
