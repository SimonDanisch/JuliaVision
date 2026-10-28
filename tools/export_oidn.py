"""Export pinned upstream OIDN networks to JuliaVision. Apache-2.0 upstream
code/weights; see tools/licenses/oidn. No retraining or replacement weights."""
import argparse, ast, hashlib, importlib.util, json, pathlib, sys, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
import export_graphs as EG

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkout",type=pathlib.Path,required=True)
    p.add_argument("--output",type=pathlib.Path,required=True)
    p.add_argument("--references",type=pathlib.Path,
                   help="Reference fixture directory (defaults to --output)")
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    references=a.references if a.references is not None else a.output
    references.mkdir(parents=True,exist_ok=True)
    checkout=a.checkout.resolve()
    tza=load_module(checkout/"training/tza.py","oidn_tza")
    # Load only the upstream network definitions; training imports require
    # unrelated dataset/image packages. The network implementation is unchanged.
    source=ast.parse((checkout/"training/model.py").read_text())
    wanted={"Conv","relu","pool","upsample","concat","UNet","UNetLarge"}
    definitions=[node for node in source.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in wanted]
    scope={"torch":torch,"nn":nn,"F":F,"np":np}
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(checkout/"training/model.py"),"exec"),scope)
    manifest={}
    torch.set_num_threads(4)
    for name in ("rt_alb_large","rt_nrm_large","rt_hdr_calb_cnrm_large"):
        file=checkout/"weights"/(name+".tza")
        data=file.read_bytes()
        if data.startswith(b"version https://git-lfs"):
            pointer=dict(line.split(" ",1) for line in data.decode().splitlines()[1:])
            digest=pointer["oid"].split(":")[1]
            target=a.output/"inputs"/file.name;target.parent.mkdir(exist_ok=True)
            if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
                urllib.request.urlretrieve("https://media.githubusercontent.com/media/RenderKit/oidn-weights/28883d1769d5930e13cf7f1676dd852bd81ed9e7/"+file.name,target)
            assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
            file=target
        archive=tza.Reader(str(file))
        state={}
        for key in archive._table:
            value,layout=archive[key]
            if layout not in ("oihw","x"):raise ValueError((key,layout))
            state[key]=torch.from_numpy(np.array(value,copy=True)).float()
        channels=state["enc_conv1a.weight"].shape[1]
        model=scope["UNetLarge"](channels,3).eval()
        model.load_state_dict(state,strict=True)
        torch.manual_seed(123)
        x=torch.rand(1,channels,64,64)
        shapes=({2:16*torch.export.Dim("h",min=1,max=512),
                 3:16*torch.export.Dim("w",min=1,max=512)},)
        with torch.no_grad():
            ep=torch.export.export(model,(x,),dynamic_shapes=shapes,strict=False).run_decompositions()
            y=model(x)
        graph=EG.convert(ep,({2:"h",3:"w"},),name)
        (a.output/(name+".json")).write_text(json.dumps(graph,indent=1))
        save_file({**state,**EG.save_constants(ep)},str(a.output/(name+".safetensors")))
        save_file({"input":x,"output":y},str(references/(name+"-reference.safetensors")))
        manifest[name]={"channels":channels,"weights_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
            "ops":len(graph["ops"])}
        print(name,manifest[name],flush=True)
    manifest["upstream_commit"]="6602ee2ca38a1a2a02135beed8f6e68eed630180"
    manifest["weights_commit"]="28883d1769d5930e13cf7f1676dd852bd81ed9e7"
    (a.output/"provenance.json").write_text(json.dumps(manifest,indent=2))
if __name__=="__main__":main()

