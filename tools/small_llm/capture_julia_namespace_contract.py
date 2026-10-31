"""Capture installed binding names, CPU-only jail; no candidate code executed."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from bigcodebench_x_julia_adapter import julia_sandbox_command, JULIA_COMMAND, ROOT

PROGRAM = '''
using CairoMakie, JSON3, Base64, Statistics, LinearAlgebra, Random
modules=("Makie","CairoMakie","JSON3","Base","Base64","Statistics","LinearAlgebra","Random")
bindings=Dict(name=>sort!(string.(filter(symbol->isdefined(getfield(Main,Symbol(name)),symbol),
    names(getfield(Main,Symbol(name));all=true,imported=true)))) for name in modules)
println(JSON3.write((;bindings,julia_version=string(VERSION))))
'''


def capture(environment, destination):
    if destination.exists():
        raise FileExistsError("Preserve namespace snapshot")
    with tempfile.TemporaryDirectory(prefix="julia-namespace-") as root:
        task=Path(root);(task/"program.jl").write_text(PROGRAM)
        command=julia_sandbox_command(task,environment)
        command[-2:]=JULIA_COMMAND
        result=subprocess.run(command,capture_output=True,text=True,timeout=190)
        if result.returncode:
            raise RuntimeError(result.stderr[-4096:])
        data=json.loads(result.stdout)
    data.update({"complete":True,"scope":"Installed namespace binding presence ONLY, not call signatures, reachability or semantic correctness. No generated code, benchmark tests or expected answers used.",
                 "project_sha256":hashlib.sha256((ROOT/"Project.toml").read_bytes()).hexdigest(),
                 "manifest_sha256":hashlib.sha256((ROOT/"Manifest.toml").read_bytes()).hexdigest(),
                 "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    destination.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps({"modules":len(data["bindings"]),"bindings":sum(map(len,data["bindings"].values()))}))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("destination",type=Path);args=parser.parse_args()
    capture(Path(sys.executable).absolute().parent.parent,args.destination)
