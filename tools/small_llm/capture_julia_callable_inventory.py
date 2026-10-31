"""Capture published callable names, not candidate programs or benchmark data.

This is independent of the already-frozen namespace diagnostic intervention.
The inventory is useful for sparse token constraints, not proof of call validity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from bigcodebench_x_julia_adapter import julia_sandbox_command, JULIA_COMMAND, ROOT

PROGRAM = r'''
using CairoMakie, JSON3, Base64, Statistics, LinearAlgebra, Random
rows=Any[]
for module_name in ("Makie","CairoMakie","JSON3","Base64","Statistics","LinearAlgebra","Random")
    mod=getfield(Main,Symbol(module_name))
    for symbol in sort!(names(mod;all=false,imported=false);by=string)
        name=string(symbol)
        Base.isidentifier(name) && isdefined(mod,symbol) || continue
        value=getfield(mod,symbol)
        (value isa Function || value isa DataType || value isa UnionAll) || continue
        push!(rows,(;qualified=module_name*"."*name,
            kind=value isa Function ? "function" : "type"))
    end
end
println(JSON3.write((;rows,julia_version=string(VERSION))))
'''


def capture(environment, destination):
    if destination.exists():
        raise FileExistsError("Preserve callable inventory")
    with tempfile.TemporaryDirectory(prefix="julia-callables-") as root:
        task = Path(root)
        (task / "program.jl").write_text(PROGRAM)
        command = julia_sandbox_command(task, environment)
        command[-2:] = JULIA_COMMAND
        result = subprocess.run(command, capture_output=True, text=True, timeout=190)
        if result.returncode:
            raise RuntimeError(result.stderr[-4096:])
        data = json.loads(result.stdout)
    names = [row["qualified"] for row in data["rows"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Empty or duplicate callable inventory")
    if not all(row["kind"] in ("function", "type") for row in data["rows"]):
        raise ValueError("Invalid binding classification")
    data.update({
        "complete": True,
        "scope": "Installed exported callable/type names only. No signatures, replacement APIs, semantic relevance or valid argument combinations guaranteed. No generated candidate or benchmark tests read.",
        "project_sha256": hashlib.sha256((ROOT / "Project.toml").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((ROOT / "Manifest.toml").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    destination.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"callables": len(names), "modules": len({n.split('.')[0] for n in names})}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    capture(Path(sys.executable).absolute().parent.parent, args.destination)
