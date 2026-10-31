"""Task-independent installed public docstrings for automatic context retrieval."""
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
        documentation=Base.Docs.doc(Base.Docs.Binding(mod,symbol))
        text=documentation===nothing ? "" : sprint(show,MIME("text/plain"),documentation)
        # Cap each record independently, before any task is consulted.
        text=first(text,8000)
        push!(rows,(;qualified=module_name*"."*name,text))
    end
end
println(JSON3.write((;rows,julia_version=string(VERSION))))
'''


def capture(environment, destination):
    if destination.exists():
        raise FileExistsError("Preserve public documentation snapshot")
    with tempfile.TemporaryDirectory(prefix="julia-public-docs-") as root:
        task = Path(root)
        (task / "program.jl").write_text(PROGRAM)
        command = julia_sandbox_command(task, environment)
        command[-2:] = JULIA_COMMAND
        result = subprocess.run(command, capture_output=True, text=True, timeout=190)
        if result.returncode:
            raise RuntimeError(result.stderr[-4096:])
        data = json.loads(result.stdout)
    names = [row["qualified"] for row in data["rows"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate documentation inventory")
    data.update({
        "complete": True,
        "scope": "Installed exported callable/type docstrings, captured without any task, generated candidate or benchmark tests. Per-record first8000 characters, lexical retrieval only; documentation is not proof of correctness.",
        "project_sha256": hashlib.sha256((ROOT / "Project.toml").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((ROOT / "Manifest.toml").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    destination.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"records": len(names), "characters": sum(len(r["text"]) for r in data["rows"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    capture(Path(sys.executable).absolute().parent.parent, args.destination)
