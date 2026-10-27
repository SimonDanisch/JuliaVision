"""Task-blind public binding identity metadata; preserve old docstrings exactly."""
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
rows=Any[];identities=IdDict{Any,Int}()
for module_name in ("Makie","CairoMakie","JSON3","Base64","Statistics","LinearAlgebra","Random")
    mod=getfield(Main,Symbol(module_name))
    for symbol in sort!(names(mod;all=false,imported=false);by=string)
        name=string(symbol)
        Base.isidentifier(name) && isdefined(mod,symbol) || continue
        value=getfield(mod,symbol)
        (value isa Function || value isa DataType || value isa UnionAll) || continue
        identity=get!(identities,value) do
            length(identities)+1
        end
        push!(rows,(;qualified=module_name*"."*name,binding_id=identity))
    end
end
println(JSON3.write((;rows,julia_version=string(VERSION))))
'''


def attach_bindings(documents, captured):
    if not documents.get("complete"):
        raise ValueError("Incomplete public doc snapshot")
    old = documents["rows"]; identities = captured["rows"]
    names = [row["qualified"] for row in old]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty/duplicate public names")
    if names != [row["qualified"] for row in identities]:
        raise ValueError("Binding/doc cohort or order drift")
    if any(type(row["binding_id"]) is not int or row["binding_id"] <= 0 for row in identities):
        raise ValueError("Invalid binding identity")
    if documents["julia_version"] != captured["julia_version"]:
        raise ValueError("Julia version drift")
    # Identical text alone is NOT evidence of aliasing. The actual Julia ===
    # identity is captured with IdDict; group keys must include BOTH fields.
    return [dict(row, binding_id=identity["binding_id"]) for row, identity in zip(old, identities)]


def capture(documents_path, environment, destination):
    if destination.exists():
        raise FileExistsError("Preserve binding capture")
    documents = json.loads(documents_path.read_text())
    for name, field in (("Project.toml", "project_sha256"), ("Manifest.toml", "manifest_sha256")):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != documents[field]:
            raise ValueError("Environment changed since public documentation capture")
    with tempfile.TemporaryDirectory(prefix="julia-binding-identity-") as root:
        task = Path(root); (task / "program.jl").write_text(PROGRAM)
        command = julia_sandbox_command(task, environment); command[-2:] = JULIA_COMMAND
        result = subprocess.run(command, capture_output=True, text=True, timeout=190)
        if result.returncode:
            raise RuntimeError(result.stderr[-4096:])
        captured = json.loads(result.stdout)
    rows = attach_bindings(documents, captured)
    packet = dict(complete=True, rows=rows, julia_version=captured["julia_version"],
                  scope="Task-blind source metadata captured AFTER the earlier retrieval study. Actual === binding identity, NOT inferred from doc similarity; original public documentation text/order retained byte-for-byte. No task, candidate, hidden tests or query consulted. Not model quality.",
                  parent_docs_sha256=hashlib.sha256(documents_path.read_bytes()).hexdigest(),
                  project_sha256=documents["project_sha256"], manifest_sha256=documents["manifest_sha256"],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  capture_program_sha256=hashlib.sha256(PROGRAM.encode()).hexdigest())
    destination.write_text(json.dumps(packet, indent=2) + "\n")
    print(json.dumps(dict(records=len(rows), actual_distinct_bindings=len({r["binding_id"] for r in rows}))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("documents", type=Path); parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    capture(args.documents, Path(sys.executable).absolute().parent.parent, args.destination)
