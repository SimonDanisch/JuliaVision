"""Expanded installed-API corpus; no task/candidate/test inputs.

Coverage extension designed after observing prior retrieval limitations. It is
NOT retrofitted into frozen studies and has no new neural-quality evidence.
Trusted introspection runs in the existing CPU-only/networkless jail.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from bigcodebench_x_julia_adapter import julia_sandbox_command, JULIA_COMMAND, ROOT

MODULES = ("Base", "Makie", "CairoMakie", "JSON3", "Base64", "Statistics", "LinearAlgebra", "Random",
           "Dates", "Printf", "DelimitedFiles", "TOML", "SHA", "Serialization", "SparseArrays", "Unicode", "UUIDs")
PROGRAM = 'using CairoMakie, JSON3, Base64, Statistics, LinearAlgebra, Random, Dates, Printf, DelimitedFiles, TOML, SHA, Serialization, SparseArrays, Unicode, UUIDs\n'
PROGRAM += 'const CAPTURE_MODULES=(' + ','.join(json.dumps(name) for name in MODULES) + ',)\n'
PROGRAM += r'''
rows=Any[];identities=IdDict{Any,Int}();bindings=Dict{String,Vector{String}}()
for module_name in CAPTURE_MODULES
    mod=getfield(Main,Symbol(module_name))
    # names() alone need not enumerate every implicitly available Base/Core
    # binding. Verify effective candidates with isdefined, never call them.
    candidates=union(names(mod;all=true,imported=true),names(Base;all=false),names(Core;all=false))
    bindings[module_name]=sort!([string(s) for s in candidates if isdefined(mod,s)])
    for symbol in sort!(names(mod;all=false,imported=false);by=string)
        name=string(symbol)
        Base.isidentifier(name)&&isdefined(mod,symbol)||continue
        value=getfield(mod,symbol)
        (value isa Function||value isa DataType||value isa UnionAll)||continue
        identity=get!(identities,value) do
            length(identities)+1
        end
        documentation=Base.Docs.doc(Base.Docs.Binding(mod,symbol))
        text=documentation===nothing ? "" : sprint(show,MIME("text/plain"),documentation)
        push!(rows,(;qualified=module_name*"."*name,text=first(text,8000),binding_id=identity))
    end
    println(stderr,"CAPTURED_MODULE ",module_name," TOTAL_DOCS ",length(rows));flush(stderr)
end
println(JSON3.write((;rows,bindings,modules=collect(CAPTURE_MODULES),julia_version=string(VERSION),gpu_nodes_visible=isdir("/dev/dri"))))
'''


def trusted_capture_command():
    # The candidate executor intentionally minimizes compilation. This trusted
    # introspection loop is not a candidate; compiling it avoids interpreting
    # every namespace/docstring iteration. Never mutate JULIA_COMMAND itself.
    return ["--compile=yes" if arg == "--compile=min" else "-O1" if arg == "-O0" else arg for arg in JULIA_COMMAND]


def validate(data):
    if data["modules"] != list(MODULES) or data.get("gpu_nodes_visible") is not False:
        raise ValueError("Wrong module palette or visible hardware device")
    names = [row["qualified"] for row in data["rows"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate documents")
    if set(data["bindings"]) != set(MODULES):
        raise ValueError("Incomplete effective-binding inventory")
    for module, names in data["bindings"].items():
        if not names or names != sorted(set(names)) or not all(isinstance(name, str) for name in names):
            raise ValueError("Invalid binding inventory")
    for row in data["rows"]:
        namespace, name = row["qualified"].split(".", 1)
        if namespace not in MODULES or name not in data["bindings"][namespace]:
            raise ValueError("Document absent from effective namespace")
        if type(row["binding_id"]) is not int or row["binding_id"] <= 0 or not isinstance(row["text"], str) or len(row["text"]) > 8000:
            raise ValueError("Invalid doc/identity bounds")


def capture(environment, destination):
    if destination.exists():
        raise FileExistsError("Preserve expanded corpus")
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="julia-docs-v2-") as temporary:
        task = Path(temporary)
        (task / "program.jl").write_text(PROGRAM)
        command = julia_sandbox_command(task, environment)
        command[-2:] = trusted_capture_command()
        result = subprocess.run(command, capture_output=True, text=True, timeout=190)
        if result.returncode:
            report = {"complete": False, "scope": "CPU-only source capture failure, NOT model quality", "returncode": result.returncode,
                      "seconds": time.monotonic() - started, "source_sha256": source_sha256,
                      "stderr_tail": result.stderr[-4096:], "stdout_tail": result.stdout[-4096:]}
            for attempt in range(1, 11):
                failure = destination.with_name(destination.stem + f"_capture_failure_{attempt}.json")
                if not failure.exists():
                    failure.write_text(json.dumps(report, indent=2) + "\n")
                    break
            raise RuntimeError(f"capture exit {result.returncode} after {report['seconds']:.2f}s; stderr: {result.stderr[-4096:]}")
        if len(result.stdout.encode()) > 16 * 1024 * 1024:
            raise ValueError("Bounded capture output exceeded")
        data = json.loads(result.stdout)
    validate(data)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(Path(__file__)) != source_sha256:
        raise RuntimeError("Capture source changed during execution; no corpus committed")
    data.update(complete=True,
                scope="Posthoc coverage extension: exported callable/type docstrings and actual identity; effective isdefined-checked namespace names. Capture reads NO tasks/candidates/tests and runs with NO GPU nodes/network. Not call/type correctness or model quality; old frozen corpus/studies unchanged.",
                project_sha256=digest(ROOT / "Project.toml"), manifest_sha256=digest(ROOT / "Manifest.toml"),
                source_sha256=source_sha256, capture_program_sha256=hashlib.sha256(PROGRAM.encode()).hexdigest(),
                capture_seconds=time.monotonic() - started,
                compiler_policy="Trusted capture --compile=yes -O1; candidate benchmark command unchanged")
    destination.write_text(json.dumps(data, indent=2) + "\n")
    return {"records": len(data["rows"]), "modules": len(MODULES), "actual_distinct_bindings": len({r["binding_id"] for r in data["rows"]})}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(capture(Path(sys.executable).absolute().parent.parent, args.destination)))
