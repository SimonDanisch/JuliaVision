"""Scaffold the runner packages in dev/JuliaVision from the registry in models.py.

    uv run tools/newrunner.py            # write any package that does not exist
    uv run tools/newrunner.py --force    # overwrite them

A runner package is small and its shape is fixed — `SAM2Runner` and
`MatAnyoneRunner` are the same file with a different model in the middle — so the
skeleton is generated rather than copy-pasted nine times. What is *not*
generated is the part that matters: the workload body, which has to drive the
real call the editor makes and is different for every model. Each generated
package carries a marked TODO there and a docstring saying why.

The generated package loads and precompiles with no assets present. That is the
whole reason it can be committed before the port works: `assetdir` refuses with
the three steps that would fix it, `ready()` answers `false` so the workload
precompiles to nothing, and `Pkg.precompile` on a machine with no assets
produces a package with nothing cached rather than an error.
"""

import argparse

from models import MODELS, ROOT

JV = ROOT / "dev" / "JuliaVision"

PROJECT = """\
name = "{package}"
uuid = "{uuid}"
version = "0.1.0"

[deps]
DNNKernels = "b7e4a0c2-3f51-4d8e-9a1b-6c2d5e8f7a30"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
Lava = "3a680b1f-cb25-4bee-9cf7-bc880b76dc8c"
PrecompileTools = "aea7be01-6a6a-4083-8856-8a6e6704d82a"

# Lava is not in the General registry, so a fresh clone cannot resolve without
# being told where it lives. `[sources]` (Julia 1.11+) is how: `Pkg.instantiate`
# fetches it from the URL below instead of failing with "package not found".
[sources]
Lava = {{url = "https://github.com/SimonDanisch/Lava.jl", rev = "sd/nvidia"}}

# Without this `Pkg.test` gives the suite an environment with no `Test`, and
# every generated runtests.jl dies on `using Test` before it asserts anything.
[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test"]
"""

MODULE = '''\
"""
{title} — {feature}.

{summary}

**Not ported yet.** This package is the place the port lands, committed ahead of
the work so the graph path, the asset lookup and the workload guard are settled
and everything after this is model code. What exists: [`assetdir`](@ref) resolves
the export, [`{lower}graph`](@ref) loads it if it is there, and precompilation is
inert until it is. What does not: the workload body, and whatever ops the export
turns out to need.

Upstream: {upstream}
License: **{license}**

Ops `DNNKernels` does not have yet:
{oplist}

See `models-to-port.md` for the state of this one, and `tools/export_{name}.py`
for the export that feeds it.
"""
module {package}

using Lava, DNNKernels, KernelAbstractions
using Lava: @setup_workload, @compile_workload
using DNNKernels: loadgraph, execute!, readsafetensors

export {lower}graph, {lower}weights, assetdir

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

Throws. {title} is **not ported yet**, so there is no artifact to read from and
nothing on disk that a user of this package would have.

Porting it means, in order: export it with `uv run tools/export_{name}.py`, bind
the result with `julia --project=. tools/make_artifacts.jl {graphdir}`, and
replace this definition with `@artifact_str("{graphdir}")`. Assets come from the
artifact and from nowhere else — see `DNNKernels/src/assets.jl`.
"""
assetdir() = error(
    "{package}: {title} is not ported yet, so no artifact is bound. " *
    "Export it with `uv run tools/export_{name}.py`, bind it with " *
    "`julia --project=. tools/make_artifacts.jl {graphdir}`, then set " *
    "`assetdir() = @artifact_str(\\"{graphdir}\\")`.")

"""
    {lower}graph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function {lower}graph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "{name}.json")
    isfile(p) || throw(ArgumentError(
        "{title} graph not found at $p. Generate it with " *
        "`uv run tools/export_{name}.py`, or set {envvar}."))
    return loadgraph(p)
end

"""
    {lower}weights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function {lower}weights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("{title} weights not found at $p"))
    return readsafetensors(p)
end

"""
    ready(; dir = assetdir()) -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
ready() = false        # not ported: see `assetdir`

function __init__()
    # Read the entries the workload froze. Recording stays off: a session that
    # hits a kernel the workload missed should compile it and carry on, not
    # quietly rewrite the frozen set under a version it was not built for.
    Lava.use_frozen_kernels(KERNELS_VERSION)
    return nothing
end

# ---------------------------------------------------------------- the workload
#
# Guarded on the assets and on a working device: precompilation must not fail on
# a machine without either, it should just produce a package with nothing cached.
#
# TODO(port): drive the real call here once the graph runs. The measurement that
# matters is `Lava.no_pipeline_compilation` reporting **0 refusals** on a *fresh*
# process — a workload
# that runs a different path than the editor does leaves the editor compiling on
# first use, which is the entire cost this package exists to remove. SAM2Runner
# learned that the expensive way: its `runsam2` workload still left 45 s on the
# first click because the editor goes through a closure `runsam2` never touches.
#
# NOT `frozen_stats().misses == 0`, which reads stronger than it is: it cannot
# distinguish the frozen cache working from the driver's own shader cache having
# served everything, and its miss report identifies modules by the *sampling*
# hash, so two modules differing in one byte count as one (`STATUS.md`,
# cross-project). `no_pipeline_compilation` empties `PIPELINE_CACHE` first and
# refuses anything needing a compile. Pair it with a negative control whose
# kernel body is novel per RUN — a `Val{{K}}` with `K` from `RandomDevice` — or a
# green means nothing; verified firing here at refused = 1.
@setup_workload begin
    if ready()
        try
            backend = LavaBackend()
            graph = {lower}graph()
            weights = {lower}weights()
            @compile_workload KERNELS_VERSION begin
                # Inputs: {inputs}
                nothing
            end
        catch err
            @warn "{package}: workload skipped; first use will compile" exception = err
        end
    else
        @info "{package}: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
'''

TEST = '''\
"""
Until the port runs, this asserts the two things that are true now and must stay
true: the package loads on a machine with no assets, and the asset lookup names
a real place rather than throwing something unreadable.

The latency test that matters — `frozen_stats().misses == 0` in a fresh process
— belongs here once the workload drives the real call. See SAM2Runner/test for
the shape it should take; it has to run in a subprocess because Julia's
compile-time counter is per-process.
"""

using Test, {package}

@testset "{package}" begin
    # No `assetdir()`. That is internal: it names where the artifact happens to
    # put things, and a test that calls it has to know the layout, so a re-export
    # that moves a file breaks a test that never knew it depended on that. Ask
    # for the graph and the weights; `dir` is a config keyword on those, whose
    # default is the artifact.
    if {package}.ready()
        @info "{title}: export present"
        g = {package}.{lower}graph()
        @test g !== nothing
        w = {package}.{lower}weights()
        @test !isempty(w)
    else
        @info "{title}: no export; run tools/export_{name}.py"
        # The error has to name the path — a caller who has not run the exporter
        # should be told where to put it, not handed a MethodError later.
        @test_throws ArgumentError {package}.{lower}graph()
    end
end
'''


def write(path, text, force):
    if path.exists() and not force:
        print(f"  skip {path.relative_to(ROOT)} (exists)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"  write {path.relative_to(ROOT)}")
    return True


def main(force):
    for m in MODELS:
        print(m.package)
        pkg = JV / m.package
        oplist = "\n".join(f"  * {o}" for o in m.newops) or "  * none — the runtime already covers it"
        fields = dict(
            package=m.package, uuid=m.uuid, title=m.title, feature=m.feature,
            summary=m.summary, upstream=m.upstream, license=m.license,
            name=m.name, lower=m.name, inputs=m.inputs, oplist=oplist,
            graphdir=m.graphdir or m.name,
            envvar=f"JULIA_{m.name.upper()}_ASSETS",
        )
        write(pkg / "Project.toml", PROJECT.format(**fields), force)
        write(pkg / "src" / f"{m.package}.jl", MODULE.format(**fields), force)
        write(pkg / "test" / "runtests.jl", TEST.format(**fields), force)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true")
    main(p.parse_args().force)
