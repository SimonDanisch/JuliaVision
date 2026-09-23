"""
Nothing in this tree defines a `@kernel`, and the eager surface only shrinks.

A kernel here is a plain function over `KernelInterface`'s intrinsics, handed to
`Mantle.dispatch!` to declare or `KI.Kernel(backend, f)` to launch. That is the
whole of the rule, and it is a source-level one: a `@kernel` compiles and runs,
so nothing else in the suite would notice one coming back.

## Why this is a ratchet and not just an assertion

`@kernel` is at zero and stays at zero. The other two counts are NOT zero, and
saying so is the point of writing them down:

  * `KernelAbstractions.allocate` — 43 calls. Some are genuinely outside any
    frame (MatAnyone's memory bank, the load-time quantisers) and want Mantle
    persistent buffers; the rest belong to the eager op library.
  * the eager op library itself — the `runop!` arms that launch immediately
    instead of declaring into a graph.

Both are slated to go, and until they do the useful property is that they only
ever go DOWN. A number that drifts up is a new eager path, which is the thing
the migration exists to remove.

Update the numbers when you delete something. Raising one is a decision, and
this test is where it has to be made on purpose.
"""

using Test, Mantle, DNNKernels

# Whatever is LOADED, which is the only tree a green result says anything
# about. Under a project that pins Mantle to a git rev this is the depot copy
# and not `dev/Mantle`, so every assertion below reports the directory it read:
# a pass against a stale checkout has to be visible as one.
const SRC = [
    "Mantle" => joinpath(dirname(dirname(pathof(Mantle))), "src"),
    "DNNKernels" => joinpath(dirname(dirname(pathof(DNNKernels))), "src"),
]

"""
Every CODE line under `dir`, with its file and line number.

Docstring bodies are dropped along with `#` comments, because this tree
documents the migration at length and in `\`@kernel\`` and `f(backend)(…)`
terms — `kalaunch.jl` explains the constructor protocol it is deleting. Prose
about a pattern is not the pattern.

Counting `\"\"\"` per line and toggling on an odd count is not a Julia parser
— an escaped or nested one would fool it. None occur in either tree, and the
failure mode is a skipped line, not a missed `@kernel` on a line of its own.
"""
function sourcelines(dir)
    out = Tuple{String,Int,String}[]
    for (root, _, files) in walkdir(dir), f in files
        endswith(f, ".jl") || continue
        p = joinpath(root, f)
        indoc = false
        for (i, L) in enumerate(eachline(p))
            n = length(findall("\"\"\"", L))
            if indoc
                indoc = iseven(n)
                continue
            elseif isodd(n)
                indoc = true
                continue
            end
            push!(out, (relpath(p, dir), i, L))
        end
    end
    return out
end

iscode(L) = !occursin(r"^\s*#", L)

@testset "no @kernel is defined anywhere" begin
    for (pkg, dir) in SRC
        bad = [(f, i) for (f, i, L) in sourcelines(dir)
               if iscode(L) && occursin(r"@kernel(\s+\w+\s*=\s*\w+)*\s+function", L)]
        @test (dir => bad) == (dir => Tuple{String,Int}[])
    end
end

@testset "the eager surface only shrinks" begin
    # `f(backend)(args...)`: the KernelAbstractions constructor protocol, where
    # the backend builds the kernel object and the object is then called. Zero in
    # both packages — a kernel is launched as `KI.Kernel(backend, f)(args...)`
    # now, which this does not match, because it wants `backend` as the SOLE
    # argument. No exemption for `KI.Kernel`, then: it never matched.
    ctor = r"[A-Za-z_][A-Za-z0-9_!]*(\[[^]]*\])?\((ctx\.backend|backend)\)\("
    for (pkg, dir) in SRC
        bad = [(f, i) for (f, i, L) in sourcelines(dir)
               if iscode(L) && occursin(ctor, L)]
        @test (dir => bad) == (dir => Tuple{String,Int}[])
    end

    # These are the ones still to go. The assertion is `<=`, so deleting is free
    # and adding is a conversation.
    alloc = r"(KernelAbstractions|KA)\.allocate"
    counts = Dict(pkg => count(((f, i, L),) -> iscode(L) && occursin(alloc, L),
                               sourcelines(dir))
                  for (pkg, dir) in SRC)
    @test counts["Mantle"] <= 11
    @test counts["DNNKernels"] <= 32
end
