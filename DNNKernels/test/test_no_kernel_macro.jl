"""
Nothing in this tree defines a `@kernel`, and the eager surface only shrinks.

A kernel here is a plain function over `KernelInterface`'s intrinsics, handed to
`Mantle.dispatch!` to declare or `KI.Kernel(backend, f)` to launch. That is the
whole of the rule, and it is a source-level one: a `@kernel` compiles and runs,
so nothing else in the suite would notice one coming back.

**`src/` and not the tests**, deliberately. `@index(Global, NTuple)` recovery
can only be tested through a `@kernel`, because that is the only interface that
has it — see `test_index_recovery.jl`. The KernelAbstractions path is still live
for Raycore, Hikari, GPUFiltering and GPUArrays, which all dispatch through the
same Lava, so a test that covers it is coverage and not a leftover.

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

@testset "every package in this tree is macro-free, or is named here" begin
    # The two trees above are the DNN library and its GPU library. This walks
    # the WHOLE repository, because a runner is part of the code base too and
    # four of them had a `@kernel` after the migration said there were none:
    # RIFE's frame pack and unpack, Kokoro's alignment gather, Hunyuan3D's grid
    # queries. Nothing else scanned them.
    #
    # Two are not done. They are named, with counts that may only fall, rather
    # than excluded by a pattern that would also hide a regression:
    #
    #   * `GPUFiltering` — an image-filtering package, not part of the DNN path.
    #   * `BonsaiRunner` — not in the root project and unported.
    remaining = Dict("GPUFiltering" => 25, "BonsaiRunner" => 30)
    root = dirname(dirname(@__DIR__))          # …/JuliaVision
    defn = r"@kernel(\s+\w+\s*=\s*\w+)*\s+function"
    for pkg in sort(readdir(root))
        dir = joinpath(root, pkg, "src")
        isdir(dir) || continue
        hits = [(f, i) for (f, i, L) in sourcelines(dir) if iscode(L) && occursin(defn, L)]
        if haskey(remaining, pkg)
            # `<=`, so porting one is free and adding one is a conversation.
            @test (pkg => length(hits)) == (pkg => min(length(hits), remaining[pkg]))
        else
            @test (pkg => hits) == (pkg => Tuple{String,Int}[])
        end
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
    # Tightened whenever they drop, which is the only way a ratchet stays one:
    # Mantle 11 -> 7 over the `@kernel` migration, DNNKernels 32 -> 23 when
    # MatAnyone's memory bank and tracking state moved onto Mantle buffers.
    @test counts["Mantle"] <= 7
    @test counts["DNNKernels"] <= 23
end

"""
Named `Val{:op}` arms of `f`, from the method table rather than the source.

`ops.jl` generates a block of `runop!`s with `@eval` over a table, so counting
`function runop!` in the text undercounts by however long that table is. The
catch-all `::Val{A} where A` is not a named arm and drops out here, because its
parameter is a `TypeVar` and not a `Symbol`.
"""
function namedarms(f)
    out = Set{Symbol}()
    for m in methods(f), p in Base.unwrap_unionall(m.sig).parameters
        p isa DataType && p <: Val && p.parameters[1] isa Symbol && push!(out, p.parameters[1])
    end
    return out
end

@testset "the eager op library only shrinks" begin
    # The duplication the migration exists to remove: `runop!` and `emitop!` are
    # two implementations of the same ops, one launching immediately and one
    # declaring into a Mantle graph. `<=`, so deleting an eager arm is free.
    #
    # It cannot go to zero in one step and the number is not the goal — what
    # matters is that it never goes UP, because a new eager arm is a new second
    # implementation. `test_declared_coverage.jl` holds the other side: every op
    # any exported graph actually contains is already declarable, so the eager
    # arms are deletable rather than load-bearing.
    @test length(namedarms(DNNKernels.runop!)) <= 103

    # And the declared side does not shrink. Deleting an `emitop!` arm is how
    # this would be made to pass the wrong way.
    @test length(namedarms(DNNKernels.emitop!)) >= 97
end
