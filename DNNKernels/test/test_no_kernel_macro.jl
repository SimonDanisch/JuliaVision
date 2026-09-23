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

`@kernel` is at zero and stays at zero, and so is the eager op library —
`runop!` and `execute!` are gone, and the assertion is now that the names do not
come back.

What is NOT zero is `KernelAbstractions.allocate`, 16 calls. The load-time
quantisers and `toback` have gone to `Mantle.Buffer`; what is left is per-frame
scratch, a few output fallbacks, and Mantle's own — seven, of which three ARE
the `KA.allocate` implementation and the KI allocation hook.
Until they move, the useful property is that the count only ever goes DOWN. A
number that drifts up is a new allocation path that Mantle does not see, which
is the thing the migration exists to remove.

Update the numbers when you delete something. Raising one is a decision, and
this test is where it has to be made on purpose.
"""

using Test, Mantle, DNNKernels

# Whatever is LOADED, which is the only tree a green result says anything
# about. Under a project that pins Mantle to a git rev this is the depot copy
# and not `dev/Mantle`, so every assertion below reports the directory it read:
# a pass against a stale checkout has to be visible as one.
# …/JuliaVision: every package in this tree, walked by the two testsets that
# scan the whole repository rather than just the DNN library.
const RUNNERS = dirname(dirname(@__DIR__))

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
    # Two are not zero. They are named, with counts that may only fall, rather
    # than excluded by a pattern that would also hide a regression.
    #
    #   * `GPUFiltering` (25) — and this one is a DECISION, not a leftover. Its
    #     own docstring is "backend-agnostic … works on any KA backend —
    #     `Matrix` (CPU), `LavaArray` (Vulkan), `CuArray`", its tests run on
    #     `KA.CPU()`, and `KA.CPU` implements no `KI.kernel_function`. Porting it
    #     would delete the CPU backend it exists to support. The `@kernel`s ARE
    #     the portability. It is not on the DNN path: nothing here dispatches
    #     through it, and the runners that use it take `tofloat`/`topixel` and
    #     `resizeplanar!`, which are host-callable.
    #
    # `BonsaiRunner` was the second entry here, at 30, and is now at zero along
    # with everything else. It is `develop`ed into the root project, so this
    # counts a tree that also loads and runs.
    #
    # The same reasoning keeps ~130 `@kernel` tests in Mantle and
    # `test_index_recovery.jl` here: the KernelAbstractions path is live for
    # Raycore, Hikari, GPUFiltering and GPUArrays, all of which go through the
    # same Lava. "No `@kernel`" is a rule about the DNN library, and the DNN
    # library is at zero.
    remaining = Dict("GPUFiltering" => 25)
    defn = r"@kernel(\s+\w+\s*=\s*\w+)*\s+function"
    for pkg in sort(readdir(RUNNERS))
        dir = joinpath(RUNNERS, pkg, "src")
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

    # And every runner too, on the same argument as the `@kernel` walk above:
    # the two go together. A `@kernel` is only reachable through the
    # constructor protocol, so a package at zero for one and not the other is
    # a launch whose definition has already moved. GPUFiltering keeps both for
    # the reason given above; it is the only package that does.
    ctors = Dict("GPUFiltering" => 16)
    for pkg in sort(readdir(RUNNERS))
        dir = joinpath(RUNNERS, pkg, "src")
        isdir(dir) || continue
        hits = [(f, i) for (f, i, L) in sourcelines(dir) if iscode(L) && occursin(ctor, L)]
        if haskey(ctors, pkg)
            @test (pkg => length(hits)) == (pkg => min(length(hits), ctors[pkg]))
        else
            @test (pkg => hits) == (pkg => Tuple{String,Int}[])
        end
    end

    # These are the ones still to go. The assertion is `<=`, so deleting is free
    # and adding is a conversation.
    alloc = r"(KernelAbstractions|KA)\.allocate"
    counts = Dict(pkg => count(((f, i, L),) -> iscode(L) && occursin(alloc, L),
                               sourcelines(dir))
                  for (pkg, dir) in SRC)
    # Tightened whenever they drop, which is the only way a ratchet stays one:
    # Mantle 11 -> 7 over the `@kernel` migration, DNNKernels 32 -> 23 when
    # MatAnyone's memory bank and tracking state moved onto Mantle buffers, and
    # 23 -> 12 when load-time quantisation became declared plans over
    # `Mantle.Buffer`, and 12 -> 9 when `toback` did — that is every weight
    # upload in the tree.
    @test counts["Mantle"] <= 7
    @test counts["DNNKernels"] <= 9
end

"""
Named `Val{:op}` arms of `f`, from the method table rather than the source.

The catch-all `::Val{A} where A` is not a named arm and drops out here, because
its parameter is a `TypeVar` and not a `Symbol`.
"""
function namedarms(f)
    out = Set{Symbol}()
    for m in methods(f), p in Base.unwrap_unionall(m.sig).parameters
        p isa DataType && p <: Val && p.parameters[1] isa Symbol && push!(out, p.parameters[1])
    end
    return out
end

@testset "there is no second implementation of an op" begin
    # This was a ratchet on a number — `runop!` arms `<= 103`, falling — because
    # the eager op library could not go in one step. It went: `execute!`, the 86
    # `runop!` arms in `ops.jl`, the one in `fusepass.jl` and the machinery under
    # them are deleted, so the assertion is now that the NAMES are gone rather
    # than that a count has not grown.
    #
    # Every op had two implementations, one launching immediately and one
    # declaring into a Mantle graph, and a bug fixed in one was not fixed in the
    # other. `test_atenarg.jl` is the case that cost the most: `runop!`'s gelu
    # read `arg1` where every graph in the tree writes `approximate`, so two
    # models silently ran the exact gelu where the reference ran the
    # approximation.
    @test !isdefined(DNNKernels, :runop!)
    @test !isdefined(DNNKernels, :execute!)

    # And the declared side does not shrink. Deleting an `emitop!` arm is how
    # this would be made to pass the wrong way.
    @test length(namedarms(DNNKernels.emitop!)) >= 97
end
