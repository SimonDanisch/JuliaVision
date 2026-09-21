"""
`foldoutcasts` on SAM 2's own encoder graph.

Against the real graph rather than a synthetic one, because every property this
pass has to get right is a property of how *that* graph is shaped: the casts read
their input through a `getitem` of a multi-output tuple or through a reshape, the
reader count only comes out right if views are transparent, and a layer norm's
dtype lives in an attribute rather than in the buffer's own field. A hand-built
two-op graph exercises none of it.

The pass is only allowed to be free, so the assertions are about invariants —
nothing widens, nothing but casts is dropped, and every retyped buffer had
exactly one reader — rather than about a count that moves whenever the model is
re-exported.
"""

using Test, DNNKernels
include("fixtures.jl")
const DK = DNNKernels

# Ask the runner for the GRAPH, not for a directory to build paths in. The
# artifact is bound here too (content-addressed, same directory); the layout is not this
# file's business, and a re-export that moves one must not break this test.
#
# Walking up the filesystem for a `gen/` tree is what makes BOTH testsets here
# report `Total 0` and read as green when the walk misses: 4172 assertions that
# were not running.
const HAVE_SAM2 = true   # bound in DNNKernels/Artifacts.toml

@testset "foldoutcasts" begin
    if !HAVE_SAM2
        @info "the sam2-large artifact is not installed; skipping"
        # Recorded, so an absent graph shows up as a skip rather than as a
        # testset with zero assertions, which is indistinguishable from a pass.
        @test_skip HAVE_SAM2
    else
        g = Fixtures.sam2("sam2_encoder")
        g2, n = DK.foldoutcasts(g)

        @testset "it folds, and only casts" begin
            @test n > 0
            dropped = setdiff(Set(o.id for o in g.ops), Set(o.id for o in g2.ops))
            @test length(dropped) == n
            @test all(o.aten == "_to_copy.default" for o in g.ops if o.id in dropped)
            @test length(g2.ops) == length(g.ops) - n
            # The op order has to lose exactly the same ids, or the planner walks
            # a buffer whose op is gone.
            @test length(g2.order) <= length(g.order)
        end

        @testset "no buffer widens" begin
            # The pass may only narrow. A widened buffer would mean a producer
            # asked to write MORE bytes than the graph declared, which is the one
            # way this could cost time rather than save it.
            for (id, b) in g2.buffers
                old = get(g.buffers, id, nothing)
                old === nothing && continue
                @test sizeof(b.dtype) <= sizeof(old.dtype)
            end
        end

        @testset "every retyped producer had exactly one reader" begin
            # Counted the way the pass counts: views transparent, ops and graph
            # outputs only. This is the invariant that makes the fold safe — if
            # anything else read the wide value it would now see a narrow one.
            reads = Dict{String,Int}()
            bump!(id) = (reads[id] = get(reads, id, 0) + 1)
            for op in g.ops, i in op.ins
                bump!(last(DK.viewchain(g, i)))
            end
            for o in g.outputs
                bump!(last(DK.viewchain(g, o)))
            end
            changed = [id for (id, b) in g2.buffers
                       if haskey(g.buffers, id) && b.dtype !== g.buffers[id].dtype]
            @test !isempty(changed)
            for id in changed
                root = last(DK.viewchain(g, id))
                @test get(reads, root, 0) == 1
            end
        end

        @testset "the switch turns it off" begin
            # A keyword, so there is no module state to save and restore — which
            # a failing `@test` inside the old `try` would have skipped, leaving
            # the fold off for every test after it.
            g3, k = DK.foldoutcasts(g; enabled = false)
            @test k == 0
            @test length(g3.ops) == length(g.ops)
        end

        @testset "it is idempotent" begin
            # Running it twice must find nothing new: the casts it folded are
            # gone, and the buffers it retyped now agree with their readers.
            _, k = DK.foldoutcasts(g2)
            @test k == 0
        end
    end
end

# ── What a fold puts on an op, and who reads it ──────────────────────────────
#
# `foldrelu` deletes an activation and records it as `attrs["act"]` on the op
# BEFORE it; `fuseops` does the same with `attrs["epilogue"]`. That is only a
# fold if the emit for that op applies it — and for `add.Tensor` it did not, so
# every relu folded into a residual add was silently dropped. The result was
# finite, the right shape and dtype, and wrong only where the sum was negative.
#
# `foldrelu`'s own note says which adds those are: "a residual block ends
# `add(conv, skip) -> relu`, so the relus on the largest feature maps follow an
# *add*", 36 of them on MatAnyone. Nothing in this suite noticed;
# `tools/two_route_parity.jl` did, at `add_94`, interpreted min 0.0 against
# declared min -2.68.
#
# So the two sides are stated once each and checked against each other. Adding a
# producer to `foldrelu` now fails here until its emit reads the attribute.

# ── Ops a fusion pass CREATES, and whether the emit has one ──────────────────
#
# A coverage sweep over exported graphs cannot see these: they are not in any
# export, a pass puts them there. `fused.sdpa` is the one that cost a day —
# `fuseattention` builds it from a bmm/softmax/bmm chain, the driver runs that
# pass by default, and DepthAnything then failed with "no emit method for
# `fused.sdpa`" after the coverage table said its graph was fully declared.
#
# So the created set is stated here, beside which of them are ported, and both
# are checked. Teaching a pass to emit a new `fused.*` op fails this test until
# its `emitop!` exists, which is the one thing no graph can tell you.

"""
Every aten a pass in this package can put into a graph, and which pass.

Not just the `fused.*` ones: `foldcacheupdate` rewrites the `cat` that rebuilds
a KV cache into an `alias.default` OP, which no export contains either and which
cost a second round of the same surprise after `fused.sdpa`.
"""
const FUSION_CREATES = ("fused.elementwise" => "fusepass.jl / fusemaskedattention.jl",
                        "fused.sdpa"        => "fuseattention.jl",
                        "fused.groupedrms"  => "fusegroupedrms.jl",
                        "fused.maskedattention" => "fusemaskedattention.jl",
                        "fused.rope"        => "fuserope.jl",
                        "fused.pairrope"    => "fuserope.jl",
                        "fused.ropecache"   => "fuserope.jl",
                        "fused.swiglu"      => "fuseswiglu.jl",
                        "alias.default"     => "foldcache.jl")

"""
The created ops with no `emitop!` yet, and what each is for.

A graph that trips one of these is refused by name, which is right — what was
wrong was having no way to know which they are without running a model that
happens to hit one.

  * `fused.maskedattention` — the prefill path, whose plan
    (`maskedprefill.jl`) has its own launch shape to split.
"""
const FUSION_UNPORTED = ("fused.maskedattention",)

"""
Attributes a pass SETS on an existing op, and the emit that has to read each.

`attrs["act"]` and `attrs["epilogue"]` are checked below, against the graphs.
`attrs["inplace"]` cannot be: `foldcacheupdate` sets it and no export carries
it, so only the pass and the emit know it exists. Getting it wrong is silent in
the worst way — a KV cache that is never written, with plausible numbers
everywhere the graph looks.
"""
const PASS_SETS = ("act"      => "foldrelu.jl",
                   "epilogue" => "fusepass.jl",
                   "inplace"  => "foldcache.jl",
                   # `fusegroupedrms` sets it when the export rounds between
                   # the normalisation and the gain, and the kernel owes that
                   # rounding; `fusepairrope` sets `P` because the pair count
                   # is not recoverable from the operand it reads.
                   "midround" => "fusegroupedrms.jl",
                   "P"        => "fuserope.jl")

@testset "every attribute a pass sets is read by some emit" begin
    src = read(joinpath(@__DIR__, "..", "src", "emit.jl"), String)
    for (attr, where) in PASS_SETS
        # The pass still sets it...
        @test occursin("\"$attr\"", read(joinpath(@__DIR__, "..", "src", where), String))
        # ...and `emit.jl` still reads it. A `get(op.attrs, "inplace", …)` that
        # went away would leave the pass marking ops nothing honours.
        @test occursin("\"$attr\"", src)
    end
end

@testset "every op a fusion pass creates is accounted for" begin
    ported = Set{String}()
    for m in methods(DK.emitop!)
        p = Base.unwrap_unionall(m.sig).parameters
        length(p) >= 4 || continue
        T = p[4]
        T isa DataType && T <: Val || continue
        isempty(T.parameters) && continue
        T.parameters[1] isa Symbol || continue
        push!(ported, String(T.parameters[1]))
    end
    # The names are real: a typo here would make this test vacuous, and a pass
    # renaming its op would leave a dead entry that still "passed".
    for (aten, where) in FUSION_CREATES
        @test any(f -> occursin("\"$aten\"", read(joinpath(@__DIR__, "..", "src", f), String)),
                  split(where, " / "))
    end
    # And each is either emitted or recorded as not.
    for (aten, _) in FUSION_CREATES
        @test (aten in ported) == !(aten in FUSION_UNPORTED)
    end
    @test issubset(FUSION_UNPORTED, first.(FUSION_CREATES))
end

"""ATen ops whose `emitop!` applies `attrs["act"]`."""
const READS_ACT = ("convolution.default", "add.Tensor", "addmm.default",
                   "mul.Tensor", "div.Tensor", "sub.Tensor", "eq.Scalar",
                   "ge.Tensor", "ge.Scalar", "le.Tensor", "le.Scalar",
                   "bitwise_and.Tensor", "logical_and.default")

"""ATen ops whose `emitop!` applies `attrs["epilogue"]`."""
const READS_EPILOGUE = ("addmm.default",)

@testset "every folded activation lands on an op that applies it" begin
    graphs = Any[]
    HAVE_SAM2 && for n in ("sam2_encoder", "sam2_decoder")
        push!(graphs, (n, Fixtures.sam2(n)))
    end
    for n in Fixtures.matanyonenames()
        push!(graphs, (n, Fixtures.matanyone(n)))
    end
    @test !isempty(graphs)
    for (name, g) in graphs
        # Through the two passes that set them, in the driver's order.
        g1, _ = DK.foldrelu(g)
        g2, _ = DK.fuseops(g1)
        for op in g2.ops
            haskey(op.attrs, "act") &&
                @test op.aten in READS_ACT
            haskey(op.attrs, "epilogue") &&
                @test op.aten in READS_EPILOGUE
        end
    end
end

@testset "foldrelu folds gelu into addmm" begin
    if !HAVE_SAM2
        @info "the sam2-large artifact is not installed; skipping"
        # Recorded, so an absent graph shows up as a skip rather than as a
        # testset with zero assertions, which is indistinguishable from a pass.
        @test_skip HAVE_SAM2
    else
        g = Fixtures.sam2("sam2_encoder")
        g2, n = DK.foldrelu(g)

        # The reason this needed the whole view chain rather than
        # `resolvealias`: the graph reshapes the `addmm` result, so the shape is
        # `gelu <- view.default <- addmm` and an alias-only resolve finds no
        # producer at all. It folded 0 of 48 until that was fixed.
        @test n >= 48
        @test count(o -> o.aten == "gelu.default", g2.ops) == 0
        @test count(o -> get(o.attrs, "act", "") == "gelu", g2.ops) == 48
        @test all(o -> o.aten == "addmm.default",
                  (o for o in g2.ops if get(o.attrs, "act", "") == "gelu"))

        @testset "the alias keeps the gelu's own shape" begin
            # Aliasing onto the `addmm` buffer would be a rank error waiting to
            # happen: `addmm_2` is `[65536, 576]` and the gelu it feeds is
            # `[1, 256, 256, 576]`. The alias has to target the *view*.
            for o in g.ops
                o.aten == "gelu.default" || continue
                b = g2.buffers[o.out]
                @test b.kind === :view
                @test b.viewop == "alias.default"
                @test g2.buffers[b.of].shape == g.buffers[o.out].shape
            end
        end

        @testset "the tanh approximation is a different function and does not fold" begin
            # `gelu(approximate="tanh")` differs from the exact form by ~1e-3.
            # Folding it in as if it were the same would be a silent wrong answer.
            gl = first(o for o in g.ops if o.aten == "gelu.default")
            attrs = Dict{String,Any}(gl.attrs); attrs["arg1"] = "tanh"
            ops = [o.id == gl.id ? DK.Op(o.id, o.aten, o.ins, o.out, attrs) : o
                   for o in g.ops]
            gt = DK.Graph(g.name, g.symbols, g.inputs, g.outputs, g.buffers,
                          g.order, ops, g.fusion)
            _, nt = DK.foldrelu(gt)
            @test nt == n - 1
        end
    end
end
