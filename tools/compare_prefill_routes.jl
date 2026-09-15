# Attribute a prefill bucket's deviation to the attention route. The same
# post-fusion bucket graph runs twice -- once with the staged two-GEMM masked
# prefill, once with `staged = false` so the same op falls back to cooperative
# flash -- and both are compared against the unfused reference graph. Without
# this split, a KV-cache error at one bucket cannot be told apart from the
# deviation the direct int8 GEMM already carries at every bucket.
using HorizonRunner, DNNKernels, Mantle, KernelAbstractions, Statistics, Test

function stagedoff(g::DNNKernels.Graph)
    ops = map(g.ops) do o
        o.aten == "fused.maskedattention" || return o
        DNNKernels.Op(o.id, o.aten, o.ins, o.out,
                      merge(o.attrs, Dict{String,Any}("staged" => false)))
    end
    DNNKernels.Graph(g.name, g.symbols, g.inputs, g.outputs, copy(g.buffers),
                     g.order, ops, g.fusion)
end

# The logits and both caches, as host arrays, for one graph and one route.
function routeoutputs(model, name, args, kv, k, v)
    fill!(k, 0); fill!(v, 0)
    map(Array, DNNKernels.call(model, name, args...; dims=(; kv)))
end

function comparelabels(got, expected, label, tokens)
    for (i, (a, b)) in enumerate(zip(got, expected))
        err = maximum(abs, Float32.(a) .- Float32.(b))
        rel = err / max(maximum(abs, Float32.(b)), eps(Float32))
        println("ROUTE $label tokens=$tokens output=$i size=$(size(a)) ",
                "finite=$(all(isfinite, a)) max_abs=$err relative_max=$rel")
        @test all(isfinite, a)
    end
    @test argmax(first(got)) == argmax(first(expected))
    flush(stdout)
end

# The staged-off model exists for one comparison only. Its recorded plans hold
# command buffers and descriptor sets, so hand them back before the next bucket
# records its own; the pool keeps the reservation either way and reuses it.
function releasemodel!(model)
    for (key, value) in collect(model.scratch)
        value isa DNNKernels.RecordedPlan || continue
        Mantle.free!(value.plan)
        delete!(model.scratch, key)
    end
    empty!(model.scratch)
    GC.gc(true)
    nothing
end

function compare_prefill_routes(m, reference_graphs; tokens = 128, repeats = 7, warmup = 3)
    name = "horizon32b_prefill_bucket_$tokens"
    base = m.model
    graph = base.graphs[name]
    kv = HorizonRunner.attentionbucket(m, tokens)
    k, v = get!(() -> HorizonRunner.newcache(m), base.scratch, (:horizon_generation_cache,))
    slots = collect(Int64, 0:tokens-1)
    mask = HorizonRunner.maskbuffer(m, tokens; kv)
    HorizonRunner.causalmask!(mask, slots)
    args = (DNNKernels.toback(m.backend, reshape(mod.(slots, 1000) .+ 1, tokens, 1)),
            k, v, DNNKernels.toback(m.backend, slots),
            DNNKernels.toback(m.backend, mask),
            DNNKernels.toback(m.backend, Int64[tokens-1]))
    reference = DNNKernels.Model(Dict(name => reference_graphs[name]), base.weights,
        base.backend, base.memevery, base.memframes, base.topk; record=false)
    flashmodel = DNNKernels.Model(Dict(name => stagedoff(graph)), base.weights,
        base.backend, base.memevery, base.memframes, base.topk;
        record=base.record, record_maxpasses=base.record_maxpasses)
    @testset "prefill route comparison $tokens" begin
        expected = routeoutputs(reference, name, args, kv, k, v)
        reference = nothing
        for (label, model) in (("staged", HorizonRunner.bucketmodel(m, name)),
                               ("flash", flashmodel))
            got = routeoutputs(model, name, args, kv, k, v)
            comparelabels(got, expected, label, tokens)
            repeated = routeoutputs(model, name, args, kv, k, v)
            @test repeated == got
            got = repeated = nothing
            # Replay only: downloading the two 537 MiB caches would dominate.
            host = Matrix{Float16}(undef, m.vocab, 1)
            replay() = begin
                logits = first(DNNKernels.call(model, name, args...; dims=(; kv)))
                copyto!(host, reshape(logits, m.vocab, 1))
                nothing
            end
            for _ in 1:warmup; replay(); end
            KernelAbstractions.synchronize(m.backend)
            samples = [@elapsed(replay()) for _ in 1:repeats]
            println("ROUTE_TIME $label tokens=$tokens median_s=$(median(samples)) samples=$samples")
            flush(stdout)
        end
    end
    releasemodel!(flashmodel)
    nothing
end
