# Where a prefill bucket's time goes, attributed in situ. Uses `Diagnostics`'
# `opdouble`: one aten family runs twice in the recorded plan and the replay
# grows by exactly that family's device time, which a microbenchmark of the same
# kernel cannot tell you -- it has neither the surrounding traffic nor the real
# operands. `optimes` would answer the same question by synchronising around
# every op, and buries a 1100-op graph under its own barriers.
using HorizonRunner, DNNKernels, Mantle, KernelAbstractions, Statistics

function prefillargs(m, tokens)
    kv = HorizonRunner.attentionbucket(m, tokens)
    k, v = get!(() -> HorizonRunner.newcache(m), m.model.scratch,
                (:horizon_generation_cache,))
    slots = collect(Int64, 0:tokens-1)
    mask = HorizonRunner.maskbuffer(m, tokens; kv)
    HorizonRunner.causalmask!(mask, slots)
    (DNNKernels.toback(m.backend, reshape(mod.(slots, 1000) .+ 1, tokens, 1)), k, v,
     DNNKernels.toback(m.backend, slots), DNNKernels.toback(m.backend, mask),
     DNNKernels.toback(m.backend, Int64[tokens-1])), kv
end

function attribute_prefill_ops(m, tokens; atens = nothing, repeats = 7, warmup = 3)
    name = "horizon32b_prefill_bucket_$tokens"
    base = m.model
    graph = base.graphs[name]
    args, kv = prefillargs(m, tokens)
    counts = Dict{String,Int}()
    for o in graph.ops; counts[o.aten] = get(counts, o.aten, 0) + 1; end
    atens === nothing && (atens = first.(sort(collect(counts), by = x -> -x[2])))
    host = Matrix{Float16}(undef, m.vocab, 1)
    function timeone(double)
        model = DNNKernels.Model(Dict(name => graph), base.weights, base.backend,
            base.memevery, base.memframes, base.topk;
            record = base.record, record_maxpasses = base.record_maxpasses)
        model.diag.opdouble = double
        replay() = begin
            logits = first(DNNKernels.call(model, name, args...; dims = (; kv)))
            copyto!(host, reshape(logits, m.vocab, 1))
            nothing
        end
        for _ in 1:warmup; replay(); end
        KernelAbstractions.synchronize(m.backend)
        samples = [@elapsed(replay()) for _ in 1:repeats]
        for (key, value) in collect(model.scratch)
            value isa DNNKernels.RecordedPlan || continue
            Mantle.free!(value.plan)
        end
        empty!(model.scratch); GC.gc(true)
        median(samples)
    end
    baseline = timeone("")
    println("OPCOST tokens=$tokens baseline_s=$baseline")
    flush(stdout)
    for aten in atens
        doubled = timeone(aten)
        cost = doubled - baseline
        println("OPCOST tokens=$tokens aten=$aten count=$(counts[aten]) ",
                "total_ms=$(round(1000cost, digits=2)) ",
                "per_op_us=$(round(1e6cost/counts[aten], digits=1)) ",
                "share=$(round(100cost/baseline, digits=1))%")
        flush(stdout)
    end
    nothing
end
