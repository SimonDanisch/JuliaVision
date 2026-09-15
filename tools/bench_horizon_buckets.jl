# Include after loading bucket_horizon. Timings include logit readback.
using HorizonRunner, DNNKernels, Mantle, KernelAbstractions, Statistics

function bench_horizon_buckets(m; prompts=(32, 128, 512), depths=(32, 128, 512, 2048),
                               repeats=7, warmup=3)
    backend = m.backend
    # Same cache the generation path uses, so a fresh session can benchmark
    # without first running a generation check.
    k, v = get!(() -> HorizonRunner.newcache(m), m.model.scratch,
                (:horizon_generation_cache,))
    sync() = KernelAbstractions.synchronize(backend)
    for n in prompts
        name = "horizon32b_prefill_bucket_$n"
        kv = HorizonRunner.attentionbucket(m, n)
        slots = collect(Int64, 0:n-1)
        mask = HorizonRunner.maskbuffer(m, n; kv)
        HorizonRunner.causalmask!(mask, slots)
        args = (DNNKernels.toback(backend, reshape(mod.(slots, 1000) .+ 1, n, 1)),
                k, v, DNNKernels.toback(backend, slots), DNNKernels.toback(backend, mask))
        selectedlogit = "logits_indices" in m.model.graphs[name].inputs
        if selectedlogit
            args = (args..., DNNKernels.toback(backend, Int64[n-1]))
        end
        host=Matrix{Float16}(undef,m.vocab,1)
        run() = begin
            logits=first(HorizonRunner.bucketcall(m,name,args...;kv))
            if selectedlogit
                copyto!(host,reshape(logits,m.vocab,1))
            else
                copyto!(vec(host),Array(copy(view(logits,:,n,1))))
            end
            nothing
        end
        for _ in 1:warmup; run(); end; sync()
        stats = [(@timed run()) for _ in 1:repeats]
        samples = [s.time for s in stats]
        println("HORIZON_PP n=$n kv=$kv median_s=$(median(samples)) samples=$samples gc_s=$(sum(s.gctime for s in stats))")
        flush(stdout)
    end
    for occupied in depths
        kv = HorizonRunner.attentionbucket(m, occupied)
        slots = Int64[occupied-1]
        mask = HorizonRunner.maskbuffer(m, 1; kv)
        HorizonRunner.causalmask!(mask, slots)
        args = (DNNKernels.toback(backend, ones(Int64, 1, 1)), k, v,
                DNNKernels.toback(backend, slots), DNNKernels.toback(backend, mask))
        host = Matrix{Float16}(undef, m.vocab, 1)
        run() = begin
            out = first(HorizonRunner.bucketcall(m, "horizon32b_decode_bucket", args...; kv))
            copyto!(host, reshape(out, m.vocab, 1))
            argmax(host)
        end
        for _ in 1:warmup; run(); end; sync()
        samples = [1000 * (@elapsed run()) for _ in 1:20]
        println("HORIZON_TG occupied=$occupied kv=$kv median_ms=$(median(samples)) samples=$samples")
        flush(stdout)
    end
end
