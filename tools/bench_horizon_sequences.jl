# Match llama-bench's depth + 32 advancing decode positions. Unlike a fixed-
# position microbenchmark, fill the prefix with real model KV values first.
using HorizonRunner, DNNKernels, Mantle, KernelAbstractions, Statistics, Random

# A prompt longer than the largest bucket. It runs — as consecutive bucket
# prefills against a growing cache, which is what `generate` and the sequence
# benchmark above already do — so it is measurable, and was not measured.
# `llama-bench -p 1024,2048` is the comparison.
function prefillchunks!(m, k, v, tokens; kvof = (m, occ) -> HorizonRunner.attentionbucket(m, occ))
    n = length(tokens)
    for start in 0:m.prefill:n-1
        valid = min(m.prefill, n - start)
        t = HorizonRunner.prefillbucket(m, valid)
        kv = kvof(m, start + t)
        ids = zeros(Int64, t, 1); ids[1:valid, 1] .= tokens[start+1:start+valid]
        positions = collect(Int64, start:start+t-1)
        mask = HorizonRunner.maskbuffer(m, t; kv)
        HorizonRunner.causalmask!(mask, positions)
        args = (DNNKernels.toback(m.backend, ids), k, v,
                DNNKernels.toback(m.backend, positions),
                DNNKernels.toback(m.backend, mask),
                DNNKernels.toback(m.backend, Int64[valid-1]))
        logits = first(HorizonRunner.bucketcall(m, "horizon32b_prefill_bucket_$t", args...; kv))
        start + valid >= n && return logits
    end
end

function bench_horizon_longprompt(m; prompts=(1024, 2048), repeats=5, warmup=2,
                                  kvof = (m, occ) -> HorizonRunner.attentionbucket(m, occ),
                                  label = "pow2")
    k, v = get!(() -> HorizonRunner.newcache(m), m.model.scratch,
                (:horizon_generation_cache,))
    host = Matrix{Float16}(undef, m.vocab, 1)
    for n in prompts
        tokens = [mod(i, 1000) + 1 for i in 1:n]
        run() = begin
            fill!(k, 0); fill!(v, 0)
            copyto!(host, reshape(prefillchunks!(m, k, v, tokens; kvof), m.vocab, 1))
            argmax(vec(host))
        end
        expected = run()
        for _ in 1:warmup-1; run(); end
        KernelAbstractions.synchronize(m.backend)
        samples = Float64[]
        for _ in 1:repeats
            t = @elapsed got = run()
            got == expected || error("long prompt $n changed its selected token")
            push!(samples, t)
        end
        println("HORIZON_LONGPP kv=$label n=$n chunks=$(cld(n, m.prefill)) ",
                "median_s=$(median(samples)) samples=$samples token=$expected ",
                "pool_gib=$(Mantle.reserved(Mantle.pool(m.model.device)) / 2.0^30)")
        flush(stdout)
    end
    nothing
end

function bench_horizon_sequences(m; depths=(0,96,480,2016), count=32, repeats=7,
                                 kvof = (m, occ) -> HorizonRunner.attentionbucket(m, occ),
                                 label = "pow2")
    be=m.backend
    k,v=get!(m.model.scratch,(:horizon_generation_cache,)) do
        HorizonRunner.newcache(m)
    end
    for depth in depths
        depth+count<=m.maxlen || throw(ArgumentError("sequence exceeds cache"))
        fill!(k,0); fill!(v,0)
        rng=MersenneTwister(42)
        tokens=rand(rng,Int64(0):Int64(m.vocab-1),depth+count)
        for start in 0:m.prefill:depth-1
            valid=min(m.prefill,depth-start)
            t=HorizonRunner.prefillbucket(m,valid)
            kv=HorizonRunner.attentionbucket(m,start+t)
            ids=zeros(Int64,t,1); ids[1:valid,1].=tokens[start+1:start+valid]
            positions=collect(Int64,start:start+t-1)
            mask=HorizonRunner.maskbuffer(m,t;kv)
            HorizonRunner.causalmask!(mask,positions)
            args=(DNNKernels.toback(be,ids),k,v,DNNKernels.toback(be,positions),
                  DNNKernels.toback(be,mask),DNNKernels.toback(be,Int64[valid-1]))
            HorizonRunner.bucketcall(m,"horizon32b_prefill_bucket_$t",args...;kv)
        end
        tokh=zeros(Int64,1,1); posh=zeros(Int64,1)
        tokd=DNNKernels.toback(be,tokh); posd=DNNKernels.toback(be,posh)
        masks=Dict(kv=>(HorizonRunner.maskbuffer(m,1;kv),
                       DNNKernels.toback(be,HorizonRunner.maskbuffer(m,1;kv)))
                   for kv in unique(kvof(m,depth+i) for i in 1:count))
        host=Array{Float16}(undef,m.vocab,1,1)
        sequence()=begin
            checksum=0
            for i in 1:count
                posh[1]=depth+i-1; tokh[1]=tokens[depth+i]
                kv=kvof(m,depth+i)
                mh,md=masks[kv]; HorizonRunner.causalmask!(mh,posh)
                copyto!(posd,posh); copyto!(tokd,tokh); copyto!(md,mh)
                logits=first(HorizonRunner.bucketcall(m,"horizon32b_decode_bucket",tokd,k,v,posd,md;kv))
                copyto!(host,logits)
                checksum+=argmax(vec(host))
            end
            checksum
        end
        expected=sequence() # compile/capture every encountered bucket, untimed
        samples=Float64[]
        for _ in 1:repeats
            elapsed=@elapsed checksum=sequence()
            checksum==expected || error("repeated decode sequence changed outputs")
            push!(samples,1000elapsed/count)
        end
        println("HORIZON_SEQUENCE kv=$label depth=$depth count=$count ",
                "extents=$(sort(unique(kvof(m,depth+i) for i in 1:count))) ",
                "median_ms=$(median(samples)) samples=$samples")
        flush(stdout)
    end
    nothing
end
