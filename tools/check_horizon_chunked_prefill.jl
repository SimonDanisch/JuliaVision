# A prompt longer than the largest bucket, against the unfused graphs.
#
# `generate`'s chunking was checked against the benchmark's chunking and against
# an immediate model — both of which run the SAME algorithm, so a mistake in the
# positions, the mask or which logit column is the last prompt token's would
# agree with itself. This runs the identical chunk sequence through graphs whose
# attention is still `bmm -> softmax -> bmm`, which is the one comparison that
# can catch that, and it checks both complete caches and not just the logits.
using HorizonRunner, DNNKernels, Mantle, KernelAbstractions, Test

# One chunk's inputs. Shared by both sides so the comparison is of the graphs
# and not of two transcriptions of the bookkeeping.
function chunkargs(m, tokens, start, k, v)
    n = length(tokens)
    valid = min(m.prefill, n - start)
    T = HorizonRunner.prefillbucket(m, valid)
    kv = HorizonRunner.chunkkv(m, start + T)
    ids = zeros(Int64, T, 1)
    ids[1:valid, 1] .= @view tokens[(start + 1):(start + valid)]
    slots = collect(Int64, start:(start + T - 1))
    mask = HorizonRunner.maskbuffer(m, T; kv)
    HorizonRunner.causalmask!(mask, slots)
    (DNNKernels.toback(m.backend, ids), k, v,
     DNNKernels.toback(m.backend, slots), DNNKernels.toback(m.backend, mask),
     DNNKernels.toback(m.backend, Int64[valid - 1])), T, kv
end

"""
    check_horizon_chunked_prefill(m, reference_graphs; tokens, maxnew)

Chunked prefill against the unfused graphs. With `maxnew > 0` it also generates
that many tokens through both and compares them, which is the assertion that
carries the weight here.

**The caches drift and the tokens do not**, and the bounds say so. Measured
against the unfused graphs, relative to each output's own maximum:

    tokens  chunks   logits      K        V     tokens equal
      1024     2     0.0017   0.0092   0.0174        yes
      1536     3     0.0018   0.0302   0.0637        yes
      2048     4     0.0024   0.0302   0.0628        yes
      4096     8     0.0028   0.0302   0.0916         -

The third chunk is where it jumps, and that is where `nk` passes the staged
route's 1024 ceiling and attention moves to the flash kernel. The K figure is
then *bit-identical* at 1536, 2048 and 4096 — same 1.3345947 maximum — so this
is one route change saturating, not an accumulation that grows with the prompt.
V creeps on past it (1.676 to 2.445), which is the compounding part. The logits stay inside 0.3% throughout and
eight generated tokens are identical to the unfused graphs' at 1536 and 2048.
So the logits get a tight bound, the tokens get an exact one, and the caches get
a loose rail — a bound tight enough to fail on 0.09 would only be measuring
which side of the ceiling the last chunk fell on.
"""
function check_horizon_chunked_prefill(m, reference_graphs; tokens = 1024, maxnew::Int = 0)
    base = m.model
    k, v = get!(() -> HorizonRunner.newcache(m), base.scratch,
                (:horizon_generation_cache,))
    prompt = [mod(i, 1000) + 1 for i in 1:tokens]
    names = unique(["horizon32b_prefill_bucket_$(chunkargs(m, prompt, s, k, v)[2])"
                    for s in 0:m.prefill:(tokens - 1)])
    reference = DNNKernels.Model(Dict(n => reference_graphs[n] for n in names),
        base.weights, base.backend, base.memevery, base.memframes, base.topk;
        record = false)
    function runchunks(model)
        fill!(k, 0); fill!(v, 0)
        logits = nothing
        for start in 0:m.prefill:(tokens - 1)
            args, T, kv = chunkargs(m, prompt, start, k, v)
            logits = first(DNNKernels.call(model, "horizon32b_prefill_bucket_$T",
                                           args...; dims = (; kv)))
        end
        (Array(logits), Array(k), Array(v))
    end
    @testset "chunked $tokens-token prefill against the unfused graphs" begin
        expected = runchunks(reference)
        reference = nothing
        got = runchunks(HorizonRunner.bucketmodel(m, "horizon32b_prefill_bucket_$(m.prefill)"))
        for (i, (a, b)) in enumerate(zip(got, expected))
            err = maximum(abs, Float32.(a) .- Float32.(b))
            rel = err / max(maximum(abs, Float32.(b)), eps(Float32))
            println("CHUNKED output=$i size=$(size(a)) finite=$(all(isfinite, a)) ",
                    "max_abs=$err relative_max=$rel")
            @test all(isfinite, a)
            @test rel < (i == 1 ? .01 : .15)
        end
        @test argmax(first(got)) == argmax(first(expected))
        if maxnew > 0
            prompt = [mod(i, 1000) + 1 for i in 1:tokens]
            refmodel = DNNKernels.Model(Dict(n => reference_graphs[n]
                                             for n in keys(reference_graphs)),
                base.weights, base.backend, base.memevery, base.memframes, base.topk;
                record = false)
            refrunner = Horizon32B(m.backend, refmodel, m.maxlen, m.prefill, m.vocab)
            want = generate(refrunner, prompt; maxnew)
            gottokens = generate(m, prompt; maxnew)
            println("CHUNKED_TOKENS tokens=$tokens exact=$(gottokens == want) $gottokens")
            @test gottokens == want
        end
        # And the whole point of chunking: the cache the chunks built is the
        # cache a caller then decodes against, so it has to be filled to the
        # last prompt slot and zero past it.
        gk = got[2]
        @test !all(iszero, gk[:, tokens, :, :, :])
        @test tokens == m.maxlen || all(iszero, gk[:, tokens + 1, :, :, :])
        flush(stdout)
    end
    nothing
end
