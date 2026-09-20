"""
Two tiers.

The cheap one asserts what is true with no assets at all: the package loads and
the lookup names a real place rather than throwing something unreadable.

The real one runs a TINY random K2Horizon — same wrapper, same `build()`, same
graph names, ~4 MB — end to end and compares to PyTorch. It is the only test
that can catch a wrong op: a grouped RMSNorm reducing over the wrong axis, an
`index_copy` into a transposed cache, a mask broadcast the other way. Build it
with `tools/export_horizon_tiny.py`; skipped when it is absent, because it needs
torch.

The latency test that matters — `frozen_stats().misses == 0` in a fresh process
— belongs here once the 32B weights are bound. See SAM2Runner/test for the shape
it should take; it has to run in a subprocess because Julia's compile-time
counter is per-process.
"""

using Test, HorizonRunner, DNNKernels, Lava

const TINY = get(ENV, "HORIZON_TINY_DIR",
                 joinpath(dirname(dirname(@__DIR__)), "gen", "graphs", "horizontiny"))

@testset "HorizonRunner" begin
    # No `assetdir()`. That is internal: it names where the artifact happens to
    # put things, and a test that calls it has to know the layout, so a re-export
    # that moves a file breaks a test that never knew it depended on that. Ask
    # for the graphs and the weights; `dir` is a config keyword on those, whose
    # default is the artifact.
    if HorizonRunner.ready()
        @info "K2 Horizon 32B: export present"
        @test !isempty(HorizonRunner.horizon32bgraphs())
        @test !isempty(HorizonRunner.horizon32bweights())
    else
        @info "K2 Horizon 32B: no export; run tools/export_horizon32b.py"
        # The error has to name the path — a caller who has not run the exporter
        # should be told where to put it, not handed a MethodError later.
        @test_throws ErrorException HorizonRunner.horizon32bgraphs()
    end

    if isfile(joinpath(TINY, "reference.safetensors"))
        ref = DNNKernels.readsafetensors(joinpath(TINY, "reference.safetensors"))
        # The PyTorch comparison runs UNQUANTISED. `horizon32b` stores its matmul
        # weights as int8 by default, which is a different function — close
        # enough to generate the same tokens on the real model and ~1e-2 on a
        # tiny one whose weights are random noise, so it cannot be checked
        # against an fp32 reference. The int8 path gets its own testset below.
        m = horizon32b(; dir = TINY, quantize = false)

        @testset "mask" begin
            # Byte-exact, not approximate. The masked value is
            # `torch.finfo(dtype).min` and NOT `-Inf`: `typemin` for a Julia
            # float is `-Inf`, which would start firing the trace's own
            # safe-softmax `scores == -inf` guard and change the result.
            slots = collect(Int64, 0:(m.prefill - 1))
            @test Array(causalmask(m, slots)) == ref["mask_prefill"]
            @test Array(causalmask(m, Int64[m.prefill])) == ref["mask_decode"]
        end

        @testset "prefill" begin
            ids = Int64.(vec(ref["ids"]))
            k, v = newcache(m)
            slots = collect(Int64, 0:(m.prefill - 1))
            lp, k2, v2 = DNNKernels.call(
                m.model, "horizon32b_prefill",
                DNNKernels.toback(m.backend, reshape(ids, m.prefill, 1)), k, v,
                DNNKernels.toback(m.backend, slots), causalmask(m, slots); dims = (;))
            @test maximum(abs, Array(lp) .- ref["logits_prefill"]) < 1e-5
            # The cache too: wrong logits are loud, a cache written to the wrong
            # slot is silent until the third generated token.
            @test maximum(abs, Array(k2) .- ref["k_after"]) < 1e-5
            @test maximum(abs, Array(v2) .- ref["v_after"]) < 1e-5

            ld, _, _ = DNNKernels.call(
                m.model, "horizon32b_decode",
                DNNKernels.toback(m.backend, reshape(Int64[ref["nxt"][1, 1]], 1, 1)),
                k2, v2, DNNKernels.toback(m.backend, Int64[m.prefill]),
                causalmask(m, Int64[m.prefill]); dims = (;))
            @test maximum(abs, Array(ld) .- ref["logits_decode"]) < 1e-5
        end

        @testset "int8 weights" begin
            # What CAN be pinned about the quantised path: that stacking the
            # matmuls does not change it. `fuseqkv` replaces Q/K/V with one
            # product over the row-stacked weight, and because the int8 scales
            # are per OUTPUT CHANNEL that is exact — the same scale applies to a
            # channel whichever matrix it is considered to belong to.
            function prefilllogits(mm)
                k, v = newcache(mm)
                slots = collect(Int64, 0:(mm.prefill - 1))
                lp, _, _ = DNNKernels.call(
                    mm.model, "horizon32b_prefill",
                    DNNKernels.toback(mm.backend,
                                      reshape(Int64.(vec(ref["ids"])), mm.prefill, 1)),
                    k, v, DNNKernels.toback(mm.backend, slots),
                    causalmask(mm, slots); dims = (;))
                Array(lp)
            end
            DNNKernels.FUSEQKV[] = false
            unfused = prefilllogits(horizon32b(; dir = TINY, quantize = true))
            DNNKernels.FUSEQKV[] = true
            fused = prefilllogits(horizon32b(; dir = TINY, quantize = true))
            @test fused == unfused

            # And that int8 stays within the error per-channel int8 actually has.
            # 1e-2 is loose against this model's random weights and far below the
            # 1e-1 that a wrong scale or a mis-packed word would produce.
            @test maximum(abs, fused .- ref["logits_prefill"]) < 1e-2
        end

        @testset "generate" begin
            # A 4-token prompt into a 6-token bucket, so padding and the
            # position bookkeeping across steps are under test, not just one
            # step. Token ids, so this is exact or it is broken.
            want = Int64.(vec(ref["greedy"]))
            @test generate(m, Int64.(vec(ref["prompt"])); maxnew = length(want)) == want
            @test generate(m, Int64.(vec(ref["prompt"])); maxnew = length(want)) == want
            immediate = horizon32b(; dir=TINY, quantize=false, record=false)
            changed = reverse(Int64.(vec(ref["prompt"])))
            expected = generate(immediate, changed; maxnew=length(want))
            @test generate(m, changed; maxnew=length(want)) == expected
            @test generate(m, changed; maxnew=length(want)) == expected
        end
        @testset "prompt and attention buckets" begin
            if haskey(m.model.graphs, "horizon32b_decode_bucket")
                immediate = horizon32b(; dir=TINY, quantize=false, record=false)
                for ids in (Int64[1, 2], Int64[4, 3, 2, 1], Int64[7, 6, 5, 4, 3])
                    # Cross the attention bucket boundary and reach the full
                    # capacity fallback, then reuse every recording next run.
                    want = generate(immediate, ids; maxnew=m.maxlen, bucketed=false)
                    for _ in 1:2
                        @test generate(m, ids; maxnew=m.maxlen) == want
                    end
                end
                bm = HorizonRunner.bucketmodel(m, "horizon32b_decode_bucket")
                @test bm.weights === m.model.weights
                @test length(bm.graphs) == 1
                @test any(k -> k isa Tuple && !isempty(k) && first(k) === :plan,
                          keys(bm.scratch))
                @test HorizonRunner.prefillbucket(m, 3) == 4
                @test HorizonRunner.attentionbucket(m, 16) == 16
                @test HorizonRunner.attentionbucket(m, 17) == 32
                # Two ladders: the power-of-two one up to the largest prompt
                # bucket, multiples of it above. See `chunkkv`.
                @test HorizonRunner.chunkkv(m, m.prefill) ==
                      HorizonRunner.attentionbucket(m, m.prefill)
                @test HorizonRunner.chunkkv(m, m.prefill + 1) == 2m.prefill
                @test HorizonRunner.chunkkv(m, m.maxlen + 1) == m.maxlen
                # The invariant, whichever ladder answers: the extent covers the
                # slots that are occupied, and never exceeds the cache.
                for occ in (1, 2, m.prefill - 1, m.prefill, m.prefill + 1,
                            2m.prefill, m.maxlen - 1, m.maxlen)
                    occ >= 1 || continue
                    kvc = HorizonRunner.chunkkv(m, occ)
                    @test min(occ, m.maxlen) <= kvc <= m.maxlen
                end
                # A prompt longer than the largest bucket, which is admitted
                # rather than refused: it runs as consecutive chunks against the growing
                # cache, and the recorded path has to agree with the immediate
                # one about the positions, the masks and which logit column is
                # the last prompt token's.
                if m.maxlen > 2m.prefill
                    n = m.prefill + 2
                    long = Int64[mod(i, m.vocab) for i in 1:n]
                    immediate2 = horizon32b(; dir=TINY, quantize=false, record=false)
                    want = generate(immediate2, long; maxnew=3)
                    @test length(want) == 3
                    for _ in 1:2
                        @test generate(m, long; maxnew=3) == want
                    end
                    # The fixed graphs run one prompt pass and say so.
                    @test_throws ErrorException generate(m, long; maxnew=1, bucketed=false)
                end
                name = "horizon32b_prefill_bucket_$(m.prefill)"
                if "logits_indices" in m.model.graphs[name].inputs
                    for index in (0, 3, m.prefill - 1)
                        k, v = newcache(m)
                        slots = collect(Int64, 0:m.prefill-1)
                        mask = HorizonRunner.maskbuffer(m, m.prefill; kv=16)
                        HorizonRunner.causalmask!(mask, slots)
                        args = (DNNKernels.toback(m.backend, Int64.(ref["ids"])),
                                k, v, DNNKernels.toback(m.backend, slots),
                                DNNKernels.toback(m.backend, mask),
                                DNNKernels.toback(m.backend, Int64[index]))
                        logits, gotk, gotv = HorizonRunner.bucketcall(m, name, args...; kv=16)
                        @test size(logits) == (m.vocab, 1, 1)
                        @test maximum(abs, vec(Array(logits)) .-
                            ref["logits_prefill"][:, index+1, 1]) < 1e-5
                        @test maximum(abs, Array(gotk) .- ref["k_after"]) < 1e-5
                        @test maximum(abs, Array(gotv) .- ref["v_after"]) < 1e-5
                    end
                end
                # Check complete logits and caches, not just greedy argmax.
                # Nonzero cache history catches an incorrectly sliced prefix.
                for token in (Int64(42), Int64(24))
                    function decode_outputs(bucketed)
                        slots = Int64[m.prefill]
                        kv = bucketed ? 16 : m.maxlen
                        mask = HorizonRunner.maskbuffer(m, 1; kv)
                        HorizonRunner.causalmask!(mask, slots)
                        args = map(x -> DNNKernels.toback(m.backend, copy(x)),
                            (reshape([token], 1, 1), ref["k_after"], ref["v_after"], slots, mask))
                        outputs = bucketed ? HorizonRunner.bucketcall(m,
                            "horizon32b_decode_bucket", args...; kv) :
                            DNNKernels.call(immediate.model, "horizon32b_decode", args...; dims=(;))
                        map(Array, outputs)
                    end
                    expected = decode_outputs(false)
                    actual = decode_outputs(true)
                    for (a, b) in zip(actual, expected)
                        @test maximum(abs, a .- b) < 1e-5
                    end
                end
                @test isempty(generate(m, Int64[1]; maxnew=0))
                @test_throws ArgumentError generate(m, Int64[])
                @test_throws ArgumentError generate(m, Int64[1]; maxnew=-1)
            else
                @test_skip false # regenerate the tiny export for bucket parity
            end
        end
    else
        @info "no tiny export; run tools/export_horizon_tiny.py to enable the end-to-end tests"
    end
end
