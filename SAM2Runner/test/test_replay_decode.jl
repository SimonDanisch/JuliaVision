"""
SAM 2's decode as a baked Mantle plan, reused per click.

    julia --project=. dev/JuliaVision/SAM2Runner/test/runtests.jl

The decoder's work is identical on every click but its *inputs* are not, so this
is only correct if the plan reads the new bytes out of buffers whose addresses
never move. Two things make that true and both are under test here: `encode`
writes the frame and the dtype-converted features into the plan's own buffers
in place, and `prompt` writes each click into one persistent pair instead of
allocating a fresh one.

The failure this guards against is silent. A plan whose buffers had been
reallocated does not crash — it re-runs over whatever now lives at those
addresses and returns a *plausible* mask, which for a segmentation model is
indistinguishable from a slightly different click. So the load-bearing test is
not "the same click replays identically"; it is **a different click through the
same buffers gives the same answer the eager path gives**, where the eager path
is `DNNKernels.call` on the same graphs with the same inputs.

The masks come back in the plan's output buffer, which the next decode
overwrites. Every comparison below copies to the host first: materialising two
decodes and comparing them compares one array with itself.

This file tested the raw `Lava.CapturedSequence` replay until the baked Mantle
plans replaced it (`SAM2.plans`; see `sam2.jl` for why a plan cannot express the
GC fault the capture had). The properties under test are unchanged; the
internals they poke are not.
"""

using Test, DNNKernels, KernelAbstractions, Lava, SAM2Runner
using DNNKernels: readsafetensors, toback
using SAM2Runner: SAM2, encode, decode, segment, prompt
const KA = KernelAbstractions
const DK = DNNKernels

# The runner builds its own model from its own artifact — `sam2model` already
# takes `dir` and `res` as config, so nothing here needs to know where the
# weights are. References come through `sam2refs()` for the same reason: they
# live in a SECOND artifact (`tools/make_artifacts.jl` keeps test fixtures out of
# a model tarball) and that split is the runner's business, not this file's.
const HAVE_SAM2 = SAM2Runner.ready()

# The reference for "is the plan stale": the same graphs run eagerly, on the
# same inputs. `p.feat` holds the encoder outputs already converted to the
# decoder's dtypes — exactly what `decode` itself stages.
function eager_decode(sam, pt, lb)
    p = SAM2Runner.plansfor(sam)
    DK.call(sam.model, "sam2_decoder", p.feat[1], p.feat[2], p.feat[3], pt, lb;
            dims = sam.dims, clampattn = true)
end

@testset "SAM 2 decode through a baked plan" begin
    back = LavaBackend()
    refs = SAM2Runner.sam2refs()
    sam = SAM2Runner.sam2model(; backend = back, res = 1024)
    feats = encode(sam, toback(back, refs["sam2_encoder/in0"]))
    KA.synchronize(back)

    @testset "inputs that are not the plan's are staged by copy" begin
        # `decode` with tensors that did not come from `prompt` copies them into
        # the plan-owned pair, so a foreign click and the same click through
        # `prompt` agree.
        pt, lb = prompt(sam, [(0.5, 0.5), (0.25, 0.75)], [true, false])
        m1, _ = decode(sam, feats, pt, lb); KA.synchronize(back)
        m1 = copy(Array(m1))
        loose = toback(back, Array(pt))
        loosel = toback(back, Array(lb))
        @test loose !== pt && loosel !== lb
        m2, _ = decode(sam, feats, loose, loosel); KA.synchronize(back)
        @test copy(Array(m2)) == m1
    end

    pt, lb = prompt(sam, [(0.5, 0.5)], [true])

    @testset "the plan is built once and reused" begin
        plans = sam.plans[]
        @test plans !== nothing
        m, _ = decode(sam, feats, pt, lb); KA.synchronize(back)
        first = copy(Array(m))
        m, _ = decode(sam, feats, pt, lb); KA.synchronize(back)
        @test copy(Array(m)) == first
        @test sam.plans[] === plans            # replayed, not rebuilt
    end

    @testset "a different click through the same buffers is not stale" begin
        # `prompt` returns the same two arrays, so a plan that ignored their
        # contents would return the mask above and look entirely reasonable.
        pt2, lb2 = prompt(sam, [(0.25, 0.25)], [true])
        @test pt2 === pt && lb2 === lb
        replayed, _ = decode(sam, feats, pt2, lb2); KA.synchronize(back)
        replayed = copy(Array(replayed))
        recorded = copy(Array(eager_decode(sam, pt2, lb2)[1]))

        @test replayed == recorded
        # ...and it did move, so the equality above is not two copies of one mask.
        prompt(sam, [(0.5, 0.5)], [true])
        other, _ = decode(sam, feats, pt, lb); KA.synchronize(back)
        @test copy(Array(other)) != recorded
    end

    @testset "many clicks in a row" begin
        for i in 1:12
            p, l = prompt(sam, [(0.1 + 0.06i, 0.5)], [true])
            mask, score = segment(sam, feats, [(0.1 + 0.06i, 0.5)], [true])
            @test size(mask) == (256, 256)
            @test isfinite(score)
        end
        KA.synchronize(back)
    end

    @testset "replaying next to ordinary GPU work" begin
        # The editor renders a preview between two clicks. That interleaving is
        # what desynced the batch timeline before Lava's `replay!` learned to
        # close an open batch; `Lava/test/test_replay_interleaved.jl` covers the
        # mechanism, this covers it through the decoder.
        scratch = KA.allocate(back, Float32, 1 << 16)
        for i in 1:8
            fill!(scratch, Float32(i))                     # records, leaves a batch open
            p, l = prompt(sam, [(0.5, 0.4 + 0.01i)], [true])
            decode(sam, feats, p, l)                       # replays
            scratch .= scratch .* 2f0                      # records again
        end
        KA.synchronize(back)
        @test all(==(16f0), Array(scratch))
        m, _ = segment(sam, feats, [(0.5, 0.5)], [true])
        @test all(isfinite, Array(m))
    end

    @testset "a new embedding reaches the same plan" begin
        # `encode` writes the frame and the converted features into the plan's
        # buffers in place, so the plan's addresses still hold the right data
        # and it is NOT rebuilt. That is only safe if the next decode reads the
        # new features — a stale plan would hand back frame A's mask, and for a
        # segmentation model that is not visibly wrong.
        prompt(sam, [(0.5, 0.5)], [true])
        maskA = copy(Array(decode(sam, feats, pt, lb)[1])); KA.synchronize(back)
        plans = sam.plans[]

        feats2 = encode(sam, toback(back, refs["sam2_encoder/in0"] .* 0.5f0))
        KA.synchronize(back)
        replayed = copy(Array(decode(sam, feats2, pt, lb)[1])); KA.synchronize(back)
        @test sam.plans[] === plans            # reused, not rebuilt

        @test replayed != maskA                # the new frame did reach the mask
        @test replayed == copy(Array(eager_decode(sam, pt, lb)[1]))
    end

    @testset "encode refreshes the decoder's feature buffers in place" begin
        # The two graphs are exported under different precision policies, so
        # three encoder outputs need converting before the decoder can read
        # them. The conversion is per *frame* and lands in the plan's `feat`
        # buffers: same addresses, new contents. A reallocation here would not
        # be wrong — `decode` stages by copy — it would just cost the win.
        p = sam.plans[]
        before = p.feat
        feats3 = encode(sam, toback(back, refs["sam2_encoder/in0"]))
        KA.synchronize(back)
        @test sam.plans[] === p
        for i in 1:3
            @test p.feat[i] === before[i]
            @test Array(p.feat[i]) ≈ Array(feats3[i])
        end
    end
end
