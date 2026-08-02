"""
SAM 2's decode as a captured command buffer, replayed per click.

    julia --project=. dev/JuliaVision/DNNKernels/test/test_replay_decode.jl

The decoder's work is identical on every click but its *inputs* are not, so this
is only correct if a replay reads the new bytes out of buffers whose addresses
never move. Two things make that true and both are under test here: the decoder's
converted inputs are cached (`CACHE_DECODER_INPUTS`), and `prompt` writes each
click into one persistent pair instead of allocating a fresh one.

The failure this guards against is silent. A replay whose buffers had been
reallocated does not crash — it re-runs over whatever now lives at those
addresses and returns a *plausible* mask, which for a segmentation model is
indistinguishable from a slightly different click. So the load-bearing test is
not "the same click replays identically"; it is **a different click through the
same buffers gives the same answer the unreplayed path gives**.

The masks come back in the captured output buffer, which the next decode
overwrites. Every comparison below copies to the host first: materialising two
decodes and comparing them compares one array with itself.

Measured here: record+run 4.40 ms, replay 3.12 ms, bit-exact, 48% -> 67% of
PyTorch's decoder.
"""

using Test, DNNKernels, KernelAbstractions, Lava
using DNNKernels: readsafetensors, SAM2, encode, decode, segment, prompt, toback
const KA = KernelAbstractions
const DK = DNNKernels

const G = DNNKernels.findasset(joinpath("gen", "graphs"); from = @__DIR__)
const MODELDIR = joinpath(G, "sam2-large")

@testset "SAM 2 decode capture/replay" begin
    back = LavaBackend()
    refs = readsafetensors(joinpath(MODELDIR, "refs.safetensors"))
    sam = SAM2(MODELDIR, joinpath(MODELDIR, "weights.safetensors"); backend = back, res = 1024)
    feats = encode(sam, toback(back, refs["sam2_encoder/in0"]))
    KA.synchronize(back)

    # `decode` called with tensors that did not come from `prompt` must not
    # replay: nothing keeps those alive or writes them in place.
    @testset "only prompt-owned buffers are replayable" begin
        loose = toback(back, zeros(Float32, 2, sam.maxpoints, 1))
        loosel = toback(back, fill(Int32(1), sam.maxpoints, 1))
        @test !DK.replayable(sam, feats, loose, loosel)
        sam.replay[] = nothing
        decode(sam, feats, loose, loosel)
        KA.synchronize(back)
        @test sam.replay[] === nothing
    end

    pt, lb = prompt(sam, [(0.5, 0.5)], [true])
    @test DK.replayable(sam, feats, pt, lb)

    @testset "the first click captures, the second replays" begin
        sam.replay[] = nothing
        m, _ = decode(sam, feats, pt, lb); KA.synchronize(back)
        first = copy(Array(m))
        @test sam.replay[] !== nothing
        seq = sam.replay[][4]
        m, _ = decode(sam, feats, pt, lb); KA.synchronize(back)
        @test copy(Array(m)) == first
        @test sam.replay[][4] === seq        # replayed, not re-captured
    end

    @testset "a different click through the same buffers is not stale" begin
        # `prompt` returns the same two arrays, so a replay that ignored their
        # contents would return the mask above and look entirely reasonable.
        pt2, lb2 = prompt(sam, [(0.25, 0.25)], [true])
        @test pt2 === pt && lb2 === lb
        replayed, _ = decode(sam, feats, pt2, lb2); KA.synchronize(back)
        replayed = copy(Array(replayed))

        DK.REPLAY_DECODE[] = false
        sam.replay[] = nothing
        recorded, _ = decode(sam, feats, pt2, lb2); KA.synchronize(back)
        recorded = copy(Array(recorded))
        DK.REPLAY_DECODE[] = true

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

    @testset "a new embedding keeps the capture, and is not stale" begin
        # `encode` writes into the same slab buffers and returns the same tuple,
        # so the capture's addresses still hold the right data and the sequence
        # is deliberately NOT rebuilt. That is only safe if the replay reads the
        # new features, which is the assertion below — a stale replay would hand
        # back frame A's mask, and for a segmentation model that is not visibly
        # wrong. Checked against the recorded path rather than against a guess.
        prompt(sam, [(0.5, 0.5)], [true])
        seq = (decode(sam, feats, pt, lb); KA.synchronize(back); sam.replay[][4])
        maskA = copy(Array(decode(sam, feats, pt, lb)[1])); KA.synchronize(back)

        feats2 = encode(sam, toback(back, refs["sam2_encoder/in0"] .* 0.5f0))
        KA.synchronize(back)
        @test feats2 === feats                 # in place, hence the paragraph above
        replayed = copy(Array(decode(sam, feats2, pt, lb)[1])); KA.synchronize(back)
        @test sam.replay[][4] === seq          # served, not re-recorded

        DK.REPLAY_DECODE[] = false
        sam.replay[] = nothing
        recorded = copy(Array(decode(sam, feats2, pt, lb)[1])); KA.synchronize(back)
        DK.REPLAY_DECODE[] = true

        @test replayed == recorded
        @test replayed != maskA                # the new frame did reach the mask
    end

    @testset "a converted decoder input is refreshed per frame" begin
        # The shipped autocast decoder declares the dtypes the encoder already
        # produces, so `args === feats` and there is nothing to go stale. The
        # fp32 decoder converts, and then `cacheval` holds a *copy* — which the
        # encoder writing in place would leave holding the previous frame. This
        # pins the invalidation regardless of which export is loaded.
        conv = [i for i in 1:3
                if eltype(feats[i]) !== sam.model.graphs["sam2_decoder"].buffers[
                       sam.model.graphs["sam2_decoder"].inputs[i]].dtype]
        decode(sam, feats, pt, lb); KA.synchronize(back)
        before = sam.cacheval[]
        encode(sam, toback(back, refs["sam2_encoder/in0"]))
        @test sam.cachekey[] === nothing       # encode invalidates; `===` cannot
        decode(sam, feats, pt, lb); KA.synchronize(back)
        for i in conv
            # Refreshed in place: same buffer, new contents. A reallocation here
            # would retire the capture, which is correct but costs the win.
            @test sam.cacheval[][i] === before[i]
            @test Array(sam.cacheval[][i]) ≈ Array(feats[i])
        end
        isempty(conv) && @test sam.cacheval[] == feats[1:3]
    end
end
