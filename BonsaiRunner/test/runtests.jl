using Test, BonsaiRunner, DNNKernels, Artifacts
import Mantle

@testset "checkpoint artifact bindings" begin
    toml = joinpath(pkgdir(BonsaiRunner), "Artifacts.toml")
    @test isfile(toml)
    @test BonsaiRunner.CHECKPOINT_BYTES == 5_946_648_928
    @test BonsaiRunner.CHECKPOINT_SHA256 ==
          "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
    for name in BonsaiRunner.CHECKPOINT_ARTIFACTS
        @test artifact_hash(name, toml) !== nothing
    end
end

@testset "Qwen35 GGUF tokenizer" begin
    enc, _ = BonsaiRunner._bytemap()
    tokens = string.(enc)
    push!(tokens, string(enc[Int(UInt8('h'))+1], enc[Int(UInt8('i'))+1]))
    md = Dict{String,Any}(
        "tokenizer.ggml.model"=>"gpt2",
        "tokenizer.ggml.tokens"=>tokens,
        "tokenizer.ggml.token_type"=>ones(Int,length(tokens)),
        "tokenizer.ggml.merges"=>[string(enc[Int(UInt8('h'))+1], ' ', enc[Int(UInt8('i'))+1])],
        "tokenizer.ggml.bos_token_id"=>0,
        "tokenizer.ggml.eos_token_id"=>1)
    file = GGUFFile("",UInt32(3),md,Dict{String,GGUFTensor}(),UInt64(0))
    tk = BonsaiTokenizer(file)
    ids = encode(tk,"hi")
    @test ids == [256]
    @test decode(tk,ids) == "hi"
    @test occursin("<|im_start|>user\nhello<|im_end|>", chatprompt(["user"=>"hello"]))
end


@testset "long-context KV sizing" begin
    @test BonsaiRunner.kv_bytes(1) == 33_280
    @test BonsaiRunner.kv_bytes(250_000) == 8_320_000_000
    @test BonsaiRunner.kv_bytes(262_144) == 8_724_152_320
end

# A session owns about 274 MiB and a recording of 1527 passes, and **a dropped
# session frees nothing** — Mantle frees on a verb and never from a finalizer.
# Both halves of that are pinned here because neither fails loudly: the leak is
# invisible until the pool is exhausted, and by then nothing names the cause.
#
# Found by measuring what looked like a per-session slowdown and was not: twelve
# sessions held 11 GB, and the KV cache leaked a whole generation of itself per
# doubling because `_ensure_kv!` swapped the caches without freeing the old ones.
#
# The whole checkpoint is 5.9 GB, so this runs only where it is already fetched.
@testset "a session gives its memory back" begin
    loaded = all(artifact_exists(artifact_hash(n, joinpath(pkgdir(BonsaiRunner),
                                                           "Artifacts.toml")))
                 for n in BonsaiRunner.CHECKPOINT_ARTIFACTS)
    if !loaded
        @info "Bonsai checkpoint parts not in the artifact store; skipping"
    else
        model = Bonsai2()
        dev = Mantle.todevice(model.backend)
        onloan() = sum(b -> length(b.live),
                       Iterators.flatten(values(Mantle.pool(dev).blocks)); init = 0)

        before = onloan()
        s = session(model)
        @test onloan() > before                     # it took memory
        # …and the growth path replaces the caches rather than accumulating them.
        caps = BonsaiRunner.kv_capacity(s)
        held = onloan()
        BonsaiRunner._ensure_kv!(s, caps + 1)
        @test BonsaiRunner.kv_capacity(s) > caps
        # RECLAIM before counting: `free!` retires a region and the ledger keeps
        # it until the device has passed the submission that was reading it, so
        # the count right after the growth includes both generations either way
        # and says nothing.
        Mantle.reclaim!(Mantle.pool(dev), dev; wait = true)
        # 16 of the 64 layers are full-attention and each holds four buffers, so
        # a leaked generation is exactly 64 entries that never come back.
        @test onloan() - held < 64

        release!(s)
        Mantle.reclaim!(Mantle.pool(dev), dev; wait = true)
        @test onloan() <= before                    # and gave all of it back
        DNNKernels.releaseweights!(model.weights)
    end
end
