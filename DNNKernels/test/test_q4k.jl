"""
Q4_K weights (`q4k.jl`): a port of ggml's quantizer, and a product that reads int8
activations through `KI.dot4x8`.

Against the host port of both. The device quantizers are the host ones compiled
for the device, bit for bit where float division is correctly rounded (ROCm).
RADV's is not, so there a few weight blocks settle on another scale of the same
error, and an activation that sits near a rounding tie can land one int8 level
away. The GEMV is checked against the dequantized weight times the activations
the device quantized, to float rounding.

The host port itself was checked byte for byte against ggml's
`quantize_row_q4_K_ref` (libggml-base, llama.cpp `d4abd57`) when it was written:
512 blocks, with outliers, an all-zero block and a positive minimum among them.
"""

using Test, DNNKernels, Mantle, KernelAbstractions, Random

const DKA = DNNKernels

# ggml's `quantize_row_q8_1` as `q8act_kernel!` computes it, dequantized.
function q8roundtrip(x::AbstractVector{Float32})
    y = similar(x)
    for b in 0:(length(x) ÷ 32 - 1)
        blk = x[(32b + 1):(32b + 32)]
        d = maximum(abs, blk) / 127f0
        id = d == 0f0 ? 0f0 : 1f0 / d
        y[(32b + 1):(32b + 32)] .= d .* round.(blk .* id)
    end
    return y
end

# The activations as `q8act_kernel!` quantized them on the device, dequantized,
# and the scale of each element's block.
function q8device(backend, dev, X)
    K, n = size(X)
    xq = KernelAbstractions.allocate(backend, DKA.UVec4, K ÷ 16, n)
    xds = KernelAbstractions.allocate(backend, Float32, K ÷ 16, n)
    Mantle.kilaunch!(backend, DKA.q8act_kernel!, xq, xds, dev(X), Int32(K), Int32(n * (K ÷ 32));
                     ndrange = n * (K ÷ 32), workgroupsize = 256)
    q = reshape(reinterpret(Int8, collect(reinterpret(UInt32, vec(Array(xq))))), K, n)
    ds = Array(xds)
    d = [ds[2 * ((k - 1) ÷ 32) + 1, c] for k in 1:K, c in 1:n]
    return d .* Float32.(q), d
end

@testset "Q4_K" begin
    backend = first(Mantle.eachbackend())
    rng = MersenneTwister(7)
    rows, K = 64, 1024
    W = Float16.(randn(rng, Float32, rows, K) .* 0.02f0)
    W[rand(rng, 1:length(W), 40)] .*= 8          # outliers, as real weights have
    W[3, 257:512] .= 0                           # a block of zeros
    W[4, 1:256] .= abs.(W[4, 1:256])             # a block with a positive minimum
    relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

    Q = DKA.q4k_quantize(W)
    deq = DKA.q4k_dequantize(Q, rows, K)
    @test length(Q) == 9 * (K ÷ 256) * rows
    @test relerr(deq, Float32.(W)) < 0.09         # 0.076 measured: 4.5 bits on Gaussian weights
    @test all(iszero, deq[3, 257:512])

    dev(x) = (d = KernelAbstractions.allocate(backend, eltype(x), size(x)...); copyto!(d, x); d)
    A = DKA.quantizeq4k(backend, dev(W))
    Qd = vec(Array(Mantle.storage(A.q)))
    deqd = DKA.q4k_dequantize(Qd, rows, K)
    @test count(Qd .!= Q) <= length(Q) ÷ 10
    @test relerr(deqd, Float32.(W)) < 0.09

    @testset "product of $n columns, $R rows a group" for n in (1, 3, 8), R in (1, 2)
        X = Float16.(randn(rng, Float32, K, n))
        out = KernelAbstractions.allocate(backend, Float32, rows, n)
        DKA.q4kgemv!(backend, out, A, dev(X); R)
        Xq, d = q8device(backend, dev, X)
        host = reduce(hcat, [q8roundtrip(Float32.(X[:, c])) for c in 1:n])
        @test all(abs.(Xq .- host) .<= 1.01f0 .* d)
        ref = Float64.(deqd) * Float64.(Xq)
        @test maximum(abs.(Array(out) .- ref)) / maximum(abs.(ref)) < 1e-5
    end

    @testset "bias and epilogue in the store" begin
        X = Float16.(randn(rng, Float32, K, 2))
        bias = randn(rng, Float32, rows)
        out = KernelAbstractions.allocate(backend, Float32, rows, 2)
        DKA.q4kgemv!(backend, out, A, dev(X); bias = dev(bias), epilogue = x -> max(x, 0f0))
        Xq, _ = q8device(backend, dev, X)
        ref = max.(Float64.(deqd) * Float64.(Xq) .+ bias, 0)
        @test maximum(abs.(Array(out) .- ref)) / maximum(abs.(ref)) < 1e-5
    end

    @testset "dequantized for wide products" begin
        Wd = KernelAbstractions.allocate(backend, Float16, rows, K)
        Mantle.kilaunch!(backend, DKA.q4k_dequant_kernel!, Wd, Mantle.storage(A.q), Int32(rows), Int32(K ÷ 256);
                         ndrange = rows * (K ÷ 256) * 8, workgroupsize = 256)
        @test Array(Wd) == Float16.(deqd)
    end
end
