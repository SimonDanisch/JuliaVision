import KernelInterface as KI
using Test, DNNKernels, KernelAbstractions, Mantle, Random

function write_tiny_gguf(path)
    open(path, "w") do io
        write(io, codeunits("GGUF")); write(io, UInt32(3))
        write(io, UInt64(1)); write(io, UInt64(2))
        function str(s)
            write(io, UInt64(ncodeunits(s))); write(io, codeunits(s))
        end
        str("general.alignment"); write(io, UInt32(4)); write(io, UInt32(32))
        str("test.name"); write(io, UInt32(8)); str("bonsai")
        str("x"); write(io, UInt32(1)); write(io, UInt64(4)); write(io, UInt32(0)); write(io, UInt64(0))
        while position(io) % 32 != 0
            write(io, UInt8(0))
        end
        write(io, Float32[1,2,3,4])
    end
end

@testset "Qwen35 recurrent decode state" begin
    backend = Mantle.defaultbackend(); ctx = DNNKernels.Ctx(backend)
    rng = MersenneTwister(117)
    qh = randn(rng,Float32,128,16).*0.1f0
    kh = randn(rng,Float32,128,16).*0.1f0
    vh = randn(rng,Float32,128,48).*0.1f0
    zh = randn(rng,Float32,128,48).*0.1f0
    ah = randn(rng,Float32,48).*0.1f0
    bh = randn(rng,Float32,48).*0.1f0
    dt = randn(rng,Float32,48).*0.1f0
    avec = .-exp.(randn(rng,Float32,48).*0.1f0)
    nw = randn(rng,Float32,128).*0.1f0
    sh = randn(rng,Float32,128,128,48).*0.01f0
    wantstate = copy(sh); want = zeros(Float32,128,48)
    qn = qh ./ sqrt.(sum(abs2,qh;dims=1).+1f-6)
    kn = kh ./ sqrt.(sum(abs2,kh;dims=1).+1f-6)
    for h in 1:48
        decay = exp((max(ah[h]+dt[h],0f0)+log1p(exp(-abs(ah[h]+dt[h]))))*avec[h])
        beta = 1f0/(1f0+exp(-bh[h])); keyh = (h-1)%16+1
        raw = zeros(Float32,128)
        for col in 1:128
            sk = sum(decay.*wantstate[:,col,h].*kn[:,keyh])
            delta = (vh[col,h]-sk)*beta
            wantstate[:,col,h] .= decay.*wantstate[:,col,h] .+ kn[:,keyh].*delta
            raw[col] = sum(wantstate[:,col,h].*qn[:,keyh])/sqrt(128f0)
        end
        raw ./= sqrt(sum(abs2,raw)/128f0+1f-6)
        want[:,h] .= raw.*nw.*(zh[:,h]./(1f0.+exp.(-zh[:,h])))
    end
    todev(x) = DNNKernels.toback(backend,x)
    state = todev(sh); out = KernelAbstractions.allocate(backend,Float32,128,48)
    gated_delta_net!(ctx,out,state,todev(qh),todev(kh),todev(vh),todev(ah),todev(bh),
                     todev(dt),todev(avec),todev(zh),todev(nw))
    KernelAbstractions.synchronize(backend)
    @test maximum(abs,Array(out).-want) < 3f-5
    @test maximum(abs,Array(state).-wantstate) < 3f-6

    channels=9; history=randn(rng,Float32,3,channels); input=randn(rng,Float32,channels)
    weight=randn(rng,Float32,4,channels); href=copy(history)
    cref=[sum(vcat(href[:,c],input[c]).*weight[:,c]) for c in 1:channels]
    cref=Float32.(cref./(1f0.+exp.(-cref))); href[1:2,:].=href[2:3,:]; href[3,:].=input
    dh=todev(history); dout=KernelAbstractions.allocate(backend,Float32,channels)
    depthwise_conv4!(ctx,dout,dh,todev(input),todev(weight)); KernelAbstractions.synchronize(backend)
    @test Array(dout) ≈ cref atol=2f-6
    @test Array(dh) ≈ href atol=1f-7
end

@testset "GGUF v3 directory and mapped dense tensor" begin
    mktemp() do path, io
        close(io); write_tiny_gguf(path)
        f = readgguf(path)
        @test f.version == 3
        @test f.metadata["test.name"] == "bonsai"
        @test f["x"].dims == (4,)
        @test collect(gguftensor(f, "x")) == Float32[1,2,3,4]
    end
end

function pack_ptq1(x::AbstractMatrix{Float32})
    M, K = size(x); K % 128 == 0 || error()
    out = UInt8[]; quant = similar(x)
    for m in 1:M, b in 0:K÷128-1
        block = view(x, m, b*128+1:(b+1)*128)
        d = maximum(abs, block)
        trits = Int8.(round.(Int, clamp.(block ./ d, -1, 1)))
        scale = Float32(Float16(d)); quant[m,b*128+1:(b+1)*128] .= Float32.(trits).*scale
        qs = zeros(UInt8, 24)
        for j in 0:15
            q = sum((Int(trits[n*16+j+1])+1)*3^(4-n) for n in 0:4)
            qs[j+1] = UInt8(cld(q*256,243))
        end
        for j in 0:7
            q = sum((Int(trits[80+n*8+j+1])+1)*3^(4-n) for n in 0:4)
            qs[17+j] = UInt8(cld(q*256,243))
        end
        qh = zeros(UInt8,2)
        for j in 0:1
            q = sum((Int(trits[120+n*2+j+1])+1)*3^(3-n) for n in 0:3)
            qh[j+1] = UInt8(cld(3q*256,243))
        end
        append!(out,qs); append!(out,qh)
        bits = reinterpret(UInt16,Float16(d)); push!(out,UInt8(bits&0xff),UInt8(bits>>8))
    end
    out, quant
end

# The device layout of `PTQ1Matrix`, on the host: word q of block b of the 64 rows of
# a tile together, rows past M zero.
function tile_ptq1(bytes::Vector{UInt8}, M, K)
    blocks = K ÷ 128
    w = reinterpret(UInt32, bytes)
    out = zeros(UInt32, cld(M, 64) * 64 * blocks * 7)
    for row in 0:M-1, b in 0:blocks-1, q in 0:6
        out[((row ÷ 64 * blocks + b) * 7 + q) * 64 + row % 64 + 1] = w[(row * blocks + b) * 7 + q + 1]
    end
    out
end

function fwht_reference(x, signs; inverse=false)
    y = Float32.(x)
    inverse || (y .*= signs)
    for base in 1:1024:length(y), stride in (1,2,4,8,16,32,64,128,256,512)
        for group in 0:2stride:1023, j in 0:stride-1
            a = y[base+group+j]; b = y[base+group+j+stride]
            y[base+group+j] = a+b; y[base+group+j+stride] = a-b
        end
    end
    y ./= 32
    inverse && (y .*= signs)
    y
end

@testset "PTQ1 packed kernels and signed Hadamard" begin
    backend = Mantle.defaultbackend(); ctx = DNNKernels.Ctx(backend)
    # Five columns exercises one full four-column PTQ prefill tile and its tail.
    rng = MersenneTwister(91); M,K,N = 7,256,5
    bytes, W = pack_ptq1(randn(rng,Float32,M,K))
    A = ptq1matrix(backend,bytes,M,K)
    xh = randn(rng,Float32,K,N); x = DNNKernels.toback(backend,xh)
    out = KernelAbstractions.allocate(backend,Float32,M,N)
    ptq1mul!(ctx,out,A,x); KernelAbstractions.synchronize(backend)
    @test maximum(abs,Array(out).-W*xh) < 2f-4
    @test maximum(abs,Array(ptq1_dequant(ctx,A)).-W) < 1f-6
    # The layout every kernel reads, pinned: a tile of 64 rows, word-major, padded.
    for (MT, KT) in ((7, 256), (130, 384))
        bt, _ = pack_ptq1(randn(rng, Float32, MT, KT))
        At = ptq1matrix(backend, bt, MT, KT)
        @test collect(reinterpret(UInt32, Array(Mantle.storage(At.data)))) == tile_ptq1(bt, MT, KT)
    end
    # The decode column, `ptq1_mul_rows_kernel!`, which is most of a Bonsai token
    # and which five columns never reach; and eight, the tiled prefill. K = 5120
    # is Bonsai's hidden width, forty blocks over a tile's parts, so some parts
    # take one block more than the rest. Seven and 3201 rows leave a partial tile,
    # and 7, 3201 and 12800 rows are 16, 8 and 4 parts of the column kernel
    # (`ptq1_parts`). 65 blocks double-buffer the decode column's staged `x`, in
    # rounds that do not divide them.
    @test DNNKernels.ptq1_parts.((7, 3201, 12800)) == (16, 8, 4)
    @test DNNKernels.ptq1_rows_launch(5, 8320)[4] === Val(true)
    @test DNNKernels.ptq1_rows_launch(5, 5120)[4] === Val(false)
    for (MK, KK) in ((7, 256), (9, 5120), (3201, 512), (12800, 256), (5, 8320)), NN in (1, 8)
        bk, Wk = pack_ptq1(randn(rng, Float32, MK, KK))
        Ak = ptq1matrix(backend, bk, MK, KK)
        xk = randn(rng, Float32, KK, NN)
        outk = KernelAbstractions.allocate(backend, Float32, MK, NN)
        ptq1mul!(ctx, outk, Ak, DNNKernels.toback(backend, xk))
        KernelAbstractions.synchronize(backend)
        @test maximum(abs, Array(outk) .- Wk * xk) < 2f-4 * sqrt(KK / 256)
        if NN == 1
            bias = randn(rng, Float32, MK)
            ptq1mul!(ctx, outk, Ak, DNNKernels.toback(backend, xk), DNNKernels.toback(backend, bias))
            KernelAbstractions.synchronize(backend)
            @test maximum(abs, Array(outk) .- (Wk * xk .+ bias)) < 2f-4 * sqrt(KK / 256)
        end
    end
    # Two decode columns over one `x` in one launch (Bonsai's FFN gate and up): the
    # first matrix's tiles, then the second's, both ending in a partial tile, at a
    # width that double-buffers and one that does not.
    dev = Mantle.todevice(backend)
    for KK in (512, 8320)
        b1, W1 = pack_ptq1(randn(rng, Float32, 70, KK)); b2, W2 = pack_ptq1(randn(rng, Float32, 130, KK))
        A1, A2 = ptq1matrix(backend, b1, 70, KK), ptq1matrix(backend, b2, 130, KK)
        xk = randn(rng, Float32, KK)
        xd = Mantle.Buffer(dev, xk)
        o1, o2 = Mantle.Buffer(dev, fill(NaN32, 70)), Mantle.Buffer(dev, fill(NaN32, 130))
        g = Mantle.Graph(dev)
        DNNKernels.ptq1_rows2!(g, o1, A1, o2, A2, xd; name = "rows2")
        Mantle.runonce!(g)
        tol = 2f-4 * sqrt(KK / 256)
        @test maximum(abs, Array(Mantle.storage(o1)) .- W1 * xk) < tol
        @test maximum(abs, Array(Mantle.storage(o2)) .- W2 * xk) < tol
        foreach(Mantle.free!, (o1, o2, xd, A1.data, A2.data))
    end
    rows = DNNKernels.toback(backend,Int32[0,6,2]); emb = KernelAbstractions.allocate(backend,Float32,K,3)
    ptq1_getrows!(ctx,emb,A,rows); KernelAbstractions.synchronize(backend)
    @test Array(emb) ≈ permutedims(W[[1,7,3],:]) atol=1f-6

    # Asked as a CAPABILITY. Naming the backend made this an UndefVarError on a
    # build that compiled in a different one, which took the whole file down
    # with it rather than skipping one arm.
    if isdefined(Mantle, :vk_context) &&
       Mantle.coopmat_gemm_available(Mantle.vk_context())
        # The wide-prefill path decodes PTQ1 directly into cooperative-matrix
        # shared tiles.  Exercise one complete 128x128 output block, including
        # every byte/trit layout in two consecutive 128-value weight blocks.
        MW, KW, NW = 128, 256, 128
        widebytes, wideW = pack_ptq1(randn(rng, Float32, MW, KW))
        wideA = ptq1matrix(backend, widebytes, MW, KW)
        wideXh = Float16.(randn(rng, Float32, KW, NW))
        wideX = DNNKernels.toback(backend, wideXh)
        wideout = KernelAbstractions.allocate(backend, Float32, MW, NW)
        KI.Kernel(backend, DNNKernels.ptq1_coopmat_kernel!)(
            wideout, wideA.data, wideX, Val(MW), Val(NW), Val(KW);
            ndrange=DNNKernels.PTQ1_COOP_WG, workgroupsize = DNNKernels.PTQ1_COOP_WG)
        KernelAbstractions.synchronize(backend)
        @test Array(wideout) ≈ wideW * Float32.(wideXh) rtol=2f-4 atol=2f-3
    end

    h = randn(rng,Float32,2048); signs = rand(rng,Float32[-1,1],2048)
    dh = DNNKernels.toback(backend,h); ds = DNNKernels.toback(backend,signs)
    hadamard!(ctx,dh,ds); KernelAbstractions.synchronize(backend)
    @test Array(dh) ≈ fwht_reference(h,signs) atol=2f-5
end

# The prefill route on a device without the staged cooperative-matrix kernels: a
# Float16 expansion and the device's own GEMM for the multiples of 32 columns, and
# the shared-decode column kernel for the rest. On Metal the GEMM half replaced
# `ptq1_mul_mm_kernel!`, which made a 512-token Bonsai prefill take 94 s. 45
# columns is 32 through the GEMM and 13 through `ptq1_cols!` (four launches, the
# last of one column). Both operands are flat resources, as Bonsai's
# batch scratch is, so the views taken over them are part of what is pinned.
@testset "PTQ1 product through the device's native GEMM and the column kernel" begin
    backend = Mantle.defaultbackend(); dev = Mantle.todevice(backend)
    rng = MersenneTwister(23); M, K, N = 96, 384, 45
    bytes, W = pack_ptq1(randn(rng, Float32, M, K))
    A = ptq1matrix(backend, bytes, M, K)
    xh = Float16.(randn(rng, Float32, K, N))
    x = Mantle.Buffer(dev, vec(xh))
    out = Mantle.Buffer(dev, fill(NaN32, M * N))
    g = Mantle.Graph(dev)
    covered = DNNKernels.ptq1_native_gemm!(g, out, A, x, N; name = "ptq1_native")
    @test covered == (Mantle.native_gemm_available(dev, Float16, Float16, Float32) ? 32 : 0)
    DNNKernels.ptq1_cols!(g, out, A, x, covered, N; name = "ptq1_cols")
    Mantle.runonce!(g)
    want = W * Float32.(xh)
    got = reshape(Array(Mantle.storage(out)), M, N)
    @test !any(isnan, got)
    @test maximum(abs, got .- want) / maximum(abs, want) < 1f-5
    # The expansion alone, which `ptq1_dequant` shares, against the packing.
    @test Array(ptq1_dequant(DNNKernels.Ctx(backend), A)) == W
end
