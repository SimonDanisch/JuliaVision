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
    backend = Mantle.LavaBackend(); ctx = DNNKernels.Ctx(backend)
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
    backend = Mantle.LavaBackend(); ctx = DNNKernels.Ctx(backend)
    # Five columns exercises one full four-column PTQ prefill tile and its tail.
    rng = MersenneTwister(91); M,K,N = 7,256,5
    bytes, W = pack_ptq1(randn(rng,Float32,M,K))
    db = KernelAbstractions.allocate(backend,UInt8,length(bytes)); copyto!(db,bytes)
    A = PTQ1Matrix(db,M,K)
    xh = randn(rng,Float32,K,N); x = DNNKernels.toback(backend,xh)
    out = KernelAbstractions.allocate(backend,Float32,M,N)
    ptq1mul!(ctx,out,A,x); KernelAbstractions.synchronize(backend)
    @test maximum(abs,Array(out).-W*xh) < 2f-4
    @test maximum(abs,Array(ptq1_dequant(ctx,A)).-W) < 1f-6
    rows = DNNKernels.toback(backend,Int32[0,6,2]); emb = KernelAbstractions.allocate(backend,Float32,K,3)
    ptq1_getrows!(ctx,emb,A,rows); KernelAbstractions.synchronize(backend)
    @test Array(emb) ≈ permutedims(W[[1,7,3],:]) atol=1f-6

    if backend isa Mantle.LavaBackend &&
       Mantle.coopmat_gemm_available(Mantle.vk_context())
        # The wide-prefill path decodes PTQ1 directly into cooperative-matrix
        # shared tiles.  Exercise one complete 128x128 output block, including
        # every byte/trit layout in two consecutive 128-value weight blocks.
        MW, KW, NW = 128, 256, 128
        widebytes, wideW = pack_ptq1(randn(rng, Float32, MW, KW))
        wdb = KernelAbstractions.allocate(backend, UInt8, length(widebytes))
        copyto!(wdb, widebytes)
        wideA = PTQ1Matrix(wdb, MW, KW)
        wideXh = Float16.(randn(rng, Float32, KW, NW))
        wideX = DNNKernels.toback(backend, wideXh)
        wideout = KernelAbstractions.allocate(backend, Float32, MW, NW)
        DNNKernels.ptq1_coopmat_kernel!(backend, DNNKernels.PTQ1_COOP_WG)(
            wideout, wideA.data, wideX, Val(MW), Val(NW), Val(KW);
            ndrange=DNNKernels.PTQ1_COOP_WG)
        KernelAbstractions.synchronize(backend)
        @test Array(wideout) ≈ wideW * Float32.(wideXh) rtol=2f-4 atol=2f-3
    end

    h = randn(rng,Float32,2048); signs = rand(rng,Float32[-1,1],2048)
    dh = DNNKernels.toback(backend,h); ds = DNNKernels.toback(backend,signs)
    hadamard!(ctx,dh,ds); KernelAbstractions.synchronize(backend)
    @test Array(dh) ≈ fwht_reference(h,signs) atol=2f-5
end
