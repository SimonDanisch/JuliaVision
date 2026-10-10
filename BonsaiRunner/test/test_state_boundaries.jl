using Test, Random, LinearAlgebra, BonsaiRunner, DNNKernels
import Mantle

@testset "recurrent state across prompt chunks" begin
    dev = Mantle.Device()
    rng = MersenneTwister(91)
    function runpass(kernel, args, n; group)
        g = Mantle.Graph(dev)
        Mantle.dispatch!(g, kernel, args, n; group)
        p = Mantle.record!(Mantle.Plan(g))
        try
            Mantle.run!(p)
        finally
            Mantle.free!(p)
        end
    end

    @testset "16-token cooperative gate projection" begin
        # The gate `_recurrent_batch!` takes this path under. `caps.coopmat` alone
        # is true on Metal, whose 8x8 matrices the staged GEMM is not written at.
        if DNNKernels.coopmatkernels(Mantle.caps(dev))
            for nt in (16,32)
                weight = randn(rng,Float16,96,64) .* Float16(0.1)
                x = randn(rng,Float16,64,nt)
                w, input = Mantle.Buffer(dev,weight), Mantle.Buffer(dev,x)
                out = Mantle.Buffer(dev,Float32,(96,nt))
                g = Mantle.Graph(dev)
                BonsaiRunner._dense_coop_batch!(g,out,w,input,64,96,nt,"gate regression")
                p = Mantle.record!(Mantle.Plan(g))
                try
                    Mantle.run!(p)
                    @test Array(out) ≈ Float32.(weight)*Float32.(x) rtol=2e-3 atol=2e-4
                finally
                    Mantle.free!(p)
                    foreach(Mantle.free!,(w,input,out))
                end
            end
        else
            @test_skip false
        end
    end

    @testset "causal convolution history" begin
        channels, nt = 37, 65
        x = randn(rng, Float32, channels, nt)
        w = randn(rng, Float32, 4, channels) .* 0.1f0
        initial = randn(rng, Float32, 3, channels) .* 0.1f0
        state = copy(initial)
        want = similar(x)
        for t in 1:nt, c in 1:channels
            y = dot(w[:,c], [state[:,c]; x[c,t]])
            want[c,t] = y / (1f0 + exp(-y))
            state[:,c] = [state[2,c], state[3,c], x[c,t]]
        end
        for lengths in ((65,), (1,2,3,1,8,16,32,2), ntuple(_->1,nt))
            s, weight = Mantle.Buffer(dev, initial), Mantle.Buffer(dev, w)
            got = zeros(Float32, channels, nt)
            first = 1
            for n in lengths
                input = Mantle.Buffer(dev, x[:,first:first+n-1])
                out = Mantle.Buffer(dev, Float32, (channels,n))
                if n == 1
                    runpass(DNNKernels.depthwise_conv4_kernel!,
                        (out,s,input,weight,Int32(channels)),channels;group=256)
                else
                    runpass(BonsaiRunner.depthwise_conv4_batch_kernel!,
                        (out,s,input,weight,Int32(channels),Int32(n)),channels;group=256)
                end
                got[:,first:first+n-1] = Array(out)
                foreach(Mantle.free!, (input,out))
                first += n
            end
            @test got ≈ want rtol=2e-6 atol=2e-7
            @test Array(s) == state
            foreach(Mantle.free!, (s,weight))
        end
    end

    @testset "DeltaNet continuation and restored state" begin
        nt = 17
        q = randn(rng, Float32, 128,16,nt)
        k = randn(rng, Float32, 128,16,nt)
        rawq,rawk = copy(q),copy(k)
        q ./= sqrt.(sum(abs2,q;dims=1) .+ 1f-6)
        k ./= sqrt.(sum(abs2,k;dims=1) .+ 1f-6)
        v = randn(rng, Float32, 128,48,nt) .* 0.1f0
        alpha, beta = randn(rng, Float32, 48,nt), randn(rng, Float32, 48,nt)
        dt, a = randn(rng, Float32,48) .* 0.1f0, fill(-0.7f0,48)
        initial = randn(rng, Float32,128,128,48) .* 0.01f0
        wantstate, want = copy(initial), similar(v)
        for t in 1:nt, h in 1:48
            kh = (h-1)%16+1
            decay = exp(DNNKernels._softplus_f32(alpha[h,t]+dt[h])*a[h])
            s = view(wantstate,:,:,h)
            s .*= decay
            delta = (v[:,h,t] - transpose(s)*k[:,kh,t]) .* DNNKernels._sigmoid_f32(beta[h,t])
            s .+= k[:,kh,t] * transpose(delta)
            want[:,h,t] = transpose(s)*q[:,kh,t] ./ sqrt(128f0)
            y = view(want,:,h,t)
            y .*= DNNKernels._sigmoid_f32(1f0) / sqrt(sum(abs2,y)/128f0+1f-6)
        end
        subgroup = Mantle.caps(dev).subgroup
        for lengths in ((17,), (7,1,1,7,1), ntuple(_->1,nt))
            s, db, ab = Mantle.Buffer(dev,initial), Mantle.Buffer(dev,dt), Mantle.Buffer(dev,a)
            nw = Mantle.Buffer(dev,ones(Float32,128))
            got = similar(v)
            first = 1
            for n in lengths
                span = first:first+n-1
                inputs = map(x->Mantle.Buffer(dev,x),
                    ((n==1 ? rawq : q)[:,:,span],(n==1 ? rawk : k)[:,:,span],
                     v[:,:,span],alpha[:,span],beta[:,span]))
                out = Mantle.Buffer(dev,Float32,(128,48,n))
                z = Mantle.Buffer(dev,ones(Float32,128,48,n))
                if n == 1
                    L = DNNKernels.GDN_LANES
                    runpass(DNNKernels.gated_delta_net_kernel!,
                        (out,s,inputs...,db,ab,z,nw,1f-6,Int32(48),Int32(16),Val(L)),
                        48*128*L;group=128*L)
                else
                    runpass(BonsaiRunner.gated_delta_state_batch_kernel!,
                        (out,s,inputs...,db,ab,Int32(n),Val(subgroup)),
                        48*(128 ÷ BonsaiRunner.GDN_COLS)*subgroup;group=256)
                    runpass(BonsaiRunner.gdn_norm_gate_batch_kernel!,
                        (out,z,nw,1f-6),48n*128;group=128)
                end
                got[:,:,span] = Array(out)
                # Round-trip the live state at every continuation boundary.
                saved = Array(s)
                Mantle.free!(s)
                s = Mantle.Buffer(dev,saved)
                foreach(Mantle.free!, (inputs...,out,z))
                first += n
            end
            @test got ≈ want rtol=3e-5 atol=2e-7
            @test Array(s) ≈ wantstate rtol=3e-5 atol=2e-7
            foreach(Mantle.free!, (s,db,ab,nw))
        end
    end

    @testset "Q8 KV writes across position boundaries" begin
        nt, capacity = 17, 96
        keys = randn(rng,Float32,256,4,nt)
        values = randn(rng,Float32,256,4,nt)
        norm = rand(rng,Float32,256) .+ 0.5f0
        nb = Mantle.Buffer(dev,norm)
        for start in (31,63)
            wantk = fill(Int8(-101),1024,capacity)
            wantv = copy(wantk)
            wantks, wantvs = fill(-1f0,4,capacity), fill(-1f0,4,capacity)
            for t in 1:nt, h in 1:4
                key = keys[:,h,t] .* norm ./ sqrt(sum(abs2,keys[:,h,t])/256f0+1f-6)
                pos = start+t-1
                for d in 0:31
                    angle = Float32(pos)*exp(-log(10000f0)*Float32(2d)/64f0)
                    c,s = cos(angle),sin(angle)
                    x,y = key[d+1],key[d+33]
                    key[d+1],key[d+33] = x*c-y*s,x*s+y*c
                end
                ks = maximum(abs,key)/127f0
                vs = maximum(abs,values[:,h,t])/127f0
                rows = (h-1)*256+1:h*256
                wantk[rows,pos+1] = Int8.(clamp.(round.(key./ks),-127f0,127f0))
                wantv[rows,pos+1] = Int8.(clamp.(round.(values[:,h,t]./vs),-127f0,127f0))
                wantks[h,pos+1],wantvs[h,pos+1] = ks,vs
            end
            baseline = nothing
            for lengths in ((17,),(1,2,5,8,1),ntuple(_->1,nt))
                kc,vc = Mantle.Buffer(dev,fill(Int8(-101),1024,capacity)),Mantle.Buffer(dev,fill(Int8(-101),1024,capacity))
                ks,vs = Mantle.Buffer(dev,fill(-1f0,4,capacity)),Mantle.Buffer(dev,fill(-1f0,4,capacity))
                first = 1
                for n in lengths
                    kb,vb = Mantle.Buffer(dev,keys[:,:,first:first+n-1]),Mantle.Buffer(dev,values[:,:,first:first+n-1])
                    position = Mantle.Buffer(dev,Int32[start+first-1])
                    kernel = n==1 ? BonsaiRunner.prepare_kv_kernel! : BonsaiRunner.prepare_kv_batch_kernel!
                    runpass(kernel,
                        (kc,vc,ks,vs,kb,vb,nb,position,10000f0,1f-6),4n*256;group=256)
                    foreach(Mantle.free!,(kb,vb,position))
                    first += n
                end
                got = map(Array,(kc,vc,ks,vs))
                baseline === nothing ? (baseline=got) : (@test got == baseline)
                @test got[1] == wantk
                @test got[2] == wantv
                @test got[3] ≈ wantks rtol=2e-6
                @test got[4] ≈ wantvs rtol=2e-6
                foreach(Mantle.free!,(kc,vc,ks,vs))
            end
        end
        Mantle.free!(nb)
    end
end
