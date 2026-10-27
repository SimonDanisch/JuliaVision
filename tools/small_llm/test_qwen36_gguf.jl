using Test,Random,SHA,JSON3

function test_qwen36_gguf_host()
    G=Qwen36GGUF;DK=G.DK
    @testset "Lossless GGUF packed layout conversion" begin
        rng=MersenneTwister(365)
        bytes=rand(rng,UInt8,210*13)
        planes=G.q6_planes(bytes)
        @test DK.q6k_ggml(planes...)==bytes
        @test_throws DimensionMismatch G.q6_planes(UInt8[1])
        k,rows,ne=256,3,4;each=144rows
        gate=rand(rng,UInt8,each*ne);up=rand(rng,UInt8,each*ne)
        combined=G.combine_expert_rows(gate,up,k,rows,ne)
        for e in 0:ne-1
            @test combined[2each*e.+(1:each)]==gate[each*e.+(1:each)]
            @test combined[2each*e+each.+(1:each)]==up[each*e.+(1:each)]
        end
        @test_throws DimensionMismatch G.combine_expert_rows(gate,up[2:end],k,rows,ne)
    end
end

function test_qwen36_gguf_kernels(device)
    G=Qwen36GGUF;R=G.R;M=G.M;Q=G.Q;DK=G.DK
    folder=joinpath(@__DIR__,"gguf_q5_reference")
    reference=JSON3.read(read(joinpath(folder,"reference.json"),String))
    @assert bytes2hex(open(sha256,joinpath(folder,"reference.safetensors")))==reference.tensor_sha256
    tensors=DK.readsafetensors(joinpath(folder,"reference.safetensors"))
    owned=Any[];plan=nothing
    upload(x)=begin b=Mantle.Buffer(device,x);push!(owned,b);b;end
    @testset "Packed Q5 dense/routed and Q4 embedding vs independent oracle" begin
        for k in (256,512,1024)
            try
                rows,ne,topk,tokens=17,4,2,3
                raw=tensors["q5.$k.bytes"];packed=upload(reinterpret(DK.UVec4,vec(raw)))
                want=tensors["q5.$k.float"];input=tensors["q5.$k.input"]
                # First expert for dense, first slot of first token for input.
                x=copy(input[:,1,1]);xb=upload(x);out=upload(fill(NaN32,rows))
                q8=upload(fill(ntuple(_->VecElement(UInt32(0)),4),k÷16));ds=upload(zeros(Float32,k÷16))
                graph=Mantle.Graph(device)
                Q.linear!(graph,out,G.Q5KMatrix(packed,rows,k),xb;name="q5_dense")
                Q.dispatch(graph,DK.q8act_kernel!,(q8,ds,xb,Int32(k),Int32(k÷32)),k÷32;name="q8_oracle_capture")
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));Mantle.waitidle(device)
                before=Mantle.submissions(device)
                Mantle.run!(plan);Mantle.waitidle(device)
                @test Mantle.submissions(device)-before==1
                activation=Float32.(reinterpret(Int8,vec(Array(q8))))
                scales=vec(Array(ds))
                for i in eachindex(activation);activation[i]*=scales[2*((i-1)÷32)+1];end
                expected=transpose(want[:,1:rows])*activation
                @test Array(out)≈expected atol=3f-4 rtol=3f-5
                Mantle.free!(plan);plan=nothing
                for per_expert in (false,true)
                    sel=Int32[0 3 1;2 1 2];status=Int32[0,0,1]
                    selected=upload(sel);statusb=upload(status)
                    inputs=per_expert ? input : copy(input[:,1,:])
                    values=upload(inputs);result=upload(fill(NaN32,rows,topk,tokens))
                    graph=Mantle.Graph(device)
                    M.declare_routed_linear!(graph,result,G.Q5Experts(G.Q5KMatrix(packed,rows*ne,k),rows,ne),
                        values,selected,statusb;per_expert,name="q5_routed")
                    plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));Mantle.waitidle(device)
                    before=Mantle.submissions(device)
                    Mantle.run!(plan);Mantle.waitidle(device)
                    @test Mantle.submissions(device)-before==1
                    actual=Array(result)
                    for token in 1:2,slot in 1:topk
                        x=per_expert ? inputs[:,slot,token] : inputs[:,token]
                        # Approximation bound here includes Q8 activation rounding;
                        # the dense check above isolates exact packed-weight math.
                        first=Int(sel[slot,token])*rows+1
                        expected=transpose(want[:,first:first+rows-1])*x
                        @test actual[:,slot,token]≈expected atol=0.3f0 rtol=0.02f0
                    end
                    @test all(iszero,actual[:,:,3])
                    Mantle.free!(plan);plan=nothing
                end
            finally
                plan===nothing||Mantle.free!(plan);plan=nothing
                foreach(Mantle.free!,reverse(owned));empty!(owned)
            end
        end
        try
            rng=MersenneTwister(365);width=512;vocab=17
            dense=randn(rng,Float32,vocab,width);quant=DK.q4k_quantize(dense)
            expected=DK.q4k_dequantize(quant,vocab,width)
            packed=upload(quant);out=upload(zeros(Float32,width))
            token=upload(Int32[0])
            for value in (0,1,vocab-1)
                copyto!(token,Int32[value]);graph=Mantle.Graph(device)
                R.declare_embedding!(graph,out,DK.Q4KMatrix(packed,vocab,width),token,width)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf));Mantle.waitidle(device)
                before=Mantle.submissions(device)
                Mantle.run!(plan);Mantle.waitidle(device)
                @test Mantle.submissions(device)-before==1
                @test Array(out)≈expected[value+1,:] atol=1f-6 rtol=1f-6
                Mantle.free!(plan);plan=nothing
            end
        finally
            plan===nothing||Mantle.free!(plan)
            foreach(Mantle.free!,reverse(owned))
        end
    end
end

function test_qwen36_tiled_layout(device)
    R=Qwen36MoEExperiment;Q=R.Q
    folder=joinpath(@__DIR__,"qwen36_moe_tiny_reference")
    model=load_qwen36_moe_fixture(R,folder,device);session=nothing
    try
        c=model.config;nh=Int(c.linear_num_value_heads);nkh=Int(c.linear_num_key_heads);ratio=nh÷nkh
        perm=vec(permutedims(reshape(collect(1:128nh),128,ratio,nkh),(1,3,2)))
        headperm=vec(permutedims(reshape(collect(1:nh),ratio,nkh)))
        for il in 0:Int(c.num_hidden_layers)-1
            c.layer_types[il+1]=="linear_attention"||continue
            p="layers.$il.linear_attn."
            for suffix in ("in_proj_qkv.weight","in_proj_z.weight","in_proj_a.weight","in_proj_b.weight","out_proj.weight","conv1d.weight","A_log","dt_bias")
                name=p*suffix;weight=model.weights[name]
                raw=weight isa Q.DenseWeight ? Array(weight.data) : Array(weight)
                converted=if suffix=="in_proj_qkv.weight"
                    raw[:,vcat(collect(1:256nkh),256nkh.+perm)]
                elseif suffix=="in_proj_z.weight"
                    raw[:,perm]
                elseif suffix in ("in_proj_a.weight","in_proj_b.weight")
                    raw[:,headperm]
                elseif suffix=="out_proj.weight"
                    raw[perm,:]
                elseif suffix=="conv1d.weight"
                    raw[:,:,vcat(collect(1:256nkh),256nkh.+perm)]
                else
                    raw[headperm]
                end
                buffer=Mantle.Buffer(device,converted);push!(model.owned,buffer)
                model.weights[name]=weight isa Q.DenseWeight ? Q.DenseWeight(buffer,weight.m,weight.k) : buffer
            end
        end
        @testset "Converted tiled recurrent layout vs native Transformers" begin
            session=R.session(model;capacity=32,head_layout=:tiled)
            reference=JSON3.read(read(joinpath(folder,"reference.json"),String))
            for row in reference.rows
                R.step!(session,Int(row.token));Mantle.waitidle(device)
                @test Array(session.logits)≈Float32.(row.logits) atol=1f-4 rtol=2f-4
            end
        end
    finally
        session===nothing||close(session);close(model)
    end
end
