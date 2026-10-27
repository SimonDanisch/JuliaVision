using Test,Random,LinearAlgebra,JSON3
import Mantle,KernelAbstractions

function test_qwen35_moe(device;fixture=nothing)
    M=Qwen35MoE;rng=MersenneTwister(3523);rows=Any[]
    @testset "Portable MoE routing and shared-expert combine" begin
        for (ne,k) in ((1,1),(8,2),(16,16),(256,8))
            owned=Any[]
            upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
            plan=nothing
            try
                width=17;tokens=5
                logits=randn(rng,Float32,ne,tokens)
                logits[:,2].=0f0 # deterministic tie policy
                logits[:,3].=ne==1 ? [1000f0] : range(-1000f0,1000f0;length=ne)
                logits[:,4].=NaN32;logits[:,5].=Inf32
                routed=randn(rng,Float32,width,k,tokens);shared=randn(rng,Float32,width,tokens)
                gates=Float32[-1000,0,1000,1,-1]
                src=upload(logits);sel=upload(fill(Int32(-7),k,tokens));weights=upload(fill(NaN32,k,tokens))
                status=upload(fill(Int32(-7),tokens));out=upload(fill(NaN32,width,tokens))
                r,s,g=upload(routed),upload(shared),upload(gates)
                graph=Mantle.Graph(device)
                M.declare_router!(graph,sel,weights,status,src)
                M.declare_combine!(graph,out,r,s,g,weights,status)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                for replay in 1:2
                    before=Mantle.submissions(device)
                    Mantle.run!(plan);KernelAbstractions.synchronize(device)
                    @test Mantle.submissions(device)-before==1
                    ids=Array(sel);probs=Array(weights);codes=Array(status);got=Array(out)
                    @test codes==Int32[0,0,0,1,1]
                    for token in 1:3
                        # Float32 softmax is intentional: underflow can tie
                        # probabilities even when the original logits differ.
                        p=exp.(logits[:,token].-maximum(logits[:,token]));p./=sum(p)
                        chosen=sortperm(1:ne;by=i->(-p[i],i))[1:k]
                        expected=p[chosen]./sum(p[chosen])
                        @test ids[:,token]==chosen.-1
                        @test probs[:,token] ≈ expected atol=3f-6 rtol=3f-5
                        @test sum(probs[:,token]) ≈ 1f0 atol=3f-6
                        sig=1/(1+exp(-Float64(gates[token])))
                        want=routed[:,:,token]*expected+sig*shared[:,token]
                        @test got[:,token] ≈ want atol=3f-6 rtol=3f-5
                    end
                    @test all(==(-1),ids[:,4:5])
                    @test all(iszero,probs[:,4:5])
                    @test all(isnan,got[:,4:5])
                    push!(rows,(;experts=ne,topk=k,replay))
                    # Reuse the same recording after a queued logits update.
                    logits[:,1].=reverse(logits[:,1]);src[:]=vec(logits)
                end
            finally
                plan===nothing || Mantle.free!(plan)
                foreach(Mantle.free!,owned)
            end
        end
    end
    fixture===nothing || test_qwen35_moe_fixture(device,fixture)
    fixture===nothing || test_qwen35_moe_block_fixture(device,fixture)
    rows
end

function test_qwen35_moe_block_fixture(device,path)
    @testset "Native Transformers complete MoE block" begin
        c=JSON3.read(read(path,String)).block
        mat(rows)=hcat([Float32.(collect(row)) for row in rows]...)
        stack(experts)=cat([mat(expert) for expert in experts]...;dims=3)
        owned=Any[];plan=nothing
        upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
        try
            W=(;router=upload(mat(c.weights.router)),gate_up=upload(stack(c.weights.gate_up)),
                down=upload(stack(c.weights.down)),shared_gate_up=upload(mat(c.weights.shared_gate_up)),
                shared_down=upload(mat(c.weights.shared_down)),shared_gate=upload(mat(c.weights.shared_gate)))
            x=upload(mat(c.inputs[1]));width,tokens=size(x);topk=Int(c.topk)
            out=upload(fill(NaN32,width,tokens));selected=upload(zeros(Int32,topk,tokens))
            weights=upload(zeros(Float32,topk,tokens));status=upload(zeros(Int32,tokens))
            g=Mantle.Graph(device)
            Qwen35MoE.declare_block!(g,out,x,W,selected,weights,status)
            plan=Mantle.record!(Mantle.Plan(g;budget=Inf))
            for i in eachindex(c.inputs)
                x[:]=vec(mat(c.inputs[i]))
                before=Mantle.submissions(device)
                Mantle.run!(plan);KernelAbstractions.synchronize(device)
                @test Mantle.submissions(device)-before==1
                @test all(iszero,Array(status))
                @test Array(out) ≈ mat(c.outputs[i]) atol=3f-6 rtol=3f-5
            end
            # NaN poison in unselected experts must not reach the output.
            used=Set(Int.(Array(selected)).+1)
            for (buffer,host) in ((W.gate_up,stack(c.weights.gate_up)),(W.down,stack(c.weights.down)))
                for expert in 1:size(host,3)
                    expert in used || (host[:,:,expert].=NaN32)
                end
                buffer[:]=vec(host)
            end
            Mantle.run!(plan)
            @test Array(out) ≈ mat(last(c.outputs)) atol=3f-6 rtol=3f-5
            bad=mat(last(c.inputs));bad[:,2].=NaN32;x[:]=vec(bad)
            Mantle.run!(plan)
            @test Array(status)==Int32[0,1,0]
            @test all(==(-1),Array(selected)[:,2])
            @test all(isnan,Array(out)[:,2])
        finally
            plan===nothing || Mantle.free!(plan)
            foreach(Mantle.free!,owned)
        end
    end
end

function test_qwen35_dense_projection(device)
    rng=MersenneTwister(3527)
    @testset "Dense projection sums every subgroup" begin
        for k in (16,65,128)
            owned=Any[]
            upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
            try
                Wh=randn(rng,Float32,k,17);xh=randn(rng,Float32,k,3)
                W=upload(Wh);x=upload(xh);out=upload(fill(NaN32,17,3))
                g=Mantle.Graph(device);Qwen35MoE.declare_dense!(g,out,W,x;name="test.dense")
                Mantle.runonce!(g)
                @test Array(out) ≈ Float64.(Wh)'*Float64.(xh) atol=6e-6 rtol=3e-6
            finally
                foreach(Mantle.free!,owned)
            end
        end
    end
end

function test_qwen35_moe_fixture(device,path)
    @testset "Native Transformers router fixture (strict logits)" begin
        fixture=JSON3.read(read(path,String))
        for case in fixture.cases
            ne=Int(case.experts);k=Int(case.topk);tokens=length(case.logits)
            logits=hcat([Float32.(collect(row)) for row in case.logits]...)
            owned=Any[]
            upload(x)=begin a=Mantle.Buffer(device,x);push!(owned,a);a;end
            try
                src=upload(logits);sel=upload(zeros(Int32,k,tokens));weights=upload(zeros(Float32,k,tokens))
                status=upload(zeros(Int32,tokens))
                g=Mantle.Graph(device);Qwen35MoE.declare_router!(g,sel,weights,status,src)
                Mantle.runonce!(g)
                @test Array(sel)==hcat([Int32.(collect(row)) for row in case.selected]...)
                @test Array(weights) ≈ hcat([Float32.(collect(row)) for row in case.weights]...) atol=3f-6 rtol=3f-5
                @test all(iszero,Array(status))
            finally
                foreach(Mantle.free!,owned)
            end
        end
    end
end
