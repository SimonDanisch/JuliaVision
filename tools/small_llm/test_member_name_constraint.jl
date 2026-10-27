# The member-name mask against its host reference, and against the kernels it
# has to leave alone. The host qualification is what the study relies on: every
# reference solution passes token by token, and a nonexistent name is refused,
# including when the trigger and the name share one token (`.atomic`): the
# first version constrained only tokens STARTING at a member position, and the
# real Qwen3.6 candidate `Atomix.atomic_max(...)` went through.
isdefined(@__MODULE__,:MemberNameConstraint) || include("member_name_constraint.jl")
isdefined(@__MODULE__,:kernel_transfer_tasks) || include("kernel_transfer_tasks.jl")
using Test,Random
import Mantle,KernelInterface,Atomix

const MEMBER_MODULES=["KI"=>KernelInterface,"KernelInterface"=>KernelInterface,"Atomix"=>Atomix]

# The two Qwen3.6 atomic_max failures of 2026-10-26, verbatim.
const MEMBER_FAILURES=(
    """
    function kernel!(out, labels, x, n::Int32)
        i = KI.get_global_id().x
        if i <= n
            Atomix.atomic_max(out, labels[i], x[i], :monotonic)
        end
    end
    """,
    """
    import KernelAbstractions: KernelInterface as KI
    using Atomix

    function kernel!(out, labels, x, n::Int32)
        i = KI.get_global_id().x
        if i <= n
            Atomix.atomic_max!(out, labels[i], x[i], :monotonic)
            nothing
        end
    end
    """)

function test_member_name_host(tables,tokenizer)
    M=MemberNameConstraint
    enc(s)=SmallLLMChecks.QwenText.encode(tokenizer,s)
    @testset "member-name mask on the host" begin
        for task in [vulkan_kernel_tasks();kernel_transfer_tasks()]
            r=M.trace(tables,enc("```julia\n"*task.reference*"```"))
            @test r.refused===nothing
            @test r.constrained>=3
            mutant=replace(task.reference,"KI.get_global_id"=>"KI.get_global_idx","Atomix.@atomic"=>"Atomix.atomic_max")
            @test M.trace(tables,enc("```julia\n"*mutant*"```")).refused!==nothing
            # Thinking text is never constrained.
            @test M.trace(tables,enc("Maybe "*mutant);thinking=true).refused===nothing
        end
        for code in MEMBER_FAILURES
            ids=enc("```julia\n"*code*"```");r=M.trace(tables,ids)
            @test r.refused!==nothing
            @test endswith(SmallLLMChecks.QwenText.decode(tokenizer,ids[1:r.refused];skipspecial=false),"Atomix.atomic")
        end
        for legal in ("Atomix.@atomic :monotonic max(out[labels[i]],x[i])","Atomix.modify!(r,max,v,Atomix.monotonic)",
                      "KernelInterface.get_global_id().x","KI.get_local_id().x","y=Mantle.anything(KI.get_global_id().x)")
            @test M.trace(tables,enc(legal)).refused===nothing
        end
        # After `</think>` the answer is constrained; before it nothing is.
        ids=enc("x.Atomix.atomic_max")
        @test M.trace(tables,[tables.think_end;ids];thinking=true).refused!==nothing
        # Code only. The first masked generation lost 81% of one step's mass to
        # `.jl` after `Atomix` in a COMMENT: comments, strings and prose after a
        # closed fence name packages, not members.
        for free in ("```julia\n# Atomix.jl provides atomic_max\nAtomix.@atomic x[1] += 1\n```",
                     "x = \"Atomix.jl\"; y = \"a\\\"Atomix.jl\"\nKI.get_global_id()",
                     "```julia\nAtomix.@atomic x[1] += 1\n```\nThis uses Atomix.jl and KI.whatever.")
            @test M.trace(tables,enc(free)).refused===nothing
        end
        for code in ("```julia\nusing Atomix.jl\n```","Atomix.atomic_max(x)","# note\nAtomix.jl")
            @test M.trace(tables,enc(code)).refused!==nothing
        end
    end
end

function test_member_name_device(device,tables,tokenizer)
    M=MemberNameConstraint;S=GPUSampling;n=tables.vocabulary
    enc(s)=SmallLLMChecks.QwenText.encode(tokenizer,s)
    rng=MersenneTwister(17)
    @testset "member-name mask on the device equals the host" begin
        prefixes=("```julia\nfunction kernel!(out)\n    i = KI.","        Atomix.","        Atomix.@at","x = Atom","y = KernelInterface.get_g","plain text ")
        for prefix in prefixes
            st=M.hoststate(tables,enc(prefix))
            want=M.allowedset(tables,st)
            c=M.State(device,tables;thinking=false);sampler=S.State(device,n;seed=UInt32(3))
            plan=nothing;owned=Any[]
            try
                c.state[:]=Int32[st...,0,0]
                logits=randn(rng,Float32,n);lb=Mantle.Buffer(device,logits);push!(owned,lb)
                allowed=Mantle.Buffer(device,Int32,(n,));push!(owned,allowed)
                g=Mantle.Graph(device)
                Mantle.dispatch!(g,M.allowed!,(allowed,c.state,c.trans,c.terminal,c.modtrans,c.modroot,c.bytes,c.offsets,c.special,
                    sampler.active,c.think_end,Int32(n)),n;group=256,name="allowed")
                Mantle.dispatch!(g,M.removed_share!,(c.state,c.removed,c.other,lb,allowed,sampler.active,Int32(n),sampler.temperature,Int32(c.capacity)),
                    256;group=256,name="share")
                Mantle.dispatch!(g,M.apply!,(lb,allowed,Int32(n)),n;group=256,name="mask")
                plan=Mantle.record!(Mantle.Plan(g;budget=Inf));Mantle.run!(plan);Mantle.waitidle(device)
                got=Array(allowed).!=0
                @test got==want
                masked=Array(lb)
                @test all(masked[.!want].==-Inf32)
                @test reinterpret(UInt32,masked[want])==reinterpret(UInt32,logits[want])
                e=exp.(Float64.(logits.-maximum(logits))./Float64(sampler.temperature))
                share=sum(e[.!want])/sum(e);r=M.report(c)
                if st[5]>=0
                    @test r.member_steps==1
                    @test only(r.removed_share)≈share rtol=1e-4
                else
                    @test r.member_steps==0
                    @test r.other_max_share≈share rtol=1e-4 atol=1e-7
                end
            finally
                plan===nothing||Mantle.free!(plan);foreach(Mantle.free!,owned);close(c);close(sampler)
            end
        end
    end
    @testset "member-name state on the device follows the host" begin
        ids=[tables.think_end;enc("\n```julia\nfunction kernel!(o,x)\n  i=KI.get_global_id().x\n  Atomix.@atomic :monotonic o[i] += x[i]\nend\n```")]
        c=M.State(device,tables;thinking=true);sampler=S.State(device,n);plan=nothing
        try
            g=Mantle.Graph(device)
            Mantle.dispatch!(g,M.advance!,(c.state,c.trans,c.terminal,c.modtrans,c.modroot,c.bytes,c.offsets,c.special,
                sampler.selected,sampler.status,sampler.active,c.think_end),1;group=1,name="advance")
            plan=Mantle.record!(Mantle.Plan(g;budget=Inf))
            host=M.initialstate(true);agree=true
            for id in ids
                sampler.selected[:]=Int32[id];Mantle.run!(plan);Mantle.waitidle(device)
                host=M.hoststep(tables,host,id)[1:5]
                agree&=Array(c.state)[1:5]==collect(host)
            end
            @test agree
            @test host[1]==1
            # Inactive sampling, as during prompt chunks, leaves the state alone.
            before=Array(c.state);S.activate!(sampler,false)
            sampler.selected[:]=Int32[tables.think_end];Mantle.run!(plan);Mantle.waitidle(device)
            @test Array(c.state)==before
        finally
            plan===nothing||Mantle.free!(plan);close(c);close(sampler)
        end
    end
    @testset "sampling through the mask never picks a refused token" begin
        st=M.hoststate(tables,enc("        Atomix."));want=M.allowedset(tables,st)
        # Logits that put almost all mass on refused tokens.
        logits=fill(-10f0,n);logits[.!want].=10f0;logits[findall(want)[1:5]].=0f0
        refused=0
        for seed in 1:8
            c=M.State(device,tables;thinking=false);sampler=S.State(device,n;seed=UInt32(seed));plan=nothing;lb=nothing
            try
                c.state[:]=Int32[st...,0,0];lb=Mantle.Buffer(device,logits)
                g=Mantle.Graph(device);M.declare!(g,lb,c,sampler)
                plan=Mantle.record!(Mantle.Plan(g;budget=Inf));Mantle.run!(plan);Mantle.waitidle(device)
                chosen=only(Array(sampler.selected))
                refused+=!want[chosen+1]
                @test only(Array(sampler.status))==0
                @test only(M.report(c).removed_share)>0.99
            finally
                plan===nothing||Mantle.free!(plan);lb===nothing||Mantle.free!(lb);close(c);close(sampler)
            end
        end
        @test refused==0
    end
end
