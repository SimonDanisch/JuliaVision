include("sampling_checkpoint.jl")
using Test
import KernelInterface as KI

# Trusted toy arithmetic exercises persistent state and cache/history updates;
# it is NOT a Qwen layer substitute or a full-model parity qualification.
function checkpoint_fixture_step!(logits,token,pos,history,state,keys,values)
    if KI.get_global_id().x==1
        p=pos[1]
        previous=p>0 ? values[Int32(2)*p-Int32(1)] : 0f0
        value=state[1]+Float32(token[1])*0.03125f0+previous*0.125f0+0.0625f0
        history[1]=value;state[1]=value
        keys[Int32(2)*p+Int32(1)]=value
        values[Int32(2)*p+Int32(1)]=value*0.5f0
        for i in Int32(1):Int32(7)
            logits[i]=Float32(mod(i*Int32(3)+token[1]+p,Int32(7)))*0.25f0+value*0.03125f0*Float32(i)
        end
    end
    nothing
end

function checkpoint_fixture(device;capacity=16)
    Q=SmallLLMChecks.Qwen35Experiment;owned=Any[]
    function buffer(T,dims)
        b=Mantle.Buffer(device,zeros(T,dims));push!(owned,b);b
    end
    token=Mantle.GPURef(device,Int32(0));pos=Mantle.GPURef(device,Int32(0));push!(owned,token,pos)
    logits=buffer(Float32,(7,));best=buffer(Int32,(1,));predictions=buffer(Int32,(capacity,))
    history=buffer(Float32,(3,2));state=buffer(Float32,(2,2,1))
    keys=buffer(Float32,(2,1,capacity));values=buffer(Float32,(2,1,capacity))
    sampler=GPUSampling.State(device,7;seed=UInt32(20261024),topk=7,temperature=0.6f0,top_p=0.95f0,presence=0.5f0)
    hook=CheckpointSamplingHook(sampler)
    model=Q.Model(device,(;vocab_size=7,hidden_size=4),Dict{String,Any}(),Any[],:q4k)
    graph=Mantle.Graph(device)
    Mantle.dispatch!(graph,checkpoint_fixture_step!,(logits,token,pos,history,state,keys,values),1;group=1,name="fixture.recurrence")
    hook(graph,logits)
    Q.BR.declare_argmax!(graph,best,logits;name="fixture.argmax")
    Mantle.dispatch!(graph,Q.advance_decode!,(token,pos,predictions,best),1;group=1,name="fixture.advance")
    plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
    session=Q.Session(model,capacity,0,token,pos,logits,best,predictions,plan,owned,Any[],
        Dict{Int,Any}(0=>(history,state)),Dict{Int,Any}(1=>(keys,values)),Dict{Int,Any}(),hook,:split)
    (;session,sampler)
end

function test_sampling_checkpoint_device(device)
    Q=SmallLLMChecks.Qwen35Experiment
    @testset "Cold sampled-state restore, synthetic shared kernels" begin
        original=checkpoint_fixture(device);restored=nothing
        try
            Q.step!(original.session,2);Q.decode!(original.session,3)
            mktempdir() do root
                directory=joinpath(root,"state")
                receipt=save_sampling_checkpoint(original.session,original.sampler,directory;identity="trusted_fixture_v1")
                @test receipt.position==4
                @test receipt.complete
                @test_throws ErrorException save_sampling_checkpoint(original.session,original.sampler,directory;identity="trusted_fixture_v1")
                Q.decode!(original.session,8);Mantle.waitidle(device)
                restored=checkpoint_fixture(device)
                @test_throws ErrorException restore_sampling_checkpoint!(restored.session,restored.sampler,directory;identity="wrong_model")
                @test restored.session.position==0
                # A corrupt final buffer must be rejected BEFORE any earlier
                # target buffer is restored or host position is advanced.
                lastfile=joinpath(directory,last(receipt.resources).file)
                bytes=read(lastfile);open(io->write(io,UInt8[0]),lastfile,"w")
                @test_throws ErrorException restore_sampling_checkpoint!(restored.session,restored.sampler,directory;identity="trusted_fixture_v1")
                @test restored.session.position==0
                open(io->write(io,bytes),lastfile,"w")
                restore_sampling_checkpoint!(restored.session,restored.sampler,directory;identity="trusted_fixture_v1")
                @test restored.session.position==4
                before=Mantle.submissions(device);Q.decode!(restored.session,8)
                @test Mantle.submissions(device)-before==8
                Mantle.waitidle(device)
                @test restored.session.position==original.session.position==12
                for ((name,a),(other,b)) in zip(checkpoint_resources(original.session,original.sampler),
                        checkpoint_resources(restored.session,restored.sampler))
                    @test name==other
                    @test isequal(checkpoint_array(a),checkpoint_array(b))
                end
                @test only(Array(restored.sampler.counter))==12
                @test_throws ErrorException restore_sampling_checkpoint!(restored.session,restored.sampler,directory;identity="trusted_fixture_v1")
            end
        finally
            restored===nothing || (close(restored.session);close(restored.sampler))
            close(original.session);close(original.sampler)
        end
    end
end
