include("qwen36_moe_model.jl")
include("qwen36_moe_fixture_loader.jl")
using Test

function test_qwen36_moe_model(device,directory)
    R=Qwen36MoEExperiment
    reference=JSON3.read(read(joinpath(directory,"reference.json"),String))
    for (file,hash) in pairs(reference.fixture_sha256)
        bytes2hex(sha256(read(joinpath(directory,String(file)))))==hash||error("Reference drift")
    end
    rows=Any[]
    @testset "Full tiny MoE against independent Transformers CPU/FP32" begin
        model=load_qwen36_moe_fixture(R,directory,device);s=nothing
        try
            s=R.session(model;capacity=64,debug_layers=true,attention=:grouped)
            # Drain eager cache/state initialization before counting model runs.
            # Metal may retain those setup kernels in an open batch.
            Mantle.waitidle(device)
            for row in reference.rows
                before=Mantle.submissions(device);R.step!(s,Int(row.token));Mantle.waitidle(device)
                @test Mantle.submissions(device)-before==1
                logits=Array(s.logits);want=Float32.(row.logits)
                @test all(isfinite,logits)
                @test logits≈want atol=1f-4 rtol=2f-4
                @test argmax(logits)==argmax(want)
                for (il,expected) in enumerate(row.layers)
                    @test Array(s.layer_outputs[il])≈Float32.(expected) atol=1f-4 rtol=2f-4
                end
                for il in 0:3
                    route=s.routing[il];expected=row.routing[il+1]
                    @test only(Array(route.status))==0
                    @test vec(Array(route.selected))==Int32.(expected.selected)
                    @test vec(Array(route.weights))≈Float32.(expected.weights) atol=2f-6 rtol=2f-5
                end
                push!(rows,(;token=row.token,max_logit_error=maximum(abs,logits-want)))
            end
        finally
            s===nothing||close(s);close(model)
        end
        # Uninstrumented graph must not rely on diagnostic copies/synchronization.
        model=load_qwen36_moe_fixture(R,directory,device);plain=nothing;explicit=nothing
        try
            plain=R.session(model;capacity=64,debug_layers=false)
            for row in reference.rows
                R.step!(plain,Int(row.token));Mantle.waitidle(device)
                @test Array(plain.logits)≈Float32.(row.logits) atol=1f-4 rtol=2f-4
            end
            close(plain);plain=R.session(model;capacity=64)
            explicit=R.session(model;capacity=64)
            R.step!(plain,3);R.step!(explicit,3);Mantle.waitidle(device)
            before=Mantle.submissions(device);R.decode!(plain,7);Mantle.waitidle(device)
            @test Mantle.submissions(device)-before==7
            for _ in 1:7
                token=Int(only(Array(explicit.best)));R.step!(explicit,token);Mantle.waitidle(device)
            end
            @test Array(plain.predictions)[1:8]==Array(explicit.predictions)[1:8]
            @test reinterpret(UInt32,Array(plain.logits))==reinterpret(UInt32,Array(explicit.logits))
        finally
            plain===nothing||close(plain);explicit===nothing||close(explicit);close(model)
        end
    end
    rows
end
