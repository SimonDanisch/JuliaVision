isdefined(@__MODULE__,:BayesJSON) || include("bayes_json.jl")
using Test, JSON3
function test_bayes_json()
    root=(;name="rain",states=["no","yes"],parents=String[],rows=[(;given=String[],p=[0.8,0.2])])
    sensor=(;name="wet",states=["no","yes"],parents=["rain"],
        rows=[(;given=["yes"],p=[0.1,0.9]),(;given=["no"],p=[0.95,0.05])])
    network=(;nodes=[root,sensor],query="rain",observed=Dict("wet"=>"yes"))
    @testset "bounded Bayesian JSON packing, no inference fork" begin
        result=BayesJSON.compile(JSON3.write(network))
        @test result.nstates==4
        @test result.query==:rain
        @test result.query_states==["no","yes"]
        @test result.observed==Dict(:wet=>2)
        @test result.factors[2].variables==[:rain,:wet]
        @test result.factors[2].values==[0.95 0.05;0.1 0.9]
        @test_throws ArgumentError BayesJSON.compile(JSON3.write(network);max_states=3)
        @test_throws ArgumentError BayesJSON.compile(JSON3.write(merge(network,(;query="unknown"))))
        @test_throws ArgumentError BayesJSON.compile(JSON3.write(merge(network,(;observed=Dict("wet"=>"maybe")))))
        @test_throws ArgumentError BayesJSON.compile(JSON3.write(merge(network,(;observed=Dict("x"=>"yes")))))
        for bad in (
            merge(sensor,(;parents=["wet"])),
            merge(sensor,(;parents=["unknown"])),
            merge(sensor,(;rows=[sensor.rows[1],sensor.rows[1]])),
            merge(sensor,(;rows=sensor.rows[1:1])),
            merge(sensor,(;rows=[(;given=["yes"],p=[0.1,0.1]),sensor.rows[2]])),
            merge(sensor,(;rows=[(;given=["maybe"],p=[0.1,0.9]),sensor.rows[2]])),
            merge(sensor,(;states=["no","no"])),
        )
            @test_throws ArgumentError BayesJSON.compile(JSON3.write(merge(network,(;nodes=[root,bad]))))
        end
        cyclic=merge(root,(;parents=["wet"],rows=sensor.rows))
        @test_throws ArgumentError BayesJSON.compile(JSON3.write(merge(network,(;nodes=[cyclic,sensor]))))
        @test_throws ArgumentError BayesJSON.compile(JSON3.write(merge(network,(;code="run(...)"))))
        @test_throws ArgumentError BayesJSON.compile(" "^65537)
    end
    (;passed=true,scope="host-side schema packing only; shared CPU/GPU inference kernels unchanged")
end

function test_bayes_json_backend(device,evidence_module)
    # Independent hand arithmetic, rather than a second inference algorithm.
    network="""{"nodes":[
      {"name":"rain","states":["no","yes"],"parents":[],"rows":[{"given":[],"p":[0.8,0.2]}]},
      {"name":"wet","states":["no","yes"],"parents":["rain"],"rows":[
        {"given":["yes"],"p":[0.1,0.9]},{"given":["no"],"p":[0.95,0.05]}]}],
      "query":"rain","observed":{"wet":"yes"}}"""
    packed=BayesJSON.compile(network)
    execution=evidence_module.prepare_bayes(device,packed.factors,packed.query,packed.observed)
    try
        @testset "Bayesian JSON uses shared inference kernels" begin
            evidence_module.run!(execution)
            @test Array(execution.outputs.prior) ≈ Float32[0.8,0.2] rtol=3e-5
            @test Array(execution.outputs.posterior) ≈ Float32[0.04/0.22,0.18/0.22] rtol=3e-5
            @test Array(execution.outputs.log_likelihood) ≈ Float32.(log.([0.05,0.9])) rtol=3e-5
            @test only(Array(execution.outputs.log_evidence)) ≈ log(0.22) rtol=3e-5
            @test only(Array(execution.outputs.status))==0
        end
    finally
        close(execution)
    end
    (;passed=true,scope="JSON-to-shared-inference integration, not model extraction accuracy")
end
