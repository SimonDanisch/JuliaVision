include("bayes_precision_audit.jl")
using Test
function test_bayes_precision_audit()
    @testset "Precision is distinct from faithful problem extraction" begin
        rows=[(;id="case_$(i)_$arm",task="case_$i",arm,status=arm=="direct_probability" ? "failed" : "passed",
            probability=arm=="direct_probability" ? 0.333 : 1/3,expected_probability=1/3,
            network_faithful=arm=="bayes_network" ? true : nothing)
            for i in 1:8 for arm in ("direct_probability","bayes_network")]
        result=bayes_precision_statistics((;complete=true,rows))
        @test result.curve[1].direct_numeric==0
        @test result.curve[3].direct_numeric==8
        @test all(r->r.network_numeric==r.network_faithful_and_numeric==8,result.curve)
        @test result.errors[1].measured==8
        @test result.errors[2].max_absolute_error==0
        changed=copy(rows);changed[2]=(;changed[2]...,network_faithful=false,status="failed")
        second=bayes_precision_statistics((;complete=true,rows=changed))
        @test second.curve[1].network_numeric==8
        @test second.curve[1].network_faithful_and_numeric==7
        @test_throws ArgumentError bayes_precision_statistics((;complete=false,rows))
        @test_throws ArgumentError bayes_precision_statistics((;complete=true,rows=rows[1:15]))
        mktempdir() do directory
            save(path,value)=open(io->JSON3.write(io,value),path,"w")
            save(joinpath(directory,"conditions.json"),(;jobs=[(;id="mock_$i") for i in 1:16]))
            save(joinpath(directory,"mock_1.json"),(;answer="ungraded"))
            registered=freeze_bayes_precision_audit(;directory)
            @test length(registered.existing_generation_sha256)==1
            @test registered.absolute_tolerances==collect(BAYES_AUDIT_TOLERANCES)
            @test_throws ErrorException freeze_bayes_precision_audit(;directory)
        end
        mktempdir() do directory
            open(io->write(io,"{}"),joinpath(directory,"summary.json"),"w")
            @test_throws ErrorException freeze_bayes_precision_audit(;directory)
        end
    end
end
