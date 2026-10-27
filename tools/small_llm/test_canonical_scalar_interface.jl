include("canonical_scalar_interface.jl")
isdefined(@__MODULE__,:shared_source_kernel_tasks) || include("shared_source_kernel_tasks.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using Test,JSON3,SHA

function test_canonical_scalar_interface()
    checks=@testset "Canonical global-x alias PRIVATE fixtures" begin
        id="shared_column_bias_clip"
        base="clamp(x[Int32(get_global_id().x)]+bias[div(Int32(get_global_id().x)-1,rows)+1],-1f0,1f0)"
        for name in ("get_global_id","KI.get_global_id","KernelInterface.get_global_id")
            source=replace(base,"get_global_id"=>name);r=CanonicalScalarInterface.lower(source,id)
            @test r.accepted
            @test r.certificate.global_x_alias_substitutions==2
        end
        for bad in ("get_global_id=1;"*base,"KI=1;"*base,"KernelInterface=1;"*base,
                    replace(base,"get_global_id"=>"get_local_id"),replace(base,".x"=>".y"),
                    replace(base,"get_global_id()"=>"get_global_id(1)"),"read(\"secret\")")
            @test !CanonicalScalarInterface.lower(bad,id).accepted
        end
        task=only(filter(t->t.id==id,shared_source_kernel_tasks()))
        r=CanonicalScalarInterface.lower(base,id)
        @test JuliaTaskEval.evaluate(task,r.code).passed
        wrong=CanonicalScalarInterface.lower(replace(base,"+bias"=>"-bias"),id)
        @test wrong.accepted
        @test !JuliaTaskEval.evaluate(task,wrong.code).passed
    end
    n=Test.get_test_counts(checks)
    packet=(;complete=true,scope="PRIVATE index-alias compiler admission/software-Vulkan differential checks; not model-quality evidence",passed=n.passes+n.cumulative_passes,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("canonical_scalar_interface.jl","test_canonical_scalar_interface.jl","scalar_program_interface.jl","scalar_kernel_interface.jl","typed_pointwise_kernel_lowering.jl")))
    open(io->JSON3.pretty(io,packet),joinpath(@__DIR__,"canonical_scalar_interface_validation.json"),"w");packet
end
