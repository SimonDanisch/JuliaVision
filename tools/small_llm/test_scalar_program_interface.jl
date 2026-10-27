include("scalar_program_interface.jl")
isdefined(@__MODULE__,:shared_source_kernel_tasks) || include("shared_source_kernel_tasks.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using Test,JSON3,SHA

function test_scalar_program_interface()
    checks=@testset "Posthoc scalar program PRIVATE compiler fixtures" begin
        id="shared_indexed_interpolation"
        source="j=clamp(indices[i],1,m-1);ti=t[i];(1f0-ti)*x[j]+ti*x[j+1]"
        @test !ScalarKernelInterface.lower(source,id).accepted
        r=ScalarProgramInterface.lower(source,id)
        @test r.accepted
        for bad in ("x[i]=0;x[i]","out[i]=0f0;x[i]","j=out[i];j","n=n+1;x[i]","i=1;x[i]","j=read(\"secret\");j","for j=1:n;j;end;x[i]","j=unknown;x[i]","j=sum(x);j")
            @test !ScalarProgramInterface.lower(bad,id).accepted
        end
        @test ScalarProgramInterface.lower("begin "*source*" end",id).accepted
        @test ScalarProgramInterface.lower("j=clamp(indices[i],1,m-1);return x[j]",id).accepted
        task=only(filter(t->t.id==id,shared_source_kernel_tasks()))
        @test JuliaTaskEval.evaluate(task,r.code).passed
        wrong=ScalarProgramInterface.lower(replace(source,"+ti*x[j+1]"=>"-ti*x[j+1]"),id)
        @test wrong.accepted
        @test !JuliaTaskEval.evaluate(task,wrong.code).passed
    end
    n=Test.get_test_counts(checks)
    packet=(;complete=true,scope="PRIVATE compiler admission/software-Vulkan differential fixtures only; not model-quality evidence",passed=n.passes+n.cumulative_passes,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("scalar_program_interface.jl","test_scalar_program_interface.jl","scalar_kernel_interface.jl","typed_pointwise_kernel_lowering.jl")))
    open(io->JSON3.pretty(io,packet),joinpath(@__DIR__,"scalar_program_interface_validation.json"),"w");packet
end
