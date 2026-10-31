include("scalar_kernel_interface.jl")
isdefined(@__MODULE__,:shared_source_kernel_tasks) || include("shared_source_kernel_tasks.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using Test,JSON3,SHA

function test_scalar_kernel_interface()
    # Trusted fixtures only; never imported into model-prompt construction.
    fixtures=Dict("shared_periodic_laplacian"=>"x[mod1(i-1,n)]-2f0*x[i]+x[mod1(i+1,n)]",
        "shared_indexed_interpolation"=>"(1f0-t[i])*x[clamp(indices[i],1,m-1)]+t[i]*x[clamp(indices[i],1,m-1)+1]",
        "shared_column_bias_clip"=>"clamp(x[i]+bias[div(i-1,rows)+1],-1f0,1f0)",
        "shared_block_reverse"=>"x[div(i-1,width)*width+width-mod(i-1,width)]")
    checks=@testset "Scalar interface PRIVATE fixtures, NOT model outputs" begin
        for code in ("out[i]","read(\"secret\")","x[i];x[i]","function f();1;end","for i=1:n;x[i];end","begin x[i] end","@inbounds x[i]","x[i]=1","unknown[i]")
            @test !ScalarKernelInterface.lower(code,"shared_periodic_laplacian").accepted
        end
        @test !ScalarKernelInterface.lower("x[i]","unknown").accepted
        for task in shared_source_kernel_tasks()
            value=fixtures[task.id];lowered=ScalarKernelInterface.lower(value,task.id)
            @test lowered.accepted
            @test JuliaTaskEval.evaluate(task,lowered.code).passed
            wrong=ScalarKernelInterface.lower("-("*value*")",task.id)
            @test wrong.accepted
            @test !JuliaTaskEval.evaluate(task,wrong.code).passed
        end
    end
    n=Test.get_test_counts(checks)
    report=(;complete=true,scope="Trusted scalar adapter software-Vulkan fixtures, including wrong-arithmetic rejection; NOT model quality",passed=n.passes+n.cumulative_passes,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("scalar_kernel_interface.jl","test_scalar_kernel_interface.jl","typed_pointwise_kernel_lowering.jl")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"scalar_kernel_interface_validation.json"),"w");report
end
