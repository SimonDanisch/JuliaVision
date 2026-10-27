isdefined(@__MODULE__,:KernelVoidTail) || include("kernel_void_tail.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
isdefined(@__MODULE__,:vulkan_kernel_tasks) || include("vulkan_kernel_tasks.jl")
using Test, JSON3, SHA

function qualify_kernel_void_tail()
    output=joinpath(@__DIR__,"kernel_void_tail_validation.json");isfile(output) && error("qualification already saved")
    source="import KernelInterface as KI\nfunction kernel!(out,x,n::Int32);i=KI.get_global_id().x;if 1<=i<=n;out[i]=x[i]*2f0;end;end"
    tests=raw"""
        for n in (0,1,7,65)
            x=Float32.(1:n+3);initial=fill(-999f0,n+3);expected=copy(initial)
            expected[1:n].=x[1:n].*2f0
            @test __bench_launch__(kernel!,initial,x,Int32(n);threads=n)==expected
        end
        """
    task=(;id="void_tail",category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,tests)
    rows=Any[]
    @testset "Void tail changes entry ABI, not mathematics" begin
        original=JuliaTaskEval.evaluate(task,source);push!(rows,(;arm="original",result=original))
        @test !original.passed
        @test occursin("kernel returns a value",original.diagnostic)
        normalized=KernelVoidTail.normalize(source);@test normalized.accepted
        result=JuliaTaskEval.evaluate(task,normalized.code);push!(rows,(;arm="void_tail",result))
        @test result.passed
        wrong=KernelVoidTail.normalize(replace(source,"*2f0"=>"*3f0"))
        result=JuliaTaskEval.evaluate(task,wrong.code);push!(rows,(;arm="wrong_arithmetic",result))
        @test !result.passed
        @test occursin("Test Failed",result.diagnostic)&&!occursin("Error During Test",result.diagnostic)
    end
    report=(;complete=true,scope="trusted ABI fixtures in software Vulkan, no model-quality result",rows,
        source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("kernel_void_tail.jl","qualify_kernel_void_tail.jl","julia_eval.jl","julia_sandbox.sh")))
    open(io->JSON3.pretty(io,report),output,"w");report
end
