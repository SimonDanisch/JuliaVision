isdefined(@__MODULE__,:PointwiseKernelLowering) || include("pointwise_kernel_lowering.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
isdefined(@__MODULE__,:vulkan_kernel_tasks) || include("vulkan_kernel_tasks.jl")
using Test, JSON3, SHA

function qualify_pointwise_lowering()
    output=joinpath(@__DIR__,"pointwise_lowering_validation.json")
    isfile(output) && error("qualification already recorded")
    source="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32);for i in 1:n;out[i]=x[i]*2f0;end;end"
    chain=replace(source,"out[i]=x[i]*2f0"=>"a=x[i]*3f0;b=a+1f0;out[i]=b*b")
    broadcast="function kernel!(out::Vector{Float32},x::Vector{Float32},scales::Vector{Float32},rows::Int32,cols::Int32);for i in 1:rows*cols;r=mod1(i,rows);out[i]=(x[i]+scales[r])*3f0;end;end"
    tests=raw"""
        for n in (0,1,5,65,259)
            x=Float32[sin(i*.17) for i in 1:n+3];initial=fill(-789f0,n+3);expected=copy(initial)
            expected[1:n].=2f0 .*x[1:n]
            @test __bench_launch__(kernel!,initial,x,Int32(n);threads=n) == expected
        end
        """
    chain_tests=replace(tests,"2f0 .*x[1:n]"=>"(x[1:n].*3f0 .+1f0).^2")
    broadcast_tests=raw"""
        for (rows,cols) in ((1,1),(3,7),(17,11))
            n=rows*cols;x=Float32.(1:n+3)./7f0;scales=Float32.(1:rows)./5f0
            initial=fill(-789f0,n+3);expected=copy(initial)
            for i in 1:n;expected[i]=(x[i]+scales[mod1(i,rows)])*3f0;end
            @test __bench_launch__(kernel!,initial,x,scales,Int32(rows),Int32(cols);threads=n)==expected
        end
        """
    rows=Any[]
    @testset "Restricted lowering software-Vulkan arithmetic" begin
        for (id,code,checks) in (("multiply",source,tests),("scalar_chain",chain,chain_tests),
                ("row_broadcast",broadcast,broadcast_tests))
            lowered=PointwiseKernelLowering.lower(code);@test lowered.accepted
            task=(;id,category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,tests=checks)
            result=JuliaTaskEval.evaluate(task,lowered.code)
            push!(rows,(;id,certificate=lowered.certificate,result));@test result.passed
        end
        # Compiler lowering must not act as a mathematical task oracle.
        wrong=PointwiseKernelLowering.lower(replace(source,"*2f0"=>"*3f0"))
        @test wrong.accepted
        task=(;id="wrong_arithmetic",category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,tests)
        result=JuliaTaskEval.evaluate(task,wrong.code)
        @test !result.passed
        @test occursin("Test Failed",result.diagnostic)&&!occursin("Error During Test",result.diagnostic)
        push!(rows,(;id="wrong_arithmetic",result))
    end
    report=(;complete=true,scope="three trusted synthetic compiler fixtures plus wrong-arithmetic rejection, software Vulkan only",
        rows,source_sha256=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("pointwise_kernel_lowering.jl","qualify_pointwise_lowering.jl","julia_eval.jl","julia_sandbox.sh")))
    open(io->JSON3.pretty(io,report),output,"w");report
end
