isdefined(@__MODULE__,:CheckedTypedPointwiseLowering) || include("typed_pointwise_kernel_lowering.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
isdefined(@__MODULE__,:VULKAN_KERNEL_SETUP) || include("vulkan_kernel_tasks.jl")
using Test,JSON3,SHA

function qualify_typed_pointwise_lowering()
    output=joinpath(@__DIR__,"typed_pointwise_lowering_validation.json")
    isfile(output) && error("preserve qualification")
    gather="""
    kernel!(out::Vector{Float32},x::Vector{Float32},indices::Vector{Int32},n::Int32,m::Int32)=begin
        for i=1:n
            idx=clamp(indices[i],1,m)
            out[i]=x[idx]
        end
    end
    """
    shifted="""
    function kernel!(out::Vector{Float32},x::Vector{Float32},indices::Vector{Int32},weights::Vector{Float32},n::Int32,m::Int32,offset::Int32)
        for i=1:n
            idx=clamp(indices[i]+offset,1,m)
            out[i]=x[idx]*weights[i]+Float32(offset)
        end
    end
    """
    checks=raw"""
    for (n,m) in ((0,1),(1,3),(7,5),(67,13),(259,29))
        x=Float32[cos(i*.21) for i in 1:m+3]
        indices=Int32[mod(17i,m+11)-5 for i in 1:n+3]
        initial=fill(-789f0,n+3);expected=copy(initial)
        for i=1:n;expected[i]=x[clamp(indices[i],1,m)];end
        @test __bench_launch__(kernel!,initial,x,indices,Int32(n),Int32(m);threads=n)==expected
    end
    """
    shifted_checks=raw"""
    for (n,m) in ((0,1),(1,3),(7,5),(67,13),(259,29)),offset in (-3,0,4)
        x=Float32[cos(i*.21) for i in 1:m+3];weights=Float32[sin(i*.13) for i in 1:n+3]
        indices=Int32[mod(17i,m+11)-5 for i in 1:n+3]
        initial=fill(-789f0,n+3);expected=copy(initial)
        for i=1:n;expected[i]=x[clamp(indices[i]+offset,1,m)]*weights[i]+Float32(offset);end
        @test __bench_launch__(kernel!,initial,x,indices,weights,Int32(n),Int32(m),Int32(offset);threads=n)≈expected atol=2f-6 rtol=2f-6
    end
    """
    rows=Any[]
    @testset "Typed compiler software-Vulkan differential fixtures" begin
        for (id,source,tests,expected) in (("gather",gather,checks,true),
                ("shifted_weighted_gather",shifted,shifted_checks,true),
                ("wrong_gather_arithmetic",replace(gather,"out[i]=x[idx]"=>"out[i]=-x[idx]"),checks,false))
            lowered=CheckedTypedPointwiseLowering.lower(source);@test lowered.accepted
            task=(;id,category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP,tests)
            result=JuliaTaskEval.evaluate(task,lowered.code)
            @test result.passed==expected
            if !expected
                @test occursin("Test Failed",result.diagnostic) && !occursin("Error During Test",result.diagnostic)
            end
            push!(rows,(;id,certificate=lowered.certificate,result))
        end
    end
    report=(;complete=true,scope="trusted synthetic typed-input compiler fixtures; software Vulkan only, NOT model intelligence",
        rows,source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("typed_pointwise_kernel_lowering.jl","qualify_typed_pointwise_lowering.jl","julia_eval.jl","julia_sandbox.sh")))
    open(io->JSON3.pretty(io,report),output,"w");report
end
