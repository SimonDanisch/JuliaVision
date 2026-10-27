# ROOT ONLY. Never deploy reference sources or private cases to model workers.
include("julia_stencil_transfer_tasks.jl")
include("julia_stencil_interface.jl")
include("public_example_fallback.jl")
include("rpn_kernel_device_audit.jl")
using Test, SHA, JSON3

function private_julia_stencil_transfer_fixtures()
    [("abs(x[i])",(x,i,n)->abs(x[i])),
     ("max(x[max(1,i-1)]-x[i],0f0)",(x,i,n)->max(x[max(1,i-1)]-x[i],0f0)),
     ("(x[mod1(i-1,n)]+x[mod1(i+1,n)])*0.5f0",(x,i,n)->(x[mod1(i-1,n)]+x[mod1(i+1,n)])*0.5f0),
     ("max(x[max(1,i-1)],x[min(n,i+1)])-x[i]",(x,i,n)->max(x[max(1,i-1)],x[min(n,i+1)])-x[i]),
     ("x[i]*x[mod1(i+1,n)]",(x,i,n)->x[i]*x[mod1(i+1,n)]),
     ("1f0-clamp(x[i],0f0,1f0)",(x,i,n)->1f0-clamp(x[i],0f0,1f0)),
     ("min(min(x[max(1,i-1)],x[i]),x[min(n,i+1)])",(x,i,n)->min(x[max(1,i-1)],x[i],x[min(n,i+1)])),
     ("max(x[mod1(i-1,n)]^2,x[mod1(i+1,n)]^2)",(x,i,n)->max(x[mod1(i-1,n)]^2,x[mod1(i+1,n)]^2))]
end

function private_julia_stencil_transfer_cases(formula)
    rows=Any[]
    for n in (0,1,2,3,7,31,65),family in 1:3
        x=Float32[family==1 ? 1.7f0*cos(0.83f0*i)-0.3f0 :
            family==2 ? (isodd(i) ? 1.25f0 : -2.75f0) : 0.13f0*i-1.2f0 for i in 1:max(1,n)]
        initial=fill(-971f0,n+7);expected=copy(initial)
        for i in 1:n;expected[i]=formula(x,i,n);end
        push!(rows,(;n,x,initial,expected))
    end
    rows
end

function qualify_julia_stencil_transfer(device;destination=joinpath(@__DIR__,"julia_stencil_transfer_reference_validation.json"))
    isfile(destination)&&error("Preserve existing qualification")
    occursin("llvmpipe",lowercase(string(device)))||error("Software-only ROOT qualification")
    rows=Any[]
    for (task,(expression,formula)) in zip(julia_stencil_transfer_tasks(),private_julia_stencil_transfer_fixtures())
        source="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)\nfor i=1:n\nout[i]="*expression*"\nend\nreturn nothing\nend"
        lower=JuliaStencilInterface.lower(source)
        checks=@testset "Fresh Julia reference and public transcription" begin
            @test lower.accepted
            for case in task.public_cases
                n=length(case.input)
                @test case.output==Float32[formula(case.input,i,n) for i in 1:n]
            end
            bank=RPNExampleSemantics.Bank(task.public_cases)
            @test PublicExampleFallback.matches(lower.program,bank)
            @test !PublicExampleFallback.matches("zero\nfinish",bank)
        end
        cases=private_julia_stencil_transfer_cases(formula)
        correct=audit_rpn_kernel_device(device,lower.program,cases)
        mutant=audit_rpn_kernel_device(device,"zero\nfinish",cases)
        correct.admitted&&correct.passed||error("Reference failed $(task.id)")
        mutant.admitted&&!mutant.passed||error("Zero mutant escaped $(task.id)")
        counts=Test.get_test_counts(checks)
        push!(rows,(;id=task.id,host_checks=counts.passes+counts.cumulative_passes,correct,mutant))
        println(task.id," safe-source reference passed; zero mutant rejected");flush(stdout)
    end
    sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
        ("julia_stencil_transfer_tasks.jl","julia_stencil_interface.jl","qualify_julia_stencil_transfer.jl","rpn_kernel_device_audit.jl"))
    result=(;complete=true,scope="Infrastructure/reference qualification, NOT generated-model intelligence",sources,rows)
    open(io->JSON3.pretty(io,result),destination,"w");result
end
