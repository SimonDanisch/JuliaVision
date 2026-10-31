# ROOT ONLY. Trusted reference sources and independent probes never deployed.
include("boundary_transfer_tasks.jl")
include("julia_stencil_admission_v3.jl")
include("public_example_fallback.jl")
include("rpn_kernel_device_audit.jl")
using Test,SHA,JSON3
function private_boundary_transfer_fixtures()
    [("x[max(1,i-1)]^2",(x,i,n)->x[max(1,i-1)]^2),
     ("x[mod1(i+1,n)]^2",(x,i,n)->x[mod1(i+1,n)]^2),
     ("max(x[i]-x[max(1,i-1)],0f0)",(x,i,n)->max(x[i]-x[max(1,i-1)],0f0)),
     ("min(x[mod1(i-1,n)],x[mod1(i+1,n)])",(x,i,n)->min(x[mod1(i-1,n)],x[mod1(i+1,n)])),
     ("x[max(1,i-1)]*x[min(n,i+1)]",(x,i,n)->x[max(1,i-1)]*x[min(n,i+1)]),
     ("(x[mod1(i-1,n)]+x[mod1(i+1,n)])-2f0*x[i]",(x,i,n)->(x[mod1(i-1,n)]+x[mod1(i+1,n)])-2f0*x[i]),
     ("abs(x[max(1,i-1)]-x[min(n,i+1)])",(x,i,n)->abs(x[max(1,i-1)]-x[min(n,i+1)])),
     ("max(x[i],max(x[mod1(i-1,n)],x[mod1(i+1,n)]))",(x,i,n)->max(x[i],x[mod1(i-1,n)],x[mod1(i+1,n)]))]
end

function private_boundary_transfer_cases(formula)
    rows=Any[]
    for n in (0,1,2,4,9,33,67),family in 1:3
        x=Float32[family==1 ? 2.1f0*sin(0.63f0*i)+0.1f0 : family==2 ?
            (isodd(i) ? -0.75f0 : 1.875f0) : 0.19f0*i-1.3f0 for i in 1:max(1,n)]
        initial=fill(-983f0,n+7);expected=copy(initial)
        for i in 1:n;expected[i]=formula(x,i,n);end
        push!(rows,(;n,x,initial,expected))
    end
    rows
end

function qualify_boundary_transfer(device;destination=joinpath(@__DIR__,"boundary_transfer_reference_validation.json"))
    isfile(destination)&&error("Preserve reference qualification")
    occursin("llvmpipe",lowercase(string(device)))||error("ROOT software-only")
    rows=Any[]
    for (task,(expression,formula)) in zip(boundary_transfer_tasks(),private_boundary_transfer_fixtures())
        source="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)\nfor i=1:n\nout[i]="*expression*"\nend\nend"
        lower=JuliaStencilAdmissionV3.lower(source)
        tests=@testset "Fresh boundary reference, public transcription, wrong-program rejection" begin
            @test lower.accepted
            for case in task.public_cases
                n=length(case.input)
                @test case.output==Float32[formula(case.input,i,n) for i in 1:n]
            end
            bank=RPNExampleSemantics.Bank(task.public_cases)
            @test PublicExampleFallback.matches(lower.program,bank)
            @test !PublicExampleFallback.matches("zero\nfinish",bank)
        end
        cases=private_boundary_transfer_cases(formula)
        reference=audit_rpn_kernel_device(device,lower.program,cases)
        mutant=audit_rpn_kernel_device(device,"zero\nfinish",cases)
        reference.admitted&&reference.passed&&mutant.admitted&&!mutant.passed||error("Reference/mutant qualification failed")
        c=Test.get_test_counts(tests);push!(rows,(;id=task.id,host_checks=c.passes+c.cumulative_passes,reference,mutant))
        println(task.id," reference passed; zero mutant rejected");flush(stdout)
    end
    sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
        ("boundary_transfer_tasks.jl","qualify_boundary_transfer.jl","julia_stencil_interface.jl","julia_stencil_admission_v2.jl","julia_stencil_admission_v3.jl","rpn_kernel_device_audit.jl"))
    result=(;complete=true,scope="Infrastructure/reference qualification only; no neural quality claim",sources,rows)
    open(io->JSON3.pretty(io,result),destination,"w");result
end
