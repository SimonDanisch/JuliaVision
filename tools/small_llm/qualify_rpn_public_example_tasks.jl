# ROOT ONLY: independent formulas, reference programs and hidden numerical probes.
include("rpn_public_example_tasks.jl")
include("rpn_kernel_device_audit.jl")
using JSON3,SHA,Test

function private_public_example_fixtures()
    [(;program="one\nx0\nsquare\nsub\nzero\nmax\nfinish",formula=(x,i,n)->max(0f0,1f0-x[i]^2)),
     (;program="x0\none\nneg\nmax\none\nmin\nfinish",formula=(x,i,n)->clamp(x[i],-1f0,1f0)),
     (;program="x_next_periodic\nx_prev_periodic\nsub\nsquare\nhalf\nmul\nfinish",formula=(x,i,n)->0.5f0*(x[mod1(i+1,n)]-x[mod1(i-1,n)])^2),
     (;program="two\nx0\nmul\nx_prev_clamped\nx_next_clamped\nmax\nsub\nfinish",formula=(x,i,n)->2f0*x[i]-max(x[max(1,i-1)],x[min(n,i+1)]))]
end

function private_public_example_cases(formula)
    rows=Any[]
    for n in (0,1,2,3,7,31,65),family in 1:3
        x=Float32[family==1 ? 2.3f0*sin(Float32(i)*0.71f0)+0.2f0 :
            family==2 ? (isodd(i) ? -1.5f0 : 2.25f0) : 0.17f0*Float32(i)-0.8f0 for i in 1:max(1,n)]
        initial=fill(-913f0,n+7);expected=copy(initial)
        for i in 1:n;expected[i]=formula(x,i,n);end
        push!(rows,(;n,x,initial,expected))
    end
    rows
end

function qualify_rpn_public_example_tasks(device;destination=joinpath(@__DIR__,"rpn_public_example_reference_validation.json"))
    isfile(destination)&&error("Preserve reference qualification")
    occursin("llvmpipe",lowercase(string(device)))||error("ROOT software-only qualification")
    tasks=rpn_public_example_tasks();fixtures=private_public_example_fixtures();rows=Any[]
    sources=("rpn_public_example_tasks.jl","qualify_rpn_public_example_tasks.jl","rpn_kernel_device_audit.jl",
        "rpn_pointwise_interface.jl","rpn_prefix_grammar.jl","typed_pointwise_kernel_lowering.jl")
    hashes=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in sources)
    save(complete)=open(io->JSON3.pretty(io,(;complete,scope="PRIVATE ROOT-only formula/reference/mutant qualification, not model quality",sources=hashes,rows)),destination,"w")
    save(false)
    for (task,fixture) in zip(tasks,fixtures)
        public_checks=@testset "Public example transcription matches independent formula" begin
            @test length(task.public_cases)==2
            for case in task.public_cases
                n=length(case.input)
                @test case.output==Float32[fixture.formula(case.input,i,n) for i in 1:n]
            end
        end
        cases=private_public_example_cases(fixture.formula)
        correct=audit_rpn_kernel_device(device,fixture.program,cases)
        incorrect=audit_rpn_kernel_device(device,"zero\nfinish",cases)
        correct.admitted&&correct.passed||error("Reference failed $(task.id)")
        incorrect.admitted&&!incorrect.passed||error("Wrong arithmetic not rejected $(task.id)")
        c=Test.get_test_counts(public_checks)
        push!(rows,(;id=task.id,public_checks=c.passes+c.cumulative_passes,correct,incorrect));save(false)
        println(task.id," reference passed; admitted mutant rejected");flush(stdout)
    end
    save(true);(;complete=true,tasks=length(rows))
end
