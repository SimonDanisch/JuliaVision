include("invariant_pointwise_lowering.jl")
isdefined(@__MODULE__,:shared_source_kernel_tasks) || include("shared_source_kernel_tasks.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using Test,JSON3,SHA

function test_invariant_pointwise_lowering()
    checks=@testset "Invariant compiler PRIVATE fixtures (no neural generations)" begin
        body="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32);N=n+0;for i=1:N;out[i]=x[i];end;nothing;end"
        @test !CheckedTypedPointwiseLowering.lower(body).accepted
        lowered=InvariantPointwiseLowering.lower(body)
        @test lowered.accepted
        @test lowered.certificate.invariant_aliases==["N"]
        @test occursin("n + 0",lowered.certificate.limit)
        for prefix in ("N=x[1]","N=n/2","N=1.0","N=div(n,0)","N=sin(n)","n=n+1;N=n","N=n;N=n+1","KI=n;N=KI","N=true")
            @test !InvariantPointwiseLowering.lower(replace(body,"N=n+0"=>prefix)).accepted
        end
        @test !InvariantPointwiseLowering.lower(replace(body,"out[i]=x[i]"=>"N=i;out[i]=x[i]")).accepted
        @test !InvariantPointwiseLowering.lower(replace(body,"for i=1:N"=>"for N=1:n")).accepted
        @test InvariantPointwiseLowering.lower(replace(body,"N=n+0"=>"A=n+0;N=A*1")).accepted
        task=only(filter(t->t.id=="shared_column_bias_clip",shared_source_kernel_tasks()))
        reference=replace(task.reference_cpu,"for i=1:rows*cols"=>"N=rows*cols\nfor i=1:N")
        @test reference!=task.reference_cpu
        fixed=InvariantPointwiseLowering.lower(reference)
        @test fixed.accepted
        unchanged=InvariantPointwiseLowering.lower(task.reference_cpu)
        @test unchanged.code==CheckedTypedPointwiseLowering.lower(task.reference_cpu).code
        cpu_setup="function __bench_launch__(f,initial,args...;threads);out=copy(initial);f(out,args...);out;end"
        cpu=JuliaTaskEval.evaluate((;id=task.id,category="base",setup=cpu_setup,tests=task.tests),reference)
        gpu=JuliaTaskEval.evaluate(task,fixed.code)
        @test cpu.passed
        @test gpu.passed
        mutant=replace(reference,"+bias[column]"=>"-bias[column]")
        @test mutant!=reference
        wrong=InvariantPointwiseLowering.lower(mutant)
        @test wrong.accepted
        @test !JuliaTaskEval.evaluate(task,wrong.code).passed
    end
    n=Test.get_test_counts(checks)
    packet=(;complete=true,scope="Private compiler admission/differential fixtures only; NOT new model-quality evidence",passed=n.passes+n.cumulative_passes,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("invariant_pointwise_lowering.jl","test_invariant_pointwise_lowering.jl","typed_pointwise_kernel_lowering.jl")))
    open(io->JSON3.pretty(io,packet),joinpath(@__DIR__,"invariant_pointwise_validation.json"),"w");packet
end
