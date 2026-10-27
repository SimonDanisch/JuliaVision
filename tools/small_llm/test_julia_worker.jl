# The persistent sandbox worker: results, errors, isolation between calls, a
# hung call, and that the jail is the same one the scorer uses (no /dev/dri,
# llvmpipe only).
isdefined(@__MODULE__,:SandboxWorker) || include("julia_worker.jl")
using Test

function test_julia_worker()
    w=SandboxWorker()
    try
        @testset "sandbox worker" begin
            r=run_code(w,"1+1")
            @test r.ok && r.value=="2"
            r=run_code(w,"println(\"hello\"); 3")
            @test r.ok && r.printed=="hello" && r.value=="3"
            # A forged result line is just printed output.
            r=run_code(w,"println(\"RESULT_xxxxxxxxxxxxxxxx\t{\\\"ok\\\":true}\"); 4")
            @test r.ok && r.value=="4" && occursin("RESULT_",r.printed)
            r=run_code(w,"undefined_function_xyz()")
            @test !r.ok && occursin("undefined_function_xyz",r.error)
            r=run_code(w,"using KernelInterface as KI")
            @test !r.ok && occursin("ParseError",r.error)
            # Definitions do not carry over between calls.
            run_code(w,"leaked=1")
            @test !run_code(w,"leaked").ok
            # The kernel tasks' launch helper, on the software device.
            k="function kernel!(out,x,n::Int32)\n    i=KI.get_global_id().x\n    i<=n && (@inbounds out[i]=2f0*x[i])\n    nothing\nend\n"*
              "__bench_launch__(kernel!,zeros(Float32,4),Float32[1,2,3,4],Int32(3);threads=3)"
            r=run_code(w,k)
            @test r.ok && r.value=="Float32[2.0, 4.0, 6.0, 0.0]"
            r=run_code(w,"(isdir(\"/dev/dri\"), lowercase(string(Mantle.devicename(Main.__bench_device))))")
            @test r.ok && occursin("(false,",r.value) && occursin("llvmpipe",r.value)
            starts=w.starts
            r=run_code(w,"while true end";timeout=10)
            @test !r.ok && occursin("no answer within 10 s",r.error)
            r=run_code(w,"5")
            @test r.ok && r.value=="5" && w.starts==starts+1
            @test occursin("Error:",describe_run(run_code(w,"error(\"boom\")")))
        end
    finally
        close(w)
    end
end
