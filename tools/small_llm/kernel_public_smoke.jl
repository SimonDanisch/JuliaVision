isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
isdefined(@__MODULE__,:vulkan_kernel_tasks) || include("vulkan_kernel_tasks.jl")
using JSON3, SHA

# Public smoke inputs, deliberately independent of private tests/reference bodies.
# These record values/errors, never compare against a private or reference result.
const KERNEL_PUBLIC_PROBES=Dict(
    "ki_body_affine"=>"__bench_launch__(kernel!,fill(-9f0,5),Float32[2,-3,1,8,4],2f0,1f0,Int32(3);threads=3)",
    "ki_body_clamped_stencil"=>"__bench_launch__(kernel!,fill(-9f0,5),Float32[2,-3,1,8,4],Int32(3);threads=3)",
    "ki_body_column_major_scale"=>"__bench_launch__(kernel!,fill(-9f0,8),Float32[2,-3,1,8,4,7,0,0],Float32[2,4],Int32(2),Int32(3);threads=6)",
    "ki_body_atomic_max"=>"__bench_launch__(kernel!,fill(-Inf32,2),Int32[1,2,1,2],Float32[2,-3,1,8],Int32(4);threads=4)")

function diagnose_kernel_public_smoke(task,code;probe=get(KERNEL_PUBLIC_PROBES,task.id,nothing))
    ncodeunits(code)<=65536 || throw(ArgumentError("candidate too large"))
    probe===nothing && throw(ArgumentError("unknown public kernel fixture"))
    # Include candidate inside the caught public execution so syntax/import errors
    # are usable compiler feedback rather than silently treated as unavailable.
    program=String(code)*"\n"*probe
    checks="""
        row=try
            observation=Base.include_string(Main,$(repr(program)),"public_kernel_probe.jl")
            (;completed=true,observed=repr(observation))
        catch err
            (;completed=false,error=first(sprint(showerror,err),4000))
        end
        println("PUBLIC_KERNEL_SMOKE=",JSON3.write(row))
        """
    fixture=(;id=task.id,category="vulkan_kernel",setup=VULKAN_KERNEL_SETUP*"\nusing JSON3\n",tests=checks)
    result=JuliaTaskEval.evaluate(fixture,"")
    prefix="PUBLIC_KERNEL_SMOKE="
    lines=filter(line->startswith(line,prefix),split(result.diagnostic,'\n'))
    payload=length(lines)==1 ? JSON3.read(only(lines)[length(prefix)+1:end]) : nothing
    (;task=task.id,scope="public compiler/runtime smoke observations; no private assertions, reference outputs or correctness judgments",
        analysis_completed=result.passed&&payload!==nothing,payload,
        loading_or_harness_error=payload===nothing ? result.diagnostic : nothing,
        public_prompt_sha256=bytes2hex(sha256(task.prompt)),
        fixture_sha256=bytes2hex(sha256(fixture.setup*probe)),code_sha256=bytes2hex(sha256(code)),
        elapsed_seconds=result.elapsed_seconds)
end
