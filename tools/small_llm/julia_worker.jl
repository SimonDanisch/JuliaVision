# A sandboxed Julia that stays up across tool calls: the same bwrap jail as
# JuliaTaskEval (julia_sandbox.sh, software Vulkan only, no network, read-only
# project), started once and fed one request per line. Each call evaluates in a
# fresh module, so definitions do not carry over; the device does, which is the
# point: Mantle, Lava and llvmpipe load once instead of once per call.
isdefined(@__MODULE__,:VULKAN_KERNEL_SETUP) || include("vulkan_kernel_tasks.jl")
using JSON3,Random

# The loop the jail runs after the kernel tasks' setup. A result line carries the
# request's nonce, so nothing the code prints can pass for one.
const WORKER_LOOP=raw"""
using JSON3
println("WORKER_READY");flush(stdout)
for line in eachline(stdin)
    request=JSON3.read(line)
    m=Module(:ToolCall)
    for statement in (:(using Mantle),:(import KernelInterface as KI),:(import Atomix),:(const __bench_launch__=Main.__bench_launch__))
        Core.eval(m,statement)
    end
    result=try
        value=Base.include_string(m,String(request.code),"tool_call.jl")
        (;ok=true,value=first(repr(value),2000))
    catch err
        (;ok=false,error=first(sprint(showerror,err),3000))
    end
    flush(stdout)
    println("RESULT_",request.nonce,"\t",JSON3.write(result));flush(stdout)
end
"""

mutable struct SandboxWorker
    dir::String
    process::Union{Nothing,Base.Process}
    lines::Union{Nothing,Channel{String}}
    starts::Int
end
SandboxWorker()=SandboxWorker(mktempdir(),nothing,nothing,0)

alive(w::SandboxWorker)=w.process!==nothing && process_running(w.process)

function start!(w::SandboxWorker;timeout=120)
    # stderr joins stdout before the setup, whose startup messages are output too.
    write(joinpath(w.dir,"runner.jl"),"redirect_stderr(stdout)\n"*VULKAN_KERNEL_SETUP*"\n"*WORKER_LOOP)
    p=open(`bash $(joinpath(@__DIR__,"julia_sandbox.sh")) $(w.dir) vulkan_kernel`,"r+")
    ch=Channel{String}(Inf)
    # Lines are read on their own task so a hung call cannot block the caller.
    errormonitor(@async begin
        for l in eachline(p);put!(ch,l);end
        close(ch)
    end)
    w.process=p;w.lines=ch;w.starts+=1
    readyline(w,"WORKER_READY";timeout)===false && (stop!(w);error("sandbox worker did not start within $timeout s"))
    w
end

function stop!(w::SandboxWorker)
    if w.process!==nothing
        process_running(w.process) && kill(w.process)
        wait(w.process)
    end
    w.process=nothing;w.lines=nothing;w
end
Base.close(w::SandboxWorker)=(stop!(w);rm(w.dir;recursive=true,force=true);nothing)

# Wait for a line starting with `prefix`; everything before it is output.
function readyline(w::SandboxWorker,prefix;timeout,output=String[])
    deadline=time()+timeout
    while true
        remaining=deadline-time()
        remaining>0 || return false
        timedwait(()->isready(w.lines)||!isopen(w.lines),remaining;pollint=0.05)==:ok || return false
        isready(w.lines) || return false
        l=take!(w.lines)
        startswith(l,prefix) && return l
        push!(output,l)
    end
end

"""Run `code` in the worker: what it printed and its value, or its error.
A call that does not answer within `timeout` seconds stops the worker; the next
call starts a fresh one."""
function run_code(w::SandboxWorker,code::AbstractString;timeout=60)
    alive(w) || start!(w)
    nonce=randstring(16);output=String[]
    println(w.process,JSON3.write((;nonce,code)));flush(w.process)
    line=readyline(w,"RESULT_"*nonce*"\t";timeout,output)
    printed=join(output,"\n")
    if line===false
        dead=!alive(w);stop!(w)
        return (;ok=false,printed,error=dead ? "the sandbox process ended (time or memory limit, or a crash)" :
            "no answer within $timeout s; the call was stopped")
    end
    result=JSON3.read(split(line,'\t';limit=2)[2])
    (;ok=Bool(result.ok),printed,value=get(result,:value,nothing),error=get(result,:error,nothing))
end

"""The call's result as a tool response the model reads."""
function describe_run(r)
    parts=String[]
    isempty(r.printed) || push!(parts,"Printed:\n"*first(r.printed,2000))
    r.ok ? push!(parts,"Value: "*String(r.value)) : push!(parts,"Error:\n"*String(r.error))
    join(parts,"\n\n")
end
