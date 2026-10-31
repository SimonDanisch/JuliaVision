include("compiler_transfer2_benchmark.jl")
scalar_format_digest(path)=bytes2hex(open(sha256,path))

function freeze_scalar_format_followup(;directory=joinpath(@__DIR__,"scalar_format_followup_results"))
    parentpath=joinpath(@__DIR__,"compiler_transfer2_results/conditions.json")
    summarypath=joinpath(dirname(parentpath),"summary.json")
    parent=JSON3.read(read(parentpath,String));summary=JSON3.read(read(summarypath,String))
    summary.complete || error("parent unfinished")
    failures=filter(r->r.arm=="scalar_program",summary.rows)
    length(failures)==4 && all(r->r.status=="failed" && r.stage=="compiler_admission",failures) || error("unexpected parent result")
    verify_transfer2_frozen_compiler()
    ispath(directory) && !isempty(readdir(directory)) && error("preserve prior registration")
    system="You produce scalar Julia programs for a compiler, NOT GPU functions. Output ONLY optional local scalar assignments followed by one final value expression or return value. Never output function/end wrappers, loops, imports, output stores, argument mutation or a global-id lookup. The framework ALREADY supplies arguments, valid index i, bounds guard and output store. Signatures in the task and API pack describe framework context, NOT the requested output format."
    jobs=Any[]
    for original in filter(j->j.arm=="scalar_program",parent.jobs)
        task=only(filter(t->t.id==original.task,compiler_transfer2_tasks()))
        public=ScalarKernelInterface.INTERFACES[task.interface_id]
        user=String(only(values(last(original.messages))))
        marker="\n\nIdentical installed public API/source context for both interface conditions:\n"
        context=last(split(user,marker;limit=2))
        prompt="Public API context (reference only, not output format):\n"*context*
            "\n\nArithmetic/indexing specification:\n"*task.specification*
            "\n\nFINAL OUTPUT CONTRACT: Return ONLY the pure scalar Julia PROGRAM computing out[i], never a function. Framework supplies signature "*public.signature*
            " and valid1-based i in1:("*public.limit*"). All arrays Float32 except indices::Int32. Array indices must be integers. Do not assign i or arguments. Pure functions: +,-,*,/,div,mod,mod1,min,max,clamp,muladd,Float32,Int32,Int. Optional local scalar assignments, then ONE final value expression (or return value). No kernel wrapper, imports, loop, bounds guard or output store. Supply all arithmetic yourself."
        push!(jobs,(;id=String(original.task)*"_scalar_format",task=original.task,arm="scalar_program",interface_id=original.interface_id,
            messages=["system"=>system,"user"=>prompt]))
    end
    packet=merge(JSON3.read(JSON3.write(parent),Dict{String,Any}),Dict(
        "scope"=>"POSTHOC prompt-format development ablation on same four now-used authored tasks; NOT prospective transfer or new model-only intelligence evidence",
        "policy"=>"One unchanged attempt per task. Same frozen compiler, API source context, arithmetic specs, seed, total/thinking budget. Strong system/final scalar-output contract only; no hidden feedback, arithmetic examples, retries or compiler changes. Original failed results retained.",
        "jobs"=>jobs,"format_parent_conditions_sha256"=>scalar_format_digest(parentpath),
        "format_parent_summary_sha256"=>scalar_format_digest(summarypath),
        "source_sha256"=>merge(Dict(String(k)=>String(v) for (k,v) in pairs(parent.source_sha256)),
            Dict("scalar_format_followup.jl"=>scalar_format_digest(@__FILE__)))))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function score_scalar_format_followup(;directory=joinpath(@__DIR__,"scalar_format_followup_results"))
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String))
    scalar_format_digest(joinpath(@__DIR__,"compiler_transfer2_results/conditions.json"))==p.format_parent_conditions_sha256 || error("parent condition drift")
    scalar_format_digest(joinpath(@__DIR__,"compiler_transfer2_results/summary.json"))==p.format_parent_summary_sha256 || error("parent score drift")
    score_compiler_transfer2(;directory)
end
