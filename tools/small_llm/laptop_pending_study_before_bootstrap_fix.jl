isdefined(@__MODULE__,:sampled_safe_budget_julia_answer) || include("sampled_safe_budget_julia.jl")
isdefined(@__MODULE__,:load_portable_qwen35) || include("portable_qwen35.jl")
isdefined(@__MODULE__,:inference_runtime_provenance) || include("runtime_provenance.jl")
using JSON3,SHA

function laptop_original_pending_jobs()
    digest(path)=bytes2hex(open(sha256,path))
    extension=joinpath(@__DIR__,"safe_budget_transfer_extension_results/conditions.json")
    parent=joinpath(@__DIR__,"safe_reasoning_budget_results/conditions.json")
    transfer=joinpath(@__DIR__,"kernel_transfer_results/conditions.json")
    e=JSON3.read(read(extension,String));p=JSON3.read(read(parent,String));t=JSON3.read(read(transfer,String))
    e.parent_conditions_sha256==digest(parent) && p.transfer_conditions_sha256==digest(transfer) || error("original protocol drift")
    for packet in (e,p,t), (file,hash) in pairs(packet.source_sha256)
        digest(joinpath(@__DIR__,String(file)))==hash || error("original source drift: $file")
    end
    api=JSON3.read(read(joinpath(@__DIR__,"kernel_api_pack.json"),String))
    api.capture_source_sha256==digest(joinpath(@__DIR__,"capture_kernel_api_pack.jl")) || error("API capture drift")
    for card in api.cards
        mapped=nothing
        for (name,package) in pairs(api.packages)
            relative=relpath(String(card.origin_path),String(package.path))
            startswith(relative,"..") && continue
            mod=Base.require(Main,Symbol(name))
            string(pkgversion(mod))==package.version || error("public API package version drift")
            mapped=joinpath(pkgdir(mod),relative);break
        end
        mapped===nothing && error("unmapped public API source")
        digest(mapped)==card.source_sha256 || error("mapped API source bytes changed")
        bytes2hex(sha256(String(card.text)))==card.text_sha256 || error("API text drift")
    end
    jobs=Any[]
    for job in e.order
        original=joinpath(dirname(transfer),String(job.id)*".json")
        digest(original)==e.original_sha256[Symbol(job.id)] || error("original prompt record drift")
        row=JSON3.read(read(original,String))
        row.conditions_sha256==digest(transfer) || error("original generation drift")
        # Migration is permitted only for NEVER ATTEMPTED original conditions.
        for suffix in (".json","_attempt.json")
            isfile(joinpath(dirname(extension),String(job.id)*suffix)) && error("do not migrate an already attempted condition")
        end
        push!(jobs,(;id=job.id,task=job.task,arm=job.arm,messages=row.messages,original_sha256=digest(original)))
    end
    (;jobs,decoding=e.decoding,thinking_budget=e.thinking_budget,extension_sha256=digest(extension))
end

function freeze_laptop_pending_study()
    original=laptop_original_pending_jobs()
    gatepath=joinpath(@__DIR__,"laptop_portable_identity_gate.json")
    gate=JSON3.read(read(gatepath,String))
    gate.complete && gate.normal_final_waitidle && gate.hardware_fixture_passed==156 && gate.full_model_passed==10 || error("Laptop portable gate incomplete")
    for (file,hash) in pairs(gate.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash || error("Laptop gate source drift")
    end
    directory=joinpath(@__DIR__,"laptop_pending_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("already registered")
    packet=(;scope="Migration of six NEVER ATTEMPTED predeclared controller conditions to Laptop8060S, with a separately qualified exact-tensor portable loader. Hardware/driver/package closure differs from Bosgame: NOT a strict same-engine causal comparison with old outputs.",
        policy="same exact original messages/seed/sampling/budget/order; no regeneration; immutable attempt receipts; inference errors separate from code failures; results kept outside original directories",
        original...,gate_sha256=bytes2hex(open(sha256,gatepath)),portable_manifest_sha256=gate.portable_manifest_sha256,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("laptop_pending_study.jl","portable_qwen35.jl","sampled_safe_budget_julia.jl","safe_reasoning_budget.jl","reasoning_budget.jl",
                "gpu_sampling.jl","run_julia_tasks.jl","device_finish.jl","qwen35.jl","qwen_text.jl","prefill_qwen35.jl")))
    open(io->JSON3.pretty(io,packet),path,"w");packet
end

function laptop_pending_state()
    path=joinpath(@__DIR__,"laptop_pending_results/conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        bytes2hex(open(sha256,joinpath(@__DIR__,String(file))))==hash || error("Laptop protocol source drift")
    end
    original=laptop_original_pending_jobs()
    original.extension_sha256==p.extension_sha256 || error("original protocol drift")
    bytes2hex(open(sha256,joinpath(@__DIR__,"laptop_portable_identity_gate.json")))==p.gate_sha256 || error("hardware gate drift")
    bytes2hex(open(sha256,"gen/qwen35-4b-portable/portable.json"))==p.portable_manifest_sha256 || error("portable tensor identity drift")
    (;path,packet=p)
end

function run_laptop_pending_study(;seconds=210)
    seconds>0 || throw(ArgumentError("time budget"))
    s=laptop_pending_state();p=s.packet;directory=dirname(s.path)
    pending=filter(job->!isfile(joinpath(directory,String(job.id)*".json")),p.jobs)
    isempty(pending) && return (;complete=true)
    any(isfile(joinpath(directory,String(job.id)*"_attempt.json")) for job in pending) && error("unfinished prior attempt; do not regenerate")
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable");b=p.decoding
    let
        started=time();device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
            for job in pending
                time()<started+seconds-75 || return (;complete=false,completed)
                output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
                (isfile(output)||isfile(attempt)) && error("refuse regeneration")
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
                receipt=(;id=job.id,conditions_sha256=bytes2hex(open(sha256,s.path)),original_sha256=job.original_sha256,
                    seed=b.seed,budget_tokens=b.max_tokens,thinking_budget=p.thinking_budget,started_unix_seconds=time())
                open(io->JSON3.pretty(io,receipt),attempt,"w")
                answer=try
                    sampled_safe_budget_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=true,
                        thinking_budget=Int(p.thinking_budget),seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),
                        top_p=Float32(b.top_p),presence=Float32(b.presence),output_chunk=Int(b.output_chunk))
                catch err
                    failure=(;id=job.id,task=job.task,arm=job.arm,conditions_sha256=bytes2hex(open(sha256,s.path)),
                        attempt_sha256=bytes2hex(open(sha256,attempt)),inference_error=sprint(showerror,err),runtime_provenance=inference_runtime_provenance())
                    open(io->JSON3.pretty(io,failure),output,"w");rethrow()
                end
                row=(;answer...,id=job.id,task=job.task,arm=job.arm,messages,original_sha256=job.original_sha256,
                    conditions_sha256=bytes2hex(open(sha256,s.path)),attempt_sha256=bytes2hex(open(sha256,attempt)),
                    hardware="Laptop8060S",runtime_provenance=inference_runtime_provenance())
                open(io->JSON3.pretty(io,row),output,"w");push!(completed,String(job.id))
            end
            (;complete=true,completed)
        finally
            try model===nothing || close(model)
            finally finish_device!(device) end
        end
    end
end
