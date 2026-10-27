include("laptop_shared_source.jl")
bcbx_digest(path)=bytes2hex(open(sha256,path))
const BCBX_SNAPSHOT=abspath(joinpath(@__DIR__,"../../../../gen/bigcodebench-x-snapshot-20261025"))

function freeze_bigcodebench_x_study(;directory=joinpath(@__DIR__,"bigcodebench_x_results"))
    ispath(directory)&&!isempty(readdir(directory))&&error("preserve registration")
    manifestpath=joinpath(BCBX_SNAPSHOT,"manifest.json");manifest=JSON3.read(read(manifestpath,String))
    validationpath=joinpath(BCBX_SNAPSHOT,"reference_validation_v2.json")
    validation=JSON3.read(read(validationpath,String))
    validation.complete&&validation.all_reference_tests_passed&&length(validation.rows)==8 || error("external reference tests unqualified")
    validation.snapshot_manifest_sha256==bcbx_digest(manifestpath) || error("reference receipt drift")
    adapterpath=joinpath(@__DIR__,"bigcodebench_x_julia_adapter_validation.json")
    adapter=JSON3.read(read(adapterpath,String));adapter.complete&&adapter.tests==3 || error("Julia adapter unqualified")
    for (file,hash) in pairs(adapter.source_sha256)
        bcbx_digest(joinpath(@__DIR__,String(file)))==hash || error("adapter source drift")
    end
    for (file,hash) in pairs(manifest.files_sha256)
        basename(String(file))==String(file)&&bcbx_digest(joinpath(BCBX_SNAPSHOT,String(file)))==hash || error("external snapshot drift")
    end
    base=laptop_shared_source_state(joinpath(@__DIR__,"laptop_shared_source_results")).packet
    prompts=JSON3.read.(readlines(joinpath(BCBX_SNAPSHOT,"public_prompts.jsonl")))
    String.(getproperty.(prompts,:task_id))==String.(manifest.selected_ids) || error("cohort drift")
    contract="Write a complete standalone Julia1.12.7 program that reads standard input and writes standard output as specified. No explanatory prose. Installed packages: JSON3, CairoMakie, Makie and Julia standard libraries (including Base64, Statistics, LinearAlgebra, Random). Do not assume DataFrames/CSV/Plots/PyPlot/ScikitLearn are installed; implement needed computation with available libraries. No package installation or network access. Plotting must be headless; PNGs can be written to an IOBuffer then Base64-encoded when requested."
    jobs=[(;id="bcbx_"*last(split(String(row.task_id),'/'))*"_baseline",task=row.task_id,arm="baseline",
        messages=["system"=>contract,"user"=>String(row.prompt)]) for row in prompts]
    packet=(;scope="Fixed-hash eight-task public BigCodeBench-X subset, local Julia1.12/package-contract adaptation. NOT official upstream-container score, private/contamination-free benchmark, Makie aesthetics metric or frontier equivalence.",
        policy="ALL eight fixed tasks retained, one fixed-seed attempt each, no feedback/retries/selection. Original external test suites unchanged. Correctness judged by stdin/stdout tests, not code style or plot aesthetics.",
        jobs,decoding=(;base.decoding...,max_tokens=6144),thinking_budget=4096,
        parent_conditions_sha256=base.parent_conditions_sha256,gate_sha256=base.gate_sha256,
        portable_manifest_sha256=base.portable_manifest_sha256,
        snapshot_manifest_sha256=bcbx_digest(manifestpath),reference_validation_sha256=bcbx_digest(validationpath),
        adapter_validation_sha256=bcbx_digest(adapterpath),
        project_sha256=bcbx_digest(abspath(joinpath(@__DIR__,"../../../../Project.toml"))),
        manifest_sha256=bcbx_digest(abspath(joinpath(@__DIR__,"../../../../Manifest.toml"))),
        source_sha256=merge(Dict(String(k)=>String(v) for (k,v) in pairs(base.source_sha256)),
            Dict(f=>bcbx_digest(joinpath(@__DIR__,f)) for f in
                ("bigcodebench_x_study.jl","score_bigcodebench_x.py","bigcodebench_x_julia_adapter.py","qualify_bigcodebench_x_reference.py"))))
    mkpath(directory);open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w");packet
end

function run_bigcodebench_x_study(;directory=joinpath(@__DIR__,"bigcodebench_x_results"))
    s=laptop_shared_source_state(directory);p=s.packet;b=p.decoding
    pending=filter(j->!isfile(joinpath(directory,String(j.id)*".json")),p.jobs)
    isempty(pending)&&return (;complete=true)
    job=first(pending);path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
    ispath(attempt)&&error("interrupted attempt; do not regenerate")
    bcbx_digest("gen/qwen35-4b-portable/portable.json")==p.portable_manifest_sha256 || error("model identity")
    tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b-portable")
    messages=[String(first(keys(m)))=>String(first(values(m))) for m in job.messages]
    length(SmallLLMChecks.QwenText.chat_ids(tokenizer,messages;thinking=true))+Int(b.max_tokens)+32<=8192 || error("capacity preflight")
    let
        device=Mantle.Device("radeon");model=nothing
        try
            model=load_portable_qwen35(SmallLLMChecks.Qwen35Experiment,"gen/qwen35-4b-portable",device)
            open(io->JSON3.pretty(io,(;id=job.id,conditions_sha256=bcbx_digest(s.path),started_unix_seconds=time())),attempt,"w")
            answer=try
                sampled_safe_budget_julia_answer(model,tokenizer,messages;max_tokens=Int(b.max_tokens),thinking=true,
                    thinking_budget=Int(p.thinking_budget),seed=UInt32(b.seed),topk=Int(b.topk),temperature=Float32(b.temperature),
                    top_p=Float32(b.top_p),presence=Float32(b.presence),output_chunk=Int(b.output_chunk))
            catch err
                open(io->JSON3.pretty(io,(;id=job.id,inference_error=sprint(showerror,err),attempt_sha256=bcbx_digest(attempt))),path,"w");rethrow()
            end
            row=(;answer...,id=job.id,task=job.task,arm=job.arm,messages,conditions_sha256=bcbx_digest(s.path),
                attempt_sha256=bcbx_digest(attempt),hardware="Laptop8060S",runtime_provenance=inference_runtime_provenance())
            open(io->JSON3.pretty(io,row),path,"w");(;output=path)
        finally
            try model===nothing||close(model)
            finally finish_device!(device) end
        end
    end
end
