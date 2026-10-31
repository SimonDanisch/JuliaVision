import KernelAbstractions
isdefined(@__MODULE__,:JuliaAPICards) || include("julia_api_cards.jl")

function prepare_kernel_feedback()
    qualified=JSON3.read(read(joinpath(@__DIR__,"julia_harness_system_validation.json"),String))
    any(r->r.id=="ka_tail"&&r.passed,qualified.references) || error("kernel reference not qualified")
    task=only(filter(t->t.id=="ka_tail",JuliaTaskEval.JULIA_TASKS))
    baseline=JSON3.read(read(joinpath(@__DIR__,"julia_tasks_results","ka_tail_baseline.json"),String))
    probe=JuliaTaskEval.diagnose(task,String(baseline.code))
    api=JuliaAPICards.cards(KernelAbstractions,[Symbol("@kernel"),Symbol("@index"),:get_backend,:synchronize];chars=2400)
    packet=(;scope="public compile/smoke feedback and installed API docstrings only; no private contract assertions/expected values",
        task=task.id,probe,api)
    open(io->JSON3.pretty(io,packet),joinpath(@__DIR__,"julia_kernel_public_feedback.json"),"w")
    packet
end

function run_kernel_repair(model,tokenizer,deadline,packet)
    dir=joinpath(@__DIR__,"julia_kernel_results");mkpath(dir)
    path=joinpath(dir,"ka_tail_repair.json");isfile(path) && return (;complete=true,already_present=true)
    time()<deadline-95 || return (;complete=false,reason="time budget margin")
    task=only(filter(t->t.id=="ka_tail",JuliaTaskEval.JULIA_TASKS))
    baseline=JSON3.read(read(joinpath(@__DIR__,"julia_tasks_results","ka_tail_baseline.json"),String))
    messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.",
        "user"=>task.prompt,"assistant"=>String(baseline.answer),
        "user"=>"The previous answer failed this public compilation/smoke probe. Use the installed public API documentation below to repair the complete code; code only.\n"*
            first(packet.probe.diagnostic,4000)*"\n\n"*JuliaAPICards.format_cards(packet.api)]
    answer=generate_julia_answer(model,tokenizer,messages;thinking=false,max_tokens=512,output_chunk=128)
    row=(;task=task.id,category=task.category,model="qwen35-4b",arm="public_compiler_and_api_cards_repair",thinking=false,
        scope="one 512-token revision, CPU PoCL actual portable kernel contract; not a Radeon kernel quality claim",
        contract_sha256=bytes2hex(sha256(task.tests)),messages,feedback=packet,answer...,
        generation_source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("julia_kernel_repair.jl","julia_api_cards.jl","run_julia_tasks.jl","qwen35.jl","prefill_qwen35.jl","qwen_text.jl","models.toml")))
    open(io->JSON3.pretty(io,row),path,"w")
    println("ka_tail compiler/API-card repair saved");flush(stdout)
    (;complete=true,path)
end

function score_kernel_controls()
    task=only(filter(t->t.id=="ka_tail",JuliaTaskEval.JULIA_TASKS));dir=joinpath(@__DIR__,"julia_kernel_results");mkpath(dir)
    rows=Any[]
    for arm in ("baseline","repair")
        source=arm=="baseline" ? joinpath(@__DIR__,"julia_tasks_results","ka_tail_baseline.json") : joinpath(dir,"ka_tail_repair.json")
        isfile(source) || continue
        candidate=JSON3.read(read(source,String))
        score=JuliaTaskEval.evaluate(task,String(candidate.code);output=joinpath(dir,"ka_tail_"*arm*"_score.json"))
        push!(rows,(;arm,score...));println("ka_tail ",arm," ",score.passed ? "PASS" : "FAIL");flush(stdout)
    end
    report=(;scope="CPU PoCL portable kernel correctness only, one authored tail task, not Radeon performance or broad kernel ability",rows)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"julia_kernel_summary.json"),"w")
    report
end
