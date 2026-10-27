isdefined(@__MODULE__,:run_humaneval_baselines) || include("run_humaneval_subset.jl")
isdefined(@__MODULE__,:sampled_julia_answer) || include("sampled_julia.jl")

function freeze_humaneval_sampling()
    directory=joinpath(@__DIR__,"humaneval_sampled_results");mkpath(directory)
    path=joinpath(directory,"conditions.json");isfile(path) && error("sampling condition already frozen")
    baseline=joinpath(@__DIR__,"humaneval_subset_results")
    original=JSON3.read(read(joinpath(baseline,"conditions.json"),String))
    report=(;scope="development decoding ablation on the SAME twelve public basic-test tasks; not held-out/frontier evidence",
        selection_sha256=original.selection_sha256,qualification_sha256=original.qualification_sha256,
        baseline=(;thinking=true,decoding="exact top-k/nucleus",max_tokens=2048,output_poll_chunk=128,
            temperature=0.6,top_p=0.95,topk=20,presence=0.0,seed=20261024),
        comparisons="one sampled output per task vs retained original greedy output; no seed search or regeneration; same prompt and output cap",
        limitation="Only decoding changes. The2048-token cap is still below Qwen's recommended long output budgets.",
        greedy_conditions_sha256=bytes2hex(sha256(read(joinpath(baseline,"conditions.json")))),
        source_sha256=bytes2hex(sha256(read(@__FILE__))))
    open(io->JSON3.pretty(io,report),path,"w")
    report
end

function run_humaneval_sampling(;seconds=240)
    directory=joinpath(@__DIR__,"humaneval_sampled_results")
    conditions=JSON3.read(read(joinpath(directory,"conditions.json"),String))
    conditions.source_sha256==bytes2hex(sha256(read(@__FILE__))) || error("sampling study source drift")
    generate_answer=(args...;kwargs...)->sampled_julia_answer(args...;kwargs...,
        seed=UInt32(conditions.baseline.seed),temperature=Float32(conditions.baseline.temperature),
        top_p=Float32(conditions.baseline.top_p),topk=Int(conditions.baseline.topk),presence=Float32(conditions.baseline.presence))
    run_humaneval_baselines(;seconds,directory,generate_answer,
        extra_source_files=("humaneval_sampling_ablation.jl","sampled_julia.jl","gpu_sampling.jl"))
end
