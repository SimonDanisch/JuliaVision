isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using JSON3, SHA

function score_native_repair_control()
    path=joinpath(@__DIR__,"native_missing_repair_512.json")
    control=JSON3.read(read(path,String))
    row=only(control.rows)
    hasproperty(row,:reference_greedy_text) || error("native CPU generation incomplete")
    control.generation_budget==512 && row.id=="missing_repair" || error("wrong control")
    task=only(filter(t->t.id=="missing_windows",JuliaTaskEval.JULIA_TASKS))
    quantized_path=joinpath(@__DIR__,"julia_jet_v4_results/missing_windows_repair.json")
    quantized=JSON3.read(read(quantized_path,String))
    quantized.contract_sha256==bytes2hex(sha256(task.tests)) || error("contract drift")
    answer=String(row.reference_greedy_text)
    code=JuliaTaskEval.extract_code(answer)
    score=JuliaTaskEval.evaluate(task,code;output=joinpath(@__DIR__,"native_missing_repair_512_score.json"))
    report=(;scope="single retained repair prompt, native CPU FP32 vs existing Q4; development control, no seed or prompt search",
        inference="independent Transformers CPU eager FP32, same512-token greedy output cap; generated code executed ONLY in restricted Julia sandbox",
        native_passed=score.passed,answer,code,generated_tokens=length(row.reference_greedy_ids),
        native_inference_seconds=row.elapsed_seconds,quantized_inference_seconds=quantized.elapsed_seconds,
        identical_extracted_code=strip(code)==strip(String(quantized.code)),
        control_sha256=bytes2hex(sha256(read(path))),
        quantized_candidate_sha256=bytes2hex(sha256(read(quantized_path))),
        original_contract_sha256=bytes2hex(sha256(task.tests)))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"native_missing_repair_512_comparison.json"),"w")
    report
end
