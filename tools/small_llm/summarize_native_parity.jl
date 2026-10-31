using JSON3, SHA
function summarize_native_parity()
    cases=(("conversation_gpu_baseline_recovery_r82.json","baseline_short",98),
        ("conversation_gpu_partial_r86.json","missing_repair",434),
        ("conversation_gpu_kernel_parity_r93.json","kernel_repair",1910))
    rows=Any[]
    for (file,id,n) in cases
        path=joinpath(@__DIR__,file);report=JSON3.read(read(path,String))
        row=only(filter(r->r.id==id,report.rows))
        @assert row.input_tokens==n && row.finite && row.relative_rms<2e-4 && row.maxabs<0.02
        @assert row.prefix_argmax==row.native_prefix_argmax && row.exact_greedy
        @assert row.generated_ids==row.native_ids && row.decode_submissions==8
        push!(rows,(;id,input_tokens=n,relative_rms=row.relative_rms,maxabs=row.maxabs,
            greedy_tokens_matched=length(row.generated_ids),decode_submissions=row.decode_submissions,
            source_report=file,source_sha256=bytes2hex(sha256(read(path)))))
    end
    summary=(;passed=true,rows,
        scope="three exact retained prompts: full-vocabulary FP32 prefix logits and8greedy next tokens vs independent CPU Transformers",
        limitations=["not a full-generation equivalence proof","not an intelligence or production throughput benchmark",
            "successful recovery tests used synchronized8-token prompt chunks; prior32-token run had one device-loss event with unproven cause"],
        separate_quality_control="native CPU FP32 and Q4 produced identical failing152-token extracted repair code; see native_missing_repair_512_comparison.json")
    open(io->JSON3.pretty(io,summary),joinpath(@__DIR__,"conversation_native_parity_summary.json"),"w")
    summary
end
