using JSON3, SHA, LinearAlgebra, Test
import Mantle, DNNKernels
isdefined(@__MODULE__,:finish_device!) || include("device_finish.jl")

function verify_native_conversations(;seconds=240,model_dir="gen/qwen35-4b",prompt_chunk=8,
        synchronize_chunks=true,case_ids=nothing)
    seconds>0 || throw(ArgumentError("time budget"))
    prompt_chunk in (8,16,32) || throw(ArgumentError("qualified prompt chunk sizes only"))
    path=joinpath(@__DIR__,"conversation_native_reference.json")
    metadata=JSON3.read(read(path,String))
    cases=filter(r->r.validate_logits,metadata.rows)
    length(cases)==3 && all(r->hasproperty(r,:reference_greedy_ids),cases) || error("CPU references incomplete")
    case_ids===nothing || (cases=filter(r->String(r.id) in case_ids,cases))
    isempty(cases) && error("no selected reference cases")
    metadata.config_sha256==bytes2hex(sha256(read(joinpath(model_dir,"config.json")))) || error("checkpoint mismatch")
    tensors=DNNKernels.readsafetensors(replace(path,".json"=>".safetensors");mmap=true)
    output=joinpath(@__DIR__,"conversation_gpu_fp32_validation_v3.json")
    Q=SmallLLMChecks.Qwen35Experiment
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;rows=Any[]
        phase="fresh Device setup completed";completed_input_tokens=0;current_case=""
        function save(complete)
            report=(;scope="full-model FP32 native-HF parity for retained 98/434/1910-token prompts; exact Julia input IDs, not an intelligence benchmark",
                complete,rows,elapsed_seconds=time()-started,phase,current_case,completed_input_tokens,
                prompt_chunk,synchronize_chunks,selected_cases=String.(getproperty.(cases,:id)),
                timing_scope="diagnostic per-chunk waits, NOT a production throughput measurement",
                reference_sha256=bytes2hex(sha256(read(path))),
                source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
                    ("verify_native_conversations.jl","qwen35.jl","prefill_qwen35.jl","grouped_attention_qwen35.jl","models.toml")))
            open(io->JSON3.pretty(io,report),output,"w")
            report
        end
        try
            save(false)
            model=Q.loadmodel(model_dir;quantize=:none,device)
            phase="FP32 model loaded";save(false)
            for case in cases
                time()<deadline-75 || return save(false)
                ids=Int.(case.julia_ids);capacity=nextpow(2,length(ids)+40)
                s=Q.session(model;capacity,kv_type=Float32,attention=:split)
                sample_started=time()
                try
                    current_case=String(case.id);completed_input_tokens=0
                    println(case.id," FP32 prefill start, ",length(ids)," tokens");flush(stdout)
                    # Known input chunks only; no generated-token readback.
                    for first in 1:prompt_chunk:length(ids)
                        time()<deadline-25 || error("bounded FP32 prefill margin reached")
                        last=min(first+prompt_chunk-1,length(ids))
                        phase="prefill $(first):$(last) pending";save(false)
                        Q.prefill!(s,ids[first:last];chunk=prompt_chunk)
                        # Diagnostic mode confines any timeout to one small
                        # known-input batch; this is not token feedback.
                        synchronize_chunks && Mantle.waitidle(device)
                        completed_input_tokens=last;phase="prefill batch completed";save(false)
                    end
                    got=Array(s.logits);want=vec(tensors[String(case.id)])
                    maxabs=maximum(abs,Float64.(got).-want)
                    relative_rms=norm(Float64.(got).-want)/norm(Float64.(want))
                    before=Mantle.submissions(device)
                    Q.decode!(s,length(case.reference_greedy_ids))
                    # Count execution submissions BEFORE the separate output
                    # transfer. Array(predictions) itself submits a readback.
                    submissions=Mantle.submissions(device)-before
                    generated=Int.(Array(s.predictions)[length(ids):length(ids)+length(case.reference_greedy_ids)-1])
                    row=(;id=case.id,input_tokens=length(ids),maxabs,relative_rms,
                        finite=all(isfinite,got),prefix_argmax=argmax(got)-1,native_prefix_argmax=argmax(want)-1,
                        generated_ids=generated,native_ids=Int.(case.reference_greedy_ids),
                        exact_greedy=generated==Int.(case.reference_greedy_ids),decode_submissions=submissions,
                        elapsed_seconds=time()-sample_started)
                    push!(rows,row);save(false)
                    @test row.finite
                    @test relative_rms<2e-4
                    @test maxabs<0.02
                    @test row.prefix_argmax==row.native_prefix_argmax
                    @test row.exact_greedy
                    @test submissions==length(generated)
                    println(case.id," native parity passed, relrms=",relative_rms);flush(stdout)
                finally
                    close(s)
                end
            end
            save(true)
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
    end
end
