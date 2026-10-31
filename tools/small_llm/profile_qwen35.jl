using JSON3, Statistics
import Mantle, DNNKernels, KernelAbstractions

function qwen_weight_inventory(model)
    rows=Any[]
    for (name,a) in sort!(collect(model.weights);by=first)
        bytes=if a isa Qwen35Experiment.DenseWeight
            sizeof(eltype(a.data))*length(a.data)
        elseif a isa DNNKernels.KQuantMatrix
            sum(x->sizeof(eltype(x))*length(x),DNNKernels.kqarrays(a))
        else
            sizeof(eltype(a))*length(a)
        end
        # The embedding lookup reads one row. The tied output head is counted
        # separately because it scans its entire (separately quantized) matrix.
        stream_bytes=name=="embed_tokens.weight" ? 4Int(model.config.hidden_size) : bytes
        push!(rows,(;name,storage_bytes=bytes,nominal_read_bytes=stream_bytes))
    end
    (;nominal_weight_read_bytes=sum(r.nominal_read_bytes for r in rows),rows)
end

function qwen_pass_category(name)
    endswith(name,".q8") && return "activation_quantization"
    startswith(name,"lm_head") && return "output_head"
    endswith(name,".gdn") && return "delta_state"
    occursin("self_attn.attention",name) || occursin("self_attn.combine",name) ? (return "attention_cache") : nothing
    occursin("prepare_",name) && return "qk_norm_rope_cache_write"
    occursin(r"\.(qkv|z|alpha|beta|out|q|k|v|gate|up|down)$",name) && return "layer_projections"
    "normalization_elementwise_other"
end

function profile_qwen35(model,metadata;contexts=(26,4096),steps=16,output=nothing)
    profiles=Any[]
    inventory=qwen_weight_inventory(model)
    for context in contexts
        s=Qwen35Experiment.session(model;capacity=max(256,nextpow(2,context+steps+4)),profile=true)
        try
            ids=repeat(Int.(metadata.chat_ids),cld(context,length(metadata.chat_ids)))[1:context]
            Qwen35Experiment.prefill!(s,ids;chunk=32)
            Qwen35Experiment.decode!(s,4)
            KernelAbstractions.synchronize(model.device)
            before=Mantle.submissions(model.device)
            elapsed=@elapsed begin
                Qwen35Experiment.decode!(s,steps)
                KernelAbstractions.synchronize(model.device)
            end
            submissions=Mantle.submissions(model.device)-before
            passes=[(;name=t.name,kind=string(t.kind),host_ms=t.host_ms,gpu_ms=t.gpu_ms,samples=t.samples,
                       category=qwen_pass_category(t.name)) for t in Mantle.timings(s.plan)]
            categories=[(;category,gpu_ms=sum(t.gpu_ms for t in passes if t.category==category))
                for category in sort!(unique(t.category for t in passes))]
            result=(;context,steps,elapsed,profiled_tps=steps/elapsed,submissions,categories,passes)
            push!(profiles,result)
            println((;context,profiled_tps=result.profiled_tps,submissions,categories));flush(stdout)
        finally
            close(s)
        end
    end
    result=(;scope="diagnostic per-pass timestamps; instrumented throughput is not the unprofiled benchmark; summed pass costs are not frame latency",inventory,profiles)
    output===nothing || open(io->JSON3.pretty(io,result),output,"w")
    result
end
