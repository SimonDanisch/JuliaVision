# A short-context baseline through the existing Mantle/DNNKernels runtime.
# Include this file, then call bench_bonsai(path; output="results.json").
using BonsaiRunner, Mantle, KernelAbstractions, JSON3, Statistics

module SmallLLMTelemetry
include(joinpath(@__DIR__, "..", "measure.jl"))
end

function bench_bonsai(path; output=nothing, samples=5, newtokens=128)
    samples >= 3 || throw(ArgumentError("use at least three samples"))
    newtokens > 0 || throw(ArgumentError("newtokens must be positive"))
    model = Bonsai2(path; device=Mantle.Device("radeon"), context=4096)
    s = nothing
    try
        ids = encode(model.tokenizer, chatprompt(["user" =>
            "In three sentences, explain why Vulkan compute dispatches need memory barriers."];
            thinking=false))
        s = session(model)
        logits = prefill!(s, ids)
        # Compile decode once. Warm-up advances real state and stays within 4K.
        for _ in 1:32
            token = BonsaiRunner._greedy(s, logits)
            logits = BonsaiRunner.step!(s, token)
        end
        KernelAbstractions.synchronize(model.backend)
        close(s)
        s = nothing
        rows = []
        reference = nothing
        for rep in 1:samples
            s = session(model)
            prefill = @timed begin
                logits = prefill!(s, ids)
                KernelAbstractions.synchronize(model.backend)
            end
            before = SmallLLMTelemetry.gpustate()
            tokens = Int[]
            # Exactly newtokens forward steps; stop tokens are retained so
            # timing is not selected by EOS. This is decode throughput, not
            # useful answer throughput or a model-quality evaluation.
            decode_time = @timed begin
                for _ in 1:newtokens
                    token = BonsaiRunner._greedy(s, logits)
                    push!(tokens, token)
                    logits = BonsaiRunner.step!(s, token)
                end
                KernelAbstractions.synchronize(model.backend)
            end
            after = SmallLLMTelemetry.gpustate()
            finite = all(isfinite, Array(logits))
            @assert finite
            reference === nothing ? (reference = tokens) : (@assert tokens == reference)
            push!(rows, (; repetition=rep, prefill_seconds=prefill.time,
                         decode_seconds=decode_time.time, decode_gc_seconds=decode_time.gctime,
                         decode_tps=newtokens/decode_time.time, before, after,
                         finite_logits=finite, token_ids=tokens))
            close(s)
            s = nothing
        end
        plateau = median([min(r.before.sm,r.after.sm) for r in rows])
        accepted = [r for r in rows if min(r.before.sm,r.after.sm) >= 0.9plateau]
        @assert !isempty(accepted)
        result = (; checkpoint=abspath(path), julia=string(VERSION),
                   hardware="Radeon 8060S", prompt_tokens=length(ids), newtokens,
                   scope="short context; fixed length; existing Bonsai runner; no speedup comparison",
                   clock_gate_mhz=0.9plateau,
                   accepted_samples=length(accepted), rejected_samples=length(rows)-length(accepted),
                   median_decode_tps=median([r.decode_tps for r in accepted]),
                   decode_tps_range=extrema([r.decode_tps for r in accepted]),
                   decoded_sample=decode(model.tokenizer, reference), samples=rows)
        output === nothing || open(io -> JSON3.pretty(io,result),output,"w")
        result
    finally
        s === nothing || close(s)
        close(model)
    end
end
