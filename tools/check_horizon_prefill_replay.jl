# Run with the workspace Julia project. The caller may supply an already loaded
# `prefill_model` to avoid loading the 32B weights twice.
using HorizonRunner, DNNKernels, Mantle, KernelAbstractions, SHA

function prefill_snapshot(outputs)
    map(outputs) do output
        host = Array(output)
        (; shape=size(host), finite=all(isfinite, host),
           digest=bytes2hex(sha256(reinterpret(UInt8, vec(host)))))
    end
end

function check_prefill_replay(m)
    @assert m.prefill == 512
    @assert !m.model.record
    k, v = newcache(m)
    # Fill the entire exported prompt bucket with valid, deterministic token IDs.
    ids = reshape(Int64[mod(i, 1000) + 1 for i in 0:511], 512, 1)
    slots = collect(Int64, 0:511)
    args = (DNNKernels.toback(m.backend, ids), k, v,
            DNNKernels.toback(m.backend, slots), causalmask(m, slots))
    reset_inputs!() = begin
        fill!(k, 0); fill!(v, 0)
        KernelAbstractions.synchronize(m.backend)
    end
    println("REFERENCE_START prefill=", m.prefill, " maxlen=", m.maxlen)
    flush(stdout)
    reset_inputs!()
    reference_time = @elapsed begin
        outputs = DNNKernels.call(m.model, "horizon32b_prefill", args...; dims=(;))
        KernelAbstractions.synchronize(m.backend)
    end
    reference = prefill_snapshot(outputs)
    println("REFERENCE_DONE seconds=", reference_time, " outputs=", reference)
    @assert all(x -> x.finite, reference)
    flush(stdout)
    reset_inputs!()
    println("RECORD_START"); flush(stdout)
    record_time = @elapsed recorded = DNNKernels.recordplan(m.model,
        m.model.graphs["horizon32b_prefill"], "horizon32b_prefill", args,
        (;), false, DNNKernels.RandomNoise())
    println("RECORD_DONE seconds=", record_time,
            " passes=", length(recorded.plan.passes),
            " maxpasses=", recorded.plan.record_maxpasses,
            " submissions=", recorded.plan.recording isa Mantle.RecordingSequence ?
                length(recorded.plan.recording.parts) : 1)
    flush(stdout)
    for iteration in 1:2
        reset_inputs!()
        println("REPLAY_START iteration=", iteration); flush(stdout)
        elapsed = @elapsed begin
            Mantle.run!(recorded.plan)
            Mantle.waitidle(Mantle.vk_context(m.backend))
        end
        actual = prefill_snapshot(recorded.outputs)
        println("REPLAY_DONE iteration=", iteration, " seconds=", elapsed,
                " exact=", actual == reference, " outputs=", actual)
        flush(stdout)
        @assert actual == reference
    end
    nothing
end

if !isdefined(Main, :prefill_model)
    prefill_model = horizon32b(dir=joinpath(@__DIR__, "..", "gen", "graphs", "horizon32b"),
                              record=false)
end
check_prefill_replay(prefill_model)
