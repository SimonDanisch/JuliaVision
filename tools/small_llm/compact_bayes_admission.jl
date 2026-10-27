isdefined(@__MODULE__,:bayes_extraction_payload)||include("bayes_extraction_admission.jl")
isdefined(@__MODULE__,:CompactBayesJSON)||include("compact_bayes_json.jl")

function admit_compact_bayes(answer)
    try
        packed=CompactBayesJSON.expand(bayes_extraction_payload(answer);max_states=256)
        (;accepted=true,reason="bounded compact CPT network",network=JSON3.read(JSON3.write(packed.network)),packed=packed.packed)
    catch err
        err isa InterruptException&&rethrow()
        (;accepted=false,reason=sprint(showerror,err),network=nothing,packed=nothing)
    end
end
