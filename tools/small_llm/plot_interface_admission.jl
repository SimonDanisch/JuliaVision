isdefined(@__MODULE__,:SafePlotJuliaSource)||include("safe_plot_julia_source.jl")

"""Same formatting normalization for both interfaces; never pick fragments."""
function plot_interface_payload(answer,arm)
    arm in ("julia_source","plot_spec")||throw(ArgumentError("Unknown interface"))
    ncodeunits(answer)<=32768||throw(ArgumentError("Bounded answer"))
    text=strip(String(answer))
    if startswith(text,"```")
        matched=match(r"\A```([A-Za-z]*)\s*\n(.*?)\n```\z"s,text)
        matched===nothing&&throw(ArgumentError("One complete fenced artifact only"))
        language=matched.captures[1]
        language in ("",arm=="julia_source" ? "julia" : "json")||throw(ArgumentError("Wrong artifact language"))
        text=strip(matched.captures[2])
    end
    String(text)
end

function admit_plot_interface(answer,arm)
    try
        payload=plot_interface_payload(answer,arm)
        if arm=="julia_source"
            SafePlotJuliaSource.lower(payload)
        else
            (;accepted=true,reason="admitted declarative plot specification",spec=SafePlotSpec.parse_spec(payload))
        end
    catch error
        error isa InterruptException&&rethrow()
        (;accepted=false,reason=sprint(showerror,error),spec=nothing)
    end
end
