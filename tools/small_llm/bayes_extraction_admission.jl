isdefined(@__MODULE__,:BayesJSON)||include("bayes_json.jl")
using JSON3

function bayes_extraction_payload(answer)
    ncodeunits(answer)<=65536||throw(ArgumentError("Bounded answer"))
    source=strip(String(answer))
    if startswith(source,"```")
        matchall=match(r"\A```(?:json)?\s*\n([\s\S]*?)\n```\z",source)
        matchall===nothing&&throw(ArgumentError("One whole JSON artifact required"))
        source=strip(String(matchall.captures[1]))
    end
    # Bound nesting BEFORE the JSON parser, while honoring escaped strings.
    depth=0;quoted=false;escaped=false
    for c in source
        if quoted
            if escaped;escaped=false
            elseif c=='\\';escaped=true
            elseif c=='"';quoted=false
            end
        elseif c=='"';quoted=true
        elseif c in ('{','[');depth+=1;depth<=16||throw(ArgumentError("Bounded JSON depth"))
        elseif c in ('}',']');depth-=1;depth>=0||throw(ArgumentError("Unbalanced JSON"))
        end
    end
    source
end

function admit_bayes_extraction(answer,arm)
    try
        source=bayes_extraction_payload(answer)
        if arm=="direct_probability"
            parsed=JSON3.read(source)
            parsed isa JSON3.Object&&Set(String.(keys(parsed)))==Set(["probability"])||throw(ArgumentError("One probability field"))
            p=parsed.probability
            p isa Real&&!(p isa Bool)&&isfinite(p)&&0<=p<=1||throw(ArgumentError("Numeric probability"))
            (;accepted=true,reason="numeric probability",probability=Float64(p),network=nothing,packed=nothing)
        elseif arm=="bayes_network"
            packed=BayesJSON.compile(source;max_states=256)
            (;accepted=true,reason="bounded discrete network",probability=nothing,network=JSON3.read(source),packed)
        else
            throw(ArgumentError("Unknown interface"))
        end
    catch err
        err isa InterruptException&&rethrow()
        (;accepted=false,reason=sprint(showerror,err),probability=nothing,network=nothing,packed=nothing)
    end
end

"""Semantic packing comparison: ignores node/state/parent/row ordering.
Only used privately AFTER all model answers; never supplies a corrected network.
"""
function bayes_extraction_faithful(a,b)
    function canonical(network)
        nodes=Dict{String,Any}()
        for node in network.nodes
            parents=String.(node.parents);states=String.(node.states)
            rows=Dict(Tuple(sort!(collect(zip(parents,String.(row.given)))))=>Dict(zip(states,Float64.(row.p))) for row in node.rows)
            nodes[String(node.name)]=(;parents=Set(parents),states=Set(states),rows)
        end
        (;nodes,query=String(network.query),observed=Dict(String(k)=>String(v) for (k,v) in pairs(network.observed)))
    end
    x,y=canonical(a),canonical(b)
    x.query==y.query&&x.observed==y.observed&&Set(keys(x.nodes))==Set(keys(y.nodes))||return false
    for key in keys(x.nodes)
        u,v=x.nodes[key],y.nodes[key]
        u.parents==v.parents&&u.states==v.states&&Set(keys(u.rows))==Set(keys(v.rows))||return false
        for row in keys(u.rows),state in u.states
            isapprox(u.rows[row][state],v.rows[row][state];atol=1e-6,rtol=0)||return false
        end
    end
    true
end
