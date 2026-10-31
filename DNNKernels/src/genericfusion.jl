"""
    patternfusion(graph) -> (graph, report)

The weight-independent pattern rewrites, in the order `Model` runs them: SwiGLU
into one op, a SwiGLU into the product that reads it, and grouped RMS norms.
Handwritten semantic graphs run the same function before [`genericfusion`](@ref).
"""
function patternfusion(g::Graph)
    g,s=FUSESWIGLU[] ? fuseswiglu(g) : (g,0)
    g,p=FUSESWIGLUMM[] ? fuseswiglumm(g) : (g,0)
    g,r=fusegroupedrms(g)
    g,(;swiglu=s,swiglu_products=p,grouped_rms=r)
end
function patternfusion(graphs::AbstractDict)
    out=Dict{String,Any}();s=p=r=0
    for (name,g) in graphs
        if g isa Graph
            g,report=patternfusion(g)
            s+=report.swiglu;p+=report.swiglu_products;r+=report.grouped_rms
        end
        out[name]=g
    end
    out,(;swiglu=s,swiglu_products=p,grouped_rms=r)
end

"""
    genericfusion(graph) -> (graph, report)

The weight-independent fusion pipeline, shared by imported Models and
handwritten semantic graphs. Custom semantic ops participate as normal graph
nodes: their declared inputs/outputs protect observable values and bound the
patterns the generic passes may absorb. Unknown operations are never guessed
to be elementwise or pure.
"""
function genericfusion(g::Graph)
    g,n=fuseops(g)
    g,m=FUSEATTENTION[] ? fusemaskedattention(g) : (g,0)
    g,e=foldepilogue(g)
    g,p=foldpremap(g)
    # Last: a residual add the passes above left alone is a plain binary add.
    g,r=foldresidual(g)
    g,(;elementwise=n,masked_attention=m,epilogues=e,premaps=p,residuals=r)
end
function genericfusion(graphs::AbstractDict)
    # `Model` takes a `Dict{String,Graph}`, as `fuseops` always returned.
    out=Dict{String,Graph}();n=m=e=p=r=0
    for (name,g) in graphs
        g,report=genericfusion(g);out[name]=g
        n+=report.elementwise;m+=report.masked_attention;e+=report.epilogues
        p+=report.premaps;r+=report.residuals
    end
    out,(;elementwise=n,masked_attention=m,epilogues=e,premaps=p,residuals=r)
end
