using JSON3
import DNNKernels as AuditDK

const RUNNER_GRAPH_PLACEHOLDERS = ["DeepFilterRunner","DemucsRunner","FluxKleinRunner",
    "ProPainterRunner","QwenImageEditRunner","ZImageRunner"]

"""Structural ratchet, not inference verification or a proof about every path.

New implemented runners must expose their DNN graph route. A placeholder that
gets an implementation must leave the explicit placeholder inventory too.
Custom neural paths still require their numerical/semantic GPU regressions.
"""
function assert_runner_graph_routes(runners)
    for r in runners
        expected=!(r.name in RUNNER_GRAPH_PLACEHOLDERS)
        r.has_model_or_semantic_graph==expected || error(
            "Runner graph route needs inspection: $(r.name); expected implemented=$expected")
    end
    true
end

# Network declarations of the experimental Qwen frontends. Selection, token
# feedback and logit hooks follow `emitsemantic!` in the same Mantle graph.
const EXPERIMENTAL_FRONTENDS=("qwen35.jl"=>"session","prefill_qwen35.jl"=>"record_prefill",
    "qwen36_moe_model.jl"=>"session")

"""Each experimental network builder declares a semantic graph, never a plain
Mantle graph, and emits it through the shared passes."""
function experimental_frontends(dir)
    map(EXPERIMENTAL_FRONTENDS) do (file,builder)
        src=read(joinpath(dir,file),String)
        start=findfirst("function $builder(",src)
        start===nothing && error("No $builder in $file")
        body=src[first(start):end]
        stop=findfirst(r"\nend\n",body)
        body=body[1:first(stop)]
        semantic=occursin("DK.SemanticGraph(",body) && occursin("emitsemantic!",body)
        direct=occursin(r"Mantle\.Graph\(",body)
        semantic && !direct || error("$file $builder does not declare a DNNKernels semantic graph")
        (;file,builder,semantic_graph=semantic,plain_mantle_graph=direct,
         standard_ops=count("standardop!",body)+count("linear!(",body)+count("linear_batch!(",body))
    end
end

"""Check the shared pipeline against every available exported neural graph.

This is a structural compiler check, not a claim of model inference parity.
The handwritten exceptions are classified separately instead of mistaking
image processing/sampling passes for bypassed neural networks.
"""
function audit_runner_graphs(;root=normpath(joinpath(@__DIR__,"../..")),output)
    ispath(output) && error("Refusing to overwrite $output")
    graphs=Any[]
    directories=unique(realpath(p) for p in (joinpath(root,"gen/graphs"),normpath(joinpath(root,"../../gen/graphs"))) if isdir(p))
    for directory in directories
        for (dir,_,files) in walkdir(directory),file in files
            endswith(file,".json") || continue
            path=joinpath(dir,file)
            raw=JSON3.read(read(path,String))
            all(k->haskey(raw,k),(:buffers,:ops,:inputs,:outputs)) || continue
            raw.buffers isa AbstractVector || continue
            all(b->b isa AbstractDict && haskey(b,:kind),raw.buffers) || continue
            raw.ops isa AbstractVector || continue
            g=AuditDK.loadgraph(path)
            optimized,report=AuditDK.genericfusion(g)
            # Pin the exact pre-extraction pass order and all graph fields.
            old,_=AuditDK.fuseops(g)
            AuditDK.FUSEATTENTION[] && ((old,_)=AuditDK.fusemaskedattention(old))
            old,_=AuditDK.foldepilogue(old);old,_=AuditDK.foldpremap(old)
            old,_=AuditDK.foldresidual(old)
            @assert repr(old)==repr(optimized)
            # The shared pattern stage is Model's former sequence of three passes.
            patterns,_=AuditDK.patternfusion(g)
            sequential,_=AuditDK.fuseswiglu(g);sequential,_=AuditDK.fuseswiglumm(sequential)
            sequential,_=AuditDK.fusegroupedrms(sequential)
            @assert repr(patterns)==repr(sequential)
            push!(graphs,(;path,ops_before=length(g.ops),ops_after=length(optimized.ops),report))
            println("Checked generic fusion: ",path);flush(stdout)
        end
    end
    @assert !isempty(graphs)
    placeholders=RUNNER_GRAPH_PLACEHOLDERS
    neural_exceptions=Dict(
        "BonsaiRunner"=>"Decode and prefill use the shared semantic DNN graph frontend; packed numerical lowerings retained",
        "Trellis2Runner"=>"Flow/transformer and sparse decoder use DNN graphs; typed sparse-gather/product/normalization lowerings retained",
        "OIDNRunner"=>"Both standard and packed-HWC neural paths now use semantic DNN graphs; specialized packed-layout lowerings retained")
    runners=Any[]
    for name in sort(readdir(root))
        endswith(name,"Runner") && isdir(joinpath(root,name,"src")) || continue
        paths=String[]
        for (dir,_,fs) in walkdir(joinpath(root,name,"src")),file in fs
            endswith(file,".jl") && push!(paths,joinpath(dir,file))
        end
        src=join(read.(paths,String),'\n')
        status=name in placeholders ? "not ported; no inference path" : get(neural_exceptions,name,"Neural network uses DNNKernels Model/Graph")
        push!(runners,(;name,status,has_model_or_semantic_graph=occursin(r"Model\(|emitgraph!|DNNKernels.Graph|SemanticGraph",src),
            has_direct_dispatch=occursin(r"kilaunch!|dispatch!",src)))
    end
    assert_runner_graph_routes(runners)
    frontends=experimental_frontends(@__DIR__)
    result=(;complete=true,scope="Implemented runners and experimental Qwen frontends declare DNNKernels graphs",runners,
        experimental_frontends=frontends,exported_graphs=graphs,graph_checks=length(graphs),pending=String[],
        note="Direct dispatch matches also include legitimate image preprocessing, sampling, meshing and custom DNN op lowerings")
    open(io->JSON3.pretty(io,result),output,"w");result
end
