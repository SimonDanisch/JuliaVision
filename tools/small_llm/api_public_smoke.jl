isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using JSON3, SHA

# Standalone public fixtures: NEVER read task.tests, task.reference or task.setup
# here. Outputs are observations, not hidden-oracle judgments.
const API_PUBLIC_FIXTURES=Dict(
    "ki_launch"=>(;category="base",setup="""
        using JSON3
        import KernelInterface as KI
        mutable struct PublicProbeBackend <: KI.Backend
            events::Vector{Any}
            synchronizations::Int
        end
        KI.max_work_group_dims(::PublicProbeBackend)=(1024,1024,64)
        KI.max_work_group_size(::KI.Kernel{PublicProbeBackend})=1024
        function KI.launch(kernel::KI.Kernel{PublicProbeBackend},groups::Dims{3},items::Dims{3},args::Tuple;kwargs...)
            isempty(kwargs) || error("unsupported low-level launch keyword arguments")
            push!(kernel.backend.events,(;groups,items,argument_count=length(args)))
            out,x=args
            out.=2 .*x
            nothing
        end
        KI.synchronize(backend::PublicProbeBackend)=(backend.synchronizations+=1;nothing)
        """,probe="""
        backend=PublicProbeBackend(Any[],0);kernel=KI.Kernel(backend,nothing)
        x=Float32[3,-2,5,-7];out=fill(NaN32,4)
        result=solve!(kernel,out,x)
        (;input=x,output=repr(out),returns_original_out=result===out,
            launches=backend.events,synchronizations=backend.synchronizations)
        """),
    "makie_update"=>(;category="makie",setup="using CairoMakie, JSON3",probe="""
        figure=Figure();axis=Axis(figure[1,1])
        plot=scatter!(axis,[2.,4.],[3.,7.];color=[0.2,0.8])
        x=[1.,3.,6.,9.];y=[4.,8.,2.,5.];colors=[0.1,0.3,0.6,0.9]
        result=solve!(plot,x,y,colors)
        (;new_x=x,new_y=y,new_colors=colors,returns_original_plot=result===plot,
            observed_positions=repr(Tuple.(plot[1][])),observed_colors=repr(plot.color[]),
            plot_count=length(axis.scene.plots))
        """)
)

function diagnose_api_public_smoke(task,code)
    ncodeunits(code)<=65536 || throw(ArgumentError("candidate too large"))
    fixture=get(API_PUBLIC_FIXTURES,task.id,nothing)
    fixture===nothing && throw(ArgumentError("no registered public API fixture"))
    checks="""
        row=try
            observation=begin
                $(fixture.probe)
            end
            (;completed=true,observation)
        catch err
            (;completed=false,error=first(sprint(showerror,err),4000))
        end
        println("PUBLIC_API_SMOKE=",JSON3.write(row))
        """
    diagnostic=(;id=task.id,category=fixture.category,setup=fixture.setup,tests=checks)
    result=JuliaTaskEval.evaluate(diagnostic,code)
    prefix="PUBLIC_API_SMOKE="
    lines=filter(line->startswith(line,prefix),split(result.diagnostic,'\n'))
    payload=length(lines)==1 ? JSON3.read(only(lines)[length(prefix)+1:end]) : nothing
    (;task=task.id,scope="standalone public-spec API fixtures; observed values/errors ONLY, no private assertions or oracle output",
        analysis_completed=result.passed&&payload!==nothing,payload,
        loading_or_harness_error=payload===nothing ? result.diagnostic : nothing,
        public_prompt_sha256=bytes2hex(sha256(task.prompt)),
        fixture_sha256=bytes2hex(sha256(fixture.setup*fixture.probe)),
        code_sha256=bytes2hex(sha256(code)),elapsed_seconds=result.elapsed_seconds)
end
