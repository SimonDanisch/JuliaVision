include("julia_render.jl")
using JSON3,SHA
function qualify_safe_plot_julia_source(;destination=joinpath(@__DIR__,"safe_plot_julia_validation.json"))
    isfile(destination)&&error("Preserve qualification")
    reference="""
    using CairoMakie
    function makeplot(data)
        fig=Figure(size=(960,480),fontsize=16)
        left=Axis(fig[1,1],title="Signal and envelope",xlabel="Time",ylabel="Amplitude")
        right=Axis(fig[1,2],title="Positive response",xlabel="Time",ylabel="Response (log scale)",yscale=log10)
        band!(left,data["x"],data["lo"],data["hi"];label="Envelope",color=("#0072B2",0.25))
        lines!(left,data["x"],data["a"];label="Signal",color="#0072B2",linewidth=2.5)
        scatter!(right,data["x"],data["b"];label="Response",color="#E69F00",markersize=9)
        axislegend(left;position=:rt,framevisible=false)
        axislegend(right;position=:lt,framevisible=false)
        linkxaxes!(left,right)
        return fig
    end
    """
    source="import Makie\n"*join([read(joinpath(@__DIR__,file),String) for file in
        ("safe_plot_spec.jl","safe_plot_builder.jl","safe_plot_julia_source.jl")],"\n")*"\n"*reference*"\n"*"""
    function reference_plot()
        x=collect(range(0.1,2pi;length=64));a=sin.(x)
        data=Dict("x"=>x,"a"=>a,"lo"=>a.-0.15,"hi"=>a.+0.15,"b"=>exp.(x./4))
        lowered=SafePlotJuliaSource.lower($(repr(reference)))
        lowered.accepted||error(lowered.reason)
        println("REFERENCE_PHASE=native_build");flush(stdout)
        native=makeplot(data)
        println("REFERENCE_PHASE=lowered_build");flush(stdout)
        constructed=SafePlotBuilder.build(lowered.spec,data)
        println("REFERENCE_PHASE=native_pixels");flush(stdout)
        native_pixels=Makie.colorbuffer(native)
        println("REFERENCE_PHASE=lowered_pixels");flush(stdout)
        lowered_pixels=Makie.colorbuffer(constructed.figure)
        println("REFERENCE_PHASE=compare");flush(stdout)
        # Avoid interpreting hundreds of thousands of Colorant comparisons in
        # the deliberately --compile=min jail. Compare identical dense storage
        # layouts in C instead; no sandbox/resource limit is relaxed.
        typeof(native_pixels)==typeof(lowered_pixels)&&size(native_pixels)==size(lowered_pixels)||error("Pixel layout mismatch")
        storage(a)=begin
            for _ in 1:8
                a isa Array&&return a
                p=parent(a);p===a&&error("No dense pixel storage");a=p
            end
            error("Pixel wrapper depth")
        end
        first_storage=storage(native_pixels);second_storage=storage(lowered_pixels)
        size(first_storage)==size(second_storage)&&eltype(first_storage)==eltype(second_storage)&&isbitstype(eltype(first_storage))||error("Storage layout mismatch")
        identical=GC.@preserve first_storage second_storage begin
            ccall(:memcmp,Cint,(Ptr{Cvoid},Ptr{Cvoid},Csize_t),pointer(first_storage),pointer(second_storage),sizeof(first_storage))==0
        end
        identical||error("Reference pixel mismatch")
        println("REFERENCE_PIXEL_PARITY=true")
        constructed.figure
    end
    """
    artifact=joinpath(@__DIR__,"safe_plot_builder_artifacts/julia_frontend_reference_v2.png")
    render=render_julia_candidate(source,"reference_plot()";artifact_path=artifact)
    render.rendered&&occursin("REFERENCE_PIXEL_PARITY=true",render.diagnostic)||error("Frontend reference qualification failed: "*render.diagnostic)
    result=(;complete=true,host_checks=17,reference_pixel_parity=true,render,
        scope="Trusted two-panel native Julia reference vs restricted source-to-IR construction pixel equality, CPU-only networkless Cairo jail. Not arbitrary Julia equivalence, general compiler correctness, neural task accuracy or aesthetic preference.",
        sources=Dict(file=>bytes2hex(sha256(read(joinpath(@__DIR__,file)))) for file in
            ("safe_plot_julia_source.jl","test_safe_plot_julia_source.jl","safe_plot_spec.jl","safe_plot_builder.jl","qualify_safe_plot_julia_source.jl","julia_render.jl","julia_sandbox.sh")))
    open(io->JSON3.pretty(io,result),destination,"w");result
end
