# Trusted infrastructure probe, never a generated candidate. This explicitly
# selects KernelAbstractions' bundled PoCL CPU backend, not Radeon/Vulkan.
module PoCLBackendProbe
ENV["JULIA_KA_CPU_THREADS"]="2"
using KernelAbstractions, JSON3

@kernel function trusted_axpby!(y,x,a,b)
    i=@index(Global,Linear)
    if i<=length(y)
        @inbounds y[i]=a*x[i]+b*y[i]
    end
end

function probe(;output=joinpath(@__DIR__,"pocl_host_probe.json"))
    x=Float32.(1:65)./7;y=ones(Float32,65);expected=2f0.*x .- 0.5f0.*y
    started=time();error_text=nothing;passed=false
    try
        backend=KernelAbstractions.CPU()
        trusted_axpby!(backend,64)(y,x,2f0,-0.5f0;ndrange=65)
        KernelAbstractions.synchronize(backend)
        passed=isapprox(y,expected)
    catch err
        err isa InterruptException && rethrow()
        error_text=first(sprint(showerror,err),16000)
    end
    row=(;scope="trusted CPU-backend infrastructure probe outside code-generation sandbox, not a model score",
        passed,error=error_text,elapsed_seconds=time()-started,julia=string(VERSION),
        kernelabstractions=string(pkgversion(KernelAbstractions)))
    open(io->JSON3.pretty(io,row),output,"w")
    row
end
end
