# Trusted API inspection only: no model candidate loading and no GPU device.
isdefined(@__MODULE__,:JuliaMethodCards) || include("julia_method_cards.jl")
using Makie, JSON3, SHA
import KernelInterface as KI

function snapshot_installed_methods()
    # Select the dispatch surfaces named by the public requests, rather than
    # dumping unrelated overloads or repeating overlapping source contexts.
    makie_methods=[which(Makie.update!,Tuple{Makie.Plot})]
    kernel_methods=filter(collect(methods(Core.kwcall))) do method
        parameters=Base.unwrap_unionall(method.sig).parameters
        method.module===KI && length(parameters)>=3 && parameters[3]<:KI.Kernel
    end
    isempty(makie_methods) && error("no installed Makie update methods")
    isempty(kernel_methods) && error("no installed KernelInterface keyword-call methods")
    makie=JuliaMethodCards.cards(makie_methods;root=pkgdir(Makie),context_lines=28)
    kernel=JuliaMethodCards.cards(kernel_methods;root=pkgdir(KI),context_lines=36)
    report=(;scope="trusted installed method-source context, no model generation or candidate execution",
        packages=Dict("Makie"=>string(pkgversion(Makie)),"KernelInterface"=>string(pkgversion(KI))),
        makie_update=makie,kernel_call=kernel,
        source_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"julia_method_cards.jl")))))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"installed_method_cards.json"),"w")
    report
end
