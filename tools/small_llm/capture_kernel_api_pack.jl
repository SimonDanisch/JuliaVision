# CPU-only installed-source capture. No candidate, evaluator assertions or
# reference solution is read. This is a new development intervention, not an
# amendment to the frozen sampled baseline or public-error repair protocol.
import KernelInterface as KI
import Atomix, BonsaiRunner
using JSON3, SHA

function capture_kernel_api_pack()
    output=joinpath(@__DIR__,"kernel_api_pack.json")
    isfile(output) && error("API pack already captured; version future captures")
    cards=Any[]
    function card(id,path,text;selection)
        push!(cards,(;id,origin_path=path,source_sha256=bytes2hex(sha256(read(path))),
            text_sha256=bytes2hex(sha256(text)),selection,text))
    end
    indexpath=joinpath(pkgdir(KI),"src","device.jl")
    doc=Base.Docs.doc(KI.get_global_id);doc===nothing && error("missing index documentation")
    card("kernelinterface_global_id",indexpath,sprint(show,MIME"text/plain"(),doc);
        selection="complete installed get_global_id documentation")
    readme=joinpath(pkgdir(Atomix),"README.md")
    card("atomix_readme",readme,read(readme,String);selection="complete installed README")
    sugar=joinpath(pkgdir(Atomix),"src","sugar.jl")
    lines=readlines(sugar)
    lastline=findfirst(line->occursin("Atomix.@atomicswap",line),lines)
    lastline===nothing && error("unknown Atomix source structure")
    # Stop BEFORE the next documentation block, preserving the complete @atomic
    # macro + parsing helpers, not selecting a task-specific maximum expression.
    excerpt=join(lines[1:lastline-3],"\n")
    endswith(strip(excerpt),"end") || error("truncated macro implementation")
    card("atomix_atomic_implementation",sugar,excerpt;selection="complete @atomic implementation before @atomicswap docs")
    kernelspath=joinpath(pkgdir(BonsaiRunner),"src","kernels.jl")
    found=match(r"(?ms)^function copy_kernel!\(.*?^end$",read(kernelspath,String))
    found===nothing && error("missing existing generic copy example")
    card("installed_generic_copy_kernel",kernelspath,String(found.match);
        selection="complete existing copy_kernel! method, NOT an authored benchmark solution")
    report=(;scope="installed API/source add-on, chosen during development after baseline failures; no model-quality result",
        exclusion="no authored benchmark reference, private assertion, generated candidate or expected task output",
        packages=Dict(string(m)=>(;version=string(pkgversion(m)),path=pkgdir(m)) for m in (KI,Atomix,BonsaiRunner)),
        licenses=Dict("Atomix"=>read(joinpath(pkgdir(Atomix),"LICENSE"),String),
            "KernelInterface"=>read(joinpath(pkgdir(KI),"LICENSE.md"),String)),
        capture_source_sha256=bytes2hex(sha256(read(@__FILE__))),cards)
    open(io->JSON3.pretty(io,report),output,"w");report
end

function load_kernel_api_pack()
    path=joinpath(@__DIR__,"kernel_api_pack.json");packet=JSON3.read(read(path,String))
    packet.capture_source_sha256==bytes2hex(sha256(read(@__FILE__))) || error("API capture source drift")
    for card in packet.cards
        bytes2hex(sha256(read(String(card.origin_path))))==card.source_sha256 || error("installed source drift")
        bytes2hex(sha256(String(card.text)))==card.text_sha256 || error("API card text drift")
    end
    (;path,packet,text=join(("### "*String(c.id)*"\n"*String(c.text) for c in packet.cards),"\n\n"))
end
