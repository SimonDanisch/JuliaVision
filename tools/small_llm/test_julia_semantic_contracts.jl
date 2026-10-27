include("julia_semantic_contracts.jl")
using .JuliaSemanticContracts, Test, JSON3, SHA
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")

function test_semantic_band_contract()
    original=only(filter(t->t.id=="makie_band",JuliaTaskEval.JULIA_TASKS))
    task=JuliaSemanticContracts.semantic_task(original)
    canonical=original.reference
    reversed=replace(canonical,"band!(a,x,mu.-sigma,mu.+sigma"=>"band!(a,x,mu.+sigma,mu.-sigma")
    mutant=replace(canonical,"band!(a,x,mu.-sigma,mu.+sigma"=>"band!(a,x,mu,mu")
    @assert reversed!=canonical && mutant!=canonical
    results=Any[]
    for (name,code,expected) in (("canonical",canonical,true),("swapped_equivalent_bounds",reversed,true),("zero_width_mutant",mutant,false))
        result=JuliaTaskEval.evaluate(task,code)
        push!(results,(;name,expected,result...))
        @test result.passed==expected
        println(name," expected=",expected," observed=",result.passed);flush(stdout)
    end
    report=(;scope="evaluator qualification only; original strict model scores remain intact",
        reason="Prompt requests a band between two bounds; swapping both complete bound curves fills the same region. Public Makie docs use swapped order too.",
        original_contract_sha256=bytes2hex(sha256(original.tests)),semantic_contract_sha256=bytes2hex(sha256(task.tests)),results)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"julia_semantic_contract_validation.json"),"w")
    report
end
