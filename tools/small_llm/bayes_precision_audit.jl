include("paired_workflow_uncertainty.jl")
using JSON3,SHA,Dates
const BAYES_AUDIT_TOLERANCES=(1e-5,1e-4,1e-3,1e-2)

function bayes_precision_statistics(primary)
    primary.complete&&length(primary.rows)==16||throw(ArgumentError("Complete ALL16 first"))
    strict=paired_summary_uncertainty(primary,"direct_probability","bayes_network")
    rows=primary.rows;tasks=sort!(unique(String(r.task) for r in rows));length(tasks)==8||throw(ArgumentError("Complete8 pairs"))
    at(row,tolerance)=row.probability!==nothing&&isfinite(row.probability)&&abs(row.probability-row.expected_probability)<=tolerance
    curve=map(BAYES_AUDIT_TOLERANCES) do tolerance
        direct=[only(filter(r->r.task==t&&r.arm=="direct_probability",rows)) for t in tasks]
        network=[only(filter(r->r.task==t&&r.arm=="bayes_network",rows)) for t in tasks]
        before=[at(r,tolerance) for r in direct];numeric=[at(r,tolerance) for r in network]
        faithful=[at(r,tolerance)&&r.network_faithful===true for r in network]
        (;absolute_tolerance=tolerance,direct_numeric=count(identity,before),network_numeric=count(identity,numeric),
            network_faithful_and_numeric=count(identity,faithful),
            paired_faithful_workflow=paired_workflow_uncertainty(before,faithful),
            paired_numeric_only=paired_workflow_uncertainty(before,numeric))
    end
    errors=map(("direct_probability","bayes_network")) do arm
        selected=filter(r->r.arm==arm,rows)
        measured=[abs(r.probability-r.expected_probability) for r in selected if r.probability!==nothing]
        (;arm,total=length(selected),measured=length(measured),unavailable=length(selected)-length(measured),
            mean_absolute_error=isempty(measured) ? nothing : sum(measured)/length(measured),
            max_absolute_error=isempty(measured) ? nothing : maximum(measured),
            faithful_networks=count(r->r.network_faithful===true,selected))
    end
    (;strict_primary_comparison=strict,curve,errors,
        caveat="Secondary uniform precision diagnostic, NOT changed primary scores or selected best threshold. Solver arithmetic accuracy is not problem-extraction faithfulness; numeric-only success can occur for an unfaithful network. Fixed8 authored tasks, extra solver compute; no clinical/frontier/calibration claim.")
end

function freeze_bayes_precision_audit(;directory=joinpath(@__DIR__,"bayes_extraction_results"))
    path=joinpath(directory,"precision_audit_conditions.json")
    isfile(path)&&error("Preserve registration")
    isfile(joinpath(directory,"summary.json"))&&error("Register BEFORE any private scores")
    digest(path)=bytes2hex(sha256(read(path)))
    primary=joinpath(directory,"conditions.json");p=JSON3.read(read(primary,String))
    length(p.jobs)==16&&length(unique(j.id for j in p.jobs))==16||error("Complete frozen cohort required")
    result=(;registered_utc=string(now(UTC)),primary_conditions_sha256=digest(primary),
        scope="Secondary PRE-GRADE diagnostic registered AFTER generation began, not original primary registration. No private grades seen; all fixed tolerance levels retained.",
        absolute_tolerances=collect(BAYES_AUDIT_TOLERANCES),
        existing_generation_sha256=Dict(j.id=>digest(joinpath(directory,j.id*".json")) for j in p.jobs if isfile(joinpath(directory,j.id*".json"))),
        source_sha256=digest(@__FILE__),statistics_source_sha256=digest(joinpath(@__DIR__,"paired_workflow_uncertainty.jl")))
    open(io->JSON3.pretty(io,result),path,"w");result
end

function audit_bayes_precision(;directory=joinpath(@__DIR__,"bayes_extraction_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    registration=joinpath(directory,"precision_audit_conditions.json");p=JSON3.read(read(registration,String))
    p.source_sha256==digest(@__FILE__)&&p.statistics_source_sha256==digest(joinpath(@__DIR__,"paired_workflow_uncertainty.jl"))||error("Audit source drift")
    p.primary_conditions_sha256==digest(joinpath(directory,"conditions.json"))||error("Primary drift")
    collect(p.absolute_tolerances)==collect(BAYES_AUDIT_TOLERANCES)||error("Tolerance drift")
    primarypath=joinpath(directory,"summary.json");primary=JSON3.read(read(primarypath,String))
    primary.conditions_sha256==p.primary_conditions_sha256||error("Score provenance")
    for (id,hash) in pairs(p.existing_generation_sha256)
        digest(joinpath(directory,String(id)*".json"))==hash||error("Pre-grade generation drift")
    end
    for row in primary.rows
        digest(joinpath(directory,row.id*".json"))==row.generation_sha256||error("Scored generation drift")
    end
    result=(;complete=true,bayes_precision_statistics(primary)...,registration_scope=p.scope,
        primary_summary_sha256=digest(primarypath),registration_sha256=digest(registration))
    open(io->JSON3.pretty(io,result),joinpath(directory,"precision_audit.json"),"w");result
end
