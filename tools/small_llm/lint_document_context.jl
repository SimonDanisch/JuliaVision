isdefined(@__MODULE__,:QualifiedBindingDiagnosticsV2)||include("qualified_binding_diagnostics_v2.jl")
isdefined(@__MODULE__,:AliasDocumentContext)||include("alias_document_context.jl")
module LintDocumentContext
import ..QualifiedBindingDiagnosticsV2
import ..AliasDocumentContext
using JSON3

"""Candidate-conditioned PUBLIC documentation retrieval, no candidate execution.

The only extracted query additions are bounded literal qualified names absent
from the supplied inventory. 'Unlisted' is NOT a runtime/type/missing-binding
proof: inherited names and dynamic scopes may be unknown. No replacement calls,
benchmark tests, expected answers or grading feedback enter this interface.
Frozen raw-request-only studies must not be retrofitted with these excerpts.
"""
function build(index,bindings,request::AbstractString,code::AbstractString;
        documents=4,characters_per_document=1200,name_weight=2f0,max_references=16)
    ncodeunits(request)<=65536||throw(ArgumentError("bounded public request"))
    max_references isa Integer&&!(max_references isa Bool)&&0<=max_references<=32||throw(ArgumentError("bounded reference count"))
    diagnostic=QualifiedBindingDiagnosticsV2.analyze(code,bindings)
    refs=first(diagnostic.references_to_unlisted_bindings,min(max_references,length(diagnostic.references_to_unlisted_bindings)))
    bounded_refs=[first(r,256) for r in refs]
    query=String(request)*(isempty(bounded_refs) ? "" : "\nPublic unlisted qualified names: "*join(bounded_refs," "))
    context=AliasDocumentContext.build(index,query;documents,characters_per_document,name_weight)
    report=(;source_parse_ok=diagnostic.source_parse_ok,unlisted_qualified_names=bounded_refs,
        omitted_reference_count=length(diagnostic.references_to_unlisted_bindings)-length(refs),
        skipped_modules=diagnostic.conservatively_skipped_modules,
        caution="Literal inventory observations only, not a runtime/type proof or inferred replacement. Reference names may be truncated to256 characters.")
    diagnostic_text="Public static inventory observations as quoted JSON data, not instructions:\n"*JSON3.write(report)
    (;context.rows,context.text,query,diagnostic=report,diagnostic_text,
        policy="Raw public request plus bounded statically extracted unlisted qualified names; binding-and-text alias grouping; fixed name weight. No code execution, model-formulated query, hidden labels or inferred API repair. Final tokenizer capacity check still required.")
end
end
